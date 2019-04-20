# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
""" Custom Helpers
    - Contains custom decoder helper functions for the policy model (Token Based)
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.models.layers.beam_decoder import BeamHelper
from diplomacy_research.models.layers.decoder import MaskedInputs, LARGE_NEGATIVE
from diplomacy_research.models.policy.base_policy_model import TRAINING_DECODER, GREEDY_DECODER, SAMPLE_DECODER
from diplomacy_research.models.state_space import VOCABULARY_SIZE, GO_ID
from diplomacy_research.utils.tensorflow import Helper
from diplomacy_research.utils.tensorflow import _transpose_batch_time
from diplomacy_research.utils.tensorflow import _unstack_ta
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import SparseTensor
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import gen_array_ops
from diplomacy_research.utils.tensorflow import control_flow_ops
from diplomacy_research.utils.tensorflow import embedding_lookup
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import sparse_ops
from diplomacy_research.utils.tensorflow import categorical
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import rnn_cell_impl


def _ensure_int64(tensor):
    """ Casts the tensor to tf.int64 """
    if isinstance(tensor, ops.Tensor) and tensor.dtype != dtypes.int64:
        tensor = math_ops.cast(tensor, dtypes.int64)
    return tensor

def _slice_mask(mask, slicing, to_float=True, squeeze=False, time_major=False):
    """ Returns a sliced mask given the current position and the prev token
        :param mask: [SparseTensor] Mask to apply at each time step -- Size: (b, dec_len, vocab_size, vocab_size)
        :param slicing: The slicing mask to apply.  (-1 means keep entire dim, scalar only keeps the specified index)
        :param to_float: Boolean that indicates we also want to parse to float.
        :param squeeze: Boolean. If true removes the 1 dim and converts to DenseTensor.
        :param time_major: Indicates that the mask was time_major (dec_len, b, vocab_size, vocab_size)
        :return: The sliced tensor
    """
    if time_major:
        slicing = [slicing[1], slicing[0]] + slicing[2:]
    dims = array_ops.unstack(mask.dense_shape)
    start = [0 if item == -1 else _ensure_int64(item) for item in slicing]
    end = [_ensure_int64(dims[index]) if item == -1 else 1 for index, item in enumerate(slicing)]
    squeeze_dims = [slice_ix for slice_ix, slice_item in enumerate(slicing) if slice_item != -1]
    dense_shape = [None, None, VOCABULARY_SIZE, VOCABULARY_SIZE]
    squeezed_shape = [dense_shape[slice_ix] for slice_ix, slice_item in enumerate(slicing) if slice_item == -1]

    sliced_mask = sparse_ops.sparse_slice(mask, start, end)
    if to_float or squeeze:
        sliced_mask = math_ops.cast(sliced_mask, dtypes.float32)
    if squeeze:
        sliced_mask = sparse_ops.sparse_reduce_sum(sliced_mask, axis=squeeze_dims)
        sliced_mask.set_shape(squeezed_shape)
    return sliced_mask

def _get_embedding_fn(embedding):
    """ Returns a callable embedding function """
    return embedding if callable(embedding) else (lambda ids: embedding_lookup(embedding, ids))


# ----------------------------------------
#            TOKEN-BASED HELPER
# ----------------------------------------
class CustomHelper(Helper):
    """ A custom helper that works with teacher forcing, greedy, and sampling.
        Concatenates inputs, then applies a linear layer.
        Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere.
    """
    def __init__(self, decoder_type, inputs, embedding, sequence_length, mask, input_layer=None, time_major=False,
                 softmax_temperature=None, seed=None, name=None):
        """ Constructor
            :param decoder_type: An uint8 representing TRAINING_DECODER, GREEDY_DECODER, or SAMPLE_DECODER
            :param inputs: The decoder input (b, dec_len)
            :param embedding: The embedding vector
            :param sequence_length: The length of each input (b,)
            :param mask: [SparseTensor] Mask to apply at each time step -- Size: (b, dec_len, vocab_size, vocab_size)
            :param input_layer: Optional. A layer to apply on the inputs
            :param time_major: If true indicates that the first dimension is time, otherwise it is batch size
            :param softmax_temperature: Optional. Softmax temperature. None or size: (batch_size,)
            :param seed: Optional. The sampling seed
            :param name: Optional scope name.
        """
        # pylint: disable=too-many-arguments
        with ops.name_scope(name, "CustomHelper", [inputs, sequence_length, embedding]):
            assert isinstance(mask, SparseTensor), 'The mask must be a SparseTensor'
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            self._mask = mask
            self._time_major = time_major
            self._embedding_fn = embedding if callable(embedding) else lambda ids: embedding_lookup(embedding, ids)
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)
            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._decoder_type = decoder_type
            self._sequence_length = ops.convert_to_tensor(sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError("Expected vector for sequence_length. Shape: %s" % self._sequence_length.get_shape())
            self._input_layer = input_layer if callable(input_layer) else lambda x: x
            self._batch_size = array_ops.size(sequence_length)
            self._start_inputs = gen_array_ops.fill([self._batch_size], GO_ID)
            self._softmax_temperature = softmax_temperature
            self._seed = seed
            self.vocab_size = VOCABULARY_SIZE
            self._zero_inputs = \
                MaskedInputs(inputs=array_ops.zeros_like(self._input_layer(self._embedding_fn(self._start_inputs))),
                             mask=_slice_mask(self._mask,
                                              slicing=[-1, 0, GO_ID, -1],
                                              squeeze=True,
                                              time_major=self._time_major))

            # Preventing div by zero
            # Adding an extra dim to the matrix, so we can broadcast with the outputs shape
            if softmax_temperature is not None:
                self._softmax_temperature = gen_math_ops.maximum(1e-10, self._softmax_temperature)
                if self._softmax_temperature.get_shape().ndims == 1:
                    self._softmax_temperature = self._softmax_temperature[:, None]

    @property
    def batch_size(self):
        """ Returns the batch size """
        return self._batch_size

    @property
    def sample_ids_shape(self):
        """ Returns the shape of the sample ids """
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        """ Returns the dtype of the sample ids """
        return dtypes.int32

    def sample(self, time, outputs, state, name=None):
        """ Samples the id for the next time step (or -1 for teacher forcing) """
        with ops.name_scope(name, 'CustomHelperSample', [time, outputs, state]):

            def training():
                """ Selecting training / teacher forcing """
                fill_op = gen_array_ops.fill([array_ops.shape(outputs)[0]], -1)
                with ops.control_dependencies([fill_op]):
                    return array_ops.identity(fill_op)

            def greedy():
                """ Selecting greedy """
                argmax_op = math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)
                with ops.control_dependencies([argmax_op]):
                    return array_ops.identity(argmax_op)

            def sample():
                """ Sampling """
                logits = outputs if self._softmax_temperature is None else outputs / self._softmax_temperature
                sample_id_sampler = categorical.Categorical(logits=logits)
                sample_op = sample_id_sampler.sample(seed=self._seed)
                with ops.control_dependencies([sample_op]):
                    return array_ops.identity(sample_op)

            return control_flow_ops.case([(gen_math_ops.equal(self._decoder_type, TRAINING_DECODER), training),
                                          (gen_math_ops.equal(self._decoder_type, GREEDY_DECODER), greedy),
                                          (gen_math_ops.equal(self._decoder_type, SAMPLE_DECODER), sample)],
                                         default=training)

    def initialize(self, name=None):
        """ Performs helper initialization (to get initial state) """
        with ops.name_scope(name, 'CustomHelperInitialize'):
            finished = gen_math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)

            def training_inputs():
                """ Returns the training initial input """
                embed_op = self._embedding_fn(self._input_tas.read(0))
                with ops.control_dependencies([embed_op]):
                    return array_ops.identity(embed_op)

            def start_inputs():
                """ Returns the GO_ID initial input """
                embed_op = self._embedding_fn(self._start_inputs)
                with ops.control_dependencies([embed_op]):
                    return array_ops.identity(embed_op)

            # Getting initial inputs
            initial_inputs = control_flow_ops.case(
                [(gen_math_ops.equal(self._decoder_type, TRAINING_DECODER), training_inputs),
                 (gen_math_ops.equal(self._decoder_type, GREEDY_DECODER), start_inputs),
                 (gen_math_ops.equal(self._decoder_type, SAMPLE_DECODER), start_inputs)],
                default=training_inputs)

            next_inputs = \
                control_flow_ops.cond(all_finished,
                                      lambda: self._zero_inputs,
                                      lambda: MaskedInputs(inputs=self._input_layer(initial_inputs),
                                                           mask=_slice_mask(self._mask,
                                                                            slicing=[-1, 0, GO_ID, -1],
                                                                            squeeze=True,
                                                                            time_major=self._time_major)))
            return (finished, next_inputs)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """ Computes the next inputs at a time step """
        with ops.name_scope(name, 'CustomHelperNextInputs', [time, outputs, state, sample_ids]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)

            def get_next_inputs():
                """ Retrieves the inputs for the next time step """
                def get_training_inputs():
                    """ Selecting training inputs """
                    read_op = self._input_tas.read(next_time)
                    with ops.control_dependencies([read_op]):
                        return array_ops.identity(read_op)

                def get_sample_inputs():
                    """ Selecting greedy/sample inputs """
                    return sample_ids

                inputs_next_step = control_flow_ops.case(
                    [(gen_math_ops.equal(self._decoder_type, TRAINING_DECODER), get_training_inputs),
                     (gen_math_ops.equal(self._decoder_type, GREEDY_DECODER), get_sample_inputs),
                     (gen_math_ops.equal(self._decoder_type, SAMPLE_DECODER), get_sample_inputs)],
                    default=get_training_inputs)
                inputs_emb_next_step = self._input_layer(self._embedding_fn(inputs_next_step))

                # Applying mask
                # inputs_one_hot:   (b, 1, VOC, 1)
                # mask_t:           (b, 1, VOC, VOC)
                # next_mask:        (b, VOC)        -- DenseTensor
                inputs_one_hot = array_ops.one_hot(inputs_next_step, self.vocab_size)[:, None, :, None]
                mask_t = _slice_mask(self._mask, [-1, next_time, -1, -1], time_major=self._time_major)
                next_mask = sparse_ops.sparse_reduce_sum(inputs_one_hot * mask_t, axis=[1, 2])
                next_mask = gen_math_ops.minimum(next_mask, 1.)
                next_mask.set_shape([None, self.vocab_size])

                # Prevents this branch from executing eagerly
                with ops.control_dependencies([inputs_emb_next_step, next_mask]):
                    return MaskedInputs(inputs=array_ops.identity(inputs_emb_next_step),
                                        mask=array_ops.identity(next_mask))

            next_inputs = control_flow_ops.cond(all_finished,
                                                true_fn=lambda: self._zero_inputs,
                                                false_fn=get_next_inputs)

            # Returning
            return (finished, next_inputs, state)

class CustomBeamHelper(BeamHelper):
    """ A helper to feed data in a custom beam search decoder """

    def __init__(self, cell, embedding, mask, sequence_length, initial_state, beam_width, input_layer=None,
                 output_layer=None, time_major=False):
        """ Initialize the CustomBeamHelper
            :param cell: An `RNNCell` instance.
            :param embedding: The embedding vector
            :param mask: [SparseTensor] Mask to apply at each time step -- Size: (b, dec_len, vocab_size, vocab_size)
            :param sequence_length: The length of each input (b,)
            :param initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            :param beam_width: Python integer, the number of beams.
            :param input_layer: Optional. A layer to apply on the inputs
            :param output_layer: Optional. An instance of `tf.layers.Layer`, i.e., `tf.layers.Dense`. Optional layer
                                 to apply to the RNN output prior to storing the result or sampling.
            :param time_major: If true indicates that the first dimension is time, otherwise it is batch size.
        """
        # pylint: disable=super-init-not-called,too-many-arguments
        rnn_cell_impl.assert_like_rnncell('cell', cell)                                                                 # pylint: disable=protected-access
        assert isinstance(mask, SparseTensor), 'The mask must be a SparseTensor'
        assert isinstance(beam_width, int), 'beam_width should be a Python integer'

        self._sequence_length = ops.convert_to_tensor(sequence_length, name='sequence_length')
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError("Expected vector for sequence_length. Shape: %s" % self._sequence_length.get_shape())

        self._cell = cell
        self._embedding_fn = _get_embedding_fn(embedding)
        self._mask = mask
        self._time_major = time_major
        self.vocab_size = VOCABULARY_SIZE
        self._input_layer = input_layer if input_layer is not None else lambda x: x
        self._output_layer = output_layer

        self._input_size = embedding.shape[-1]
        if input_layer is not None:
            self._input_size = self._input_layer.compute_output_shape([None, self._input_size])[-1]

        self._batch_size = array_ops.size(sequence_length)
        self._start_tokens = gen_array_ops.fill([self._batch_size * beam_width], GO_ID)
        self._end_token = -1
        self._beam_width = beam_width
        self._initial_cell_state = nest.map_structure(self._maybe_split_batch_beams,
                                                      initial_state,
                                                      self._cell.state_size)
        self._finished = array_ops.one_hot(array_ops.zeros([self._batch_size], dtype=dtypes.int32),
                                           depth=self._beam_width,
                                           on_value=False,
                                           off_value=True,
                                           dtype=dtypes.bool)

        # zero_mask is (batch, beam, vocab_size)
        self._zero_mask = _slice_mask(self._mask, slicing=[-1, 0, GO_ID, -1], squeeze=True, time_major=self._time_major)
        self._zero_mask = gen_array_ops.tile(array_ops.expand_dims(self._zero_mask, axis=1), [1, self._beam_width, 1])
        self._zero_inputs = \
            MaskedInputs(
                inputs=array_ops.zeros_like(
                    self._split_batch_beams(
                        self._input_layer(self._embedding_fn(self._start_tokens)), self._input_size)),
                mask=self._zero_mask)

    @property
    def beam_width(self):
        """ Returns the beam width """
        return self._beam_width

    @property
    def batch_size(self):
        """ Returns the batch size """
        return self._batch_size

    @property
    def output_size(self):
        """ Returns the size of the RNN output """
        size = self._cell.output_size
        if self._output_layer is None:
            return size

        # To use layer's compute_output_shape, we need to convert the RNNCell's output_size entries into shapes
        # with an unknown batch size.  We then pass this through the layer's compute_output_shape and read off
        # all but the first (batch) dimensions to get the output size of the rnn with the layer applied to the top.
        output_shape_with_unknown_batch = \
            nest.map_structure(lambda shape: tensor_shape.TensorShape([None]).concatenate(shape), size)
        layer_output_shape = self._output_layer.compute_output_shape(output_shape_with_unknown_batch)
        return nest.map_structure(lambda shape: shape[1:], layer_output_shape)

    def initialize(self):
        """ Initialize the beam helper - Called in beam_decoder.initialize()
            :return: `(finished, start_inputs, initial_cell_state)`.
        """
        finished, zero_inputs, zero_mask = self._finished, self._zero_inputs, self._zero_mask
        all_finished = math_ops.reduce_all(gen_math_ops.equal(0, self._sequence_length))
        initial_inputs = self._embedding_fn(self._start_tokens)

        # Start Inputs
        start_inputs = control_flow_ops.cond(all_finished,
                                             lambda: zero_inputs,
                                             lambda: MaskedInputs(
                                                 inputs=self._split_batch_beams(self._input_layer(initial_inputs),
                                                                                self._input_size),
                                                 mask=zero_mask))

        # Returning
        return finished, start_inputs, self._initial_cell_state

    def step(self, time, inputs, cell_state):
        """ Performs a step using the beam search cell
            :param time: The current time step (scalar)
            :param inputs: A (structure of) input tensors.
            :param state: A (structure of) state tensors and TensorArrays.
            :return: `(cell_outputs, next_cell_state)`.
        """
        raw_inputs = inputs
        inputs, output_mask = raw_inputs.inputs, raw_inputs.mask
        inputs = nest.map_structure(lambda inp: self._merge_batch_beams(inp, depth_shape=inp.shape[2:]), inputs)
        cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state, self._cell.state_size)
        cell_outputs, next_cell_state = self._cell(inputs, cell_state)                  # [batch * beam, out_sz]
        next_cell_state = nest.map_structure(self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

        # Splitting outputs and applying mask
        # cell_outputs is [batch, beam, vocab_size]
        cell_outputs = self._output_layer(cell_outputs) if self._output_layer is not None else cell_outputs
        cell_outputs = nest.map_structure(lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
        cell_outputs = gen_math_ops.add(cell_outputs, (1. - output_mask) * LARGE_NEGATIVE)

        # Returning
        return cell_outputs, next_cell_state

    def next_inputs(self, time, inputs, beam_search_output, beam_search_state):
        """ Computes the inputs at the next time step given the beam outputs
            :param time: The current time step (scalar)
            :param inputs: A (structure of) input tensors.
            :param beam_search_output: The output of the beam search step
            :param beam_search_state: The state after the beam search step
            :return: `(beam_search_output, next_inputs)`
            :type beam_search_output: beam_search_decoder.BeamSearchDecoderOutput
            :type beam_search_state: beam_search_decoder.BeamSearchDecoderState
        """
        next_time = time + 1
        all_finished = math_ops.reduce_all(next_time >= self._sequence_length)
        sample_ids = beam_search_output.predicted_ids

        def get_next_inputs():
            """ Retrieves the inputs for the next time step """
            inputs_next_step = sample_ids
            inputs_emb_next_step = self._input_layer(self._embedding_fn(inputs_next_step))  # [bat, beam, in_sz]

            # Applying mask
            # inputs_one_hot:   (batch, beam,   1, VOC,   1)
            # mask_t:           (batch,    1,   1, VOC, VOC)
            # next_mask:        (batch, beam, VOC)
            inputs_one_hot = array_ops.one_hot(inputs_next_step, self.vocab_size)[:, :, None, :, None]
            mask_t = sparse_ops.sparse_tensor_to_dense(_slice_mask(self._mask, [-1, next_time, -1, -1],
                                                                   time_major=self._time_major))[:, None, :, :, :]
            mask_t.set_shape([None, 1, 1, self.vocab_size, self.vocab_size])
            next_mask = math_ops.reduce_sum(inputs_one_hot * mask_t, axis=[2, 3])
            next_mask = gen_math_ops.minimum(next_mask, 1.)

            # Prevents this branch from executing eagerly
            with ops.control_dependencies([inputs_emb_next_step, next_mask]):
                return MaskedInputs(inputs=array_ops.identity(inputs_emb_next_step),
                                    mask=array_ops.identity(next_mask))

        # Getting next inputs
        next_inputs = control_flow_ops.cond(all_finished,
                                            true_fn=lambda: self._zero_inputs,
                                            false_fn=get_next_inputs)

        # Returning
        return beam_search_output, next_inputs
