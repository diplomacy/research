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
    - Contains custom decoder helper functions for the policy model (Order Based)
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.models.layers.decoder import CandidateInputs, LARGE_NEGATIVE
from diplomacy_research.models.layers.beam_decoder import BeamHelper
from diplomacy_research.models.policy.base_policy_model import TRAINING_DECODER, GREEDY_DECODER, SAMPLE_DECODER
from diplomacy_research.models.state_space import GO_ID
from diplomacy_research.utils.tensorflow import Helper
from diplomacy_research.utils.tensorflow import _transpose_batch_time
from diplomacy_research.utils.tensorflow import _unstack_ta
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import gen_array_ops
from diplomacy_research.utils.tensorflow import beam_search_decoder
from diplomacy_research.utils.tensorflow import control_flow_ops
from diplomacy_research.utils.tensorflow import embedding_lookup
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import categorical
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import rnn_cell_impl


def _get_embedding_fn(embedding):
    """ Returns a callable embedding function """
    return embedding if callable(embedding) else (lambda ids: embedding_lookup(embedding, ids))

class CustomHelper(Helper):
    """ A custom helper that work with teacher forcing, greedy, and sampling.
        Takes inputs then applies a linear layer.
        Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere.
    """
    def __init__(self, decoder_type, inputs, order_embedding, candidate_embedding, sequence_length, candidates,
                 input_layer=None, time_major=False, softmax_temperature=None, seed=None, name=None):
        """ Constructor
            :param decoder_type: An uint8 representing TRAINING_DECODER, GREEDY_DECODER, or SAMPLE_DECODER
            :param inputs: The decoder input (b, dec_len)
            :param order_embedding: The order embedding vector
            :param candidate_embedding: The candidate embedding vector
            :param sequence_length: The length of each input (b,)
            :param candidates: The candidates at each time step -- Size: (b, nb_cand, max_candidates)
            :param input_layer: Optional. A layer to apply on the inputs
            :param time_major: If true indicates that the first dimension is time, otherwise it is batch size
            :param softmax_temperature: Optional. Softmax temperature. None, scalar, or size: (batch_size,)
            :param seed: Optional. The sampling seed
            :param name: Optional scope name.
        """
        # pylint: disable=too-many-arguments
        with ops.name_scope(name, "CustomHelper", [inputs, sequence_length, order_embedding, candidate_embedding]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            candidates = ops.convert_to_tensor(candidates, name="candidates")
            self._inputs = inputs
            self._order_embedding_fn = _get_embedding_fn(order_embedding)
            self._candidate_embedding_fn = _get_embedding_fn(candidate_embedding)
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)
                candidates = nest.map_structure(_transpose_batch_time, candidates)
            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._candidate_tas = nest.map_structure(_unstack_ta, candidates)
            self._decoder_type = decoder_type
            self._sequence_length = ops.convert_to_tensor(sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError("Expected vector for sequence_length. Shape: %s" % self._sequence_length.get_shape())
            self._input_layer = input_layer if input_layer is not None else lambda x: x
            self._batch_size = array_ops.size(sequence_length)
            self._start_inputs = gen_array_ops.fill([self._batch_size], GO_ID)
            self._softmax_temperature = softmax_temperature
            self._seed = seed

            # Compute input shape
            self._zero_inputs = \
                CandidateInputs(inputs=
                                array_ops.zeros_like(self._input_layer(self._order_embedding_fn(self._start_inputs))),
                                candidates=array_ops.zeros_like(candidates[0, :]),
                                candidates_emb=array_ops.zeros_like(self._candidate_embedding_fn(candidates[0, :])))

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
        """ Samples the id for the next time step (or -1 for teacher forcing)
            Note: outputs is a tuple of (cell_outputs, candidate)
        """
        cell_outputs, candidate = outputs

        with ops.name_scope(name, 'CustomHelperSample', [time, outputs, state]):

            def training():
                """ Selecting training / teacher forcing """
                fill_op = gen_array_ops.fill([array_ops.shape(cell_outputs)[0]], -1)
                with ops.control_dependencies([fill_op]):
                    return array_ops.identity(fill_op)

            def greedy():
                """ Selecting greedy """
                argmax_id = math_ops.cast(math_ops.argmax(cell_outputs, axis=-1), dtypes.int32)
                nb_candidate = array_ops.shape(candidate)[1]
                candidate_ids = \
                    math_ops.reduce_sum(array_ops.one_hot(argmax_id, nb_candidate, dtype=dtypes.int32) * candidate,
                                        axis=-1)
                with ops.control_dependencies([candidate_ids]):
                    return array_ops.identity(candidate_ids)

            def sample():
                """ Sampling """
                logits = cell_outputs if self._softmax_temperature is None else cell_outputs / self._softmax_temperature
                sample_id_sampler = categorical.Categorical(logits=logits)
                sample_ids = sample_id_sampler.sample(seed=self._seed)
                nb_candidate = array_ops.shape(candidate)[1]
                reduce_op = math_ops.reduce_sum(array_ops.one_hot(sample_ids,
                                                                  nb_candidate,
                                                                  dtype=dtypes.int32) * candidate, axis=-1)
                with ops.control_dependencies([reduce_op]):
                    return array_ops.identity(reduce_op)

            return control_flow_ops.case([(gen_math_ops.equal(self._decoder_type, TRAINING_DECODER), training),
                                          (gen_math_ops.equal(self._decoder_type, GREEDY_DECODER), greedy),
                                          (gen_math_ops.equal(self._decoder_type, SAMPLE_DECODER), sample)],
                                         default=training)

    def initialize(self, name=None):
        """ Performs helper initialization (to get initial state) """
        with ops.name_scope(name, 'CustomHelperInitialize'):
            finished = gen_math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            initial_candidates = self._candidate_tas.read(0)

            def training_inputs():
                """ Returns the training initial input """
                embed_op = self._order_embedding_fn(self._input_tas.read(0))
                with ops.control_dependencies([embed_op]):
                    return array_ops.identity(embed_op)

            def start_inputs():
                """ Returns the GO_ID initial input """
                embed_op = self._order_embedding_fn(self._start_inputs)
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
                                      lambda: CandidateInputs(
                                          inputs=self._input_layer(initial_inputs),
                                          candidates=initial_candidates,
                                          candidates_emb=self._candidate_embedding_fn(initial_candidates)))
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
                inputs_emb_next_step = self._input_layer(self._order_embedding_fn(inputs_next_step))
                candidate_next_step = self._candidate_tas.read(next_time)
                candidate_emb_next_step = self._candidate_embedding_fn(candidate_next_step)

                # Prevents this branch from executing eagerly
                with ops.control_dependencies([inputs_emb_next_step, candidate_next_step, candidate_emb_next_step]):
                    return CandidateInputs(inputs=array_ops.identity(inputs_emb_next_step),
                                           candidates=array_ops.identity(candidate_next_step),
                                           candidates_emb=array_ops.identity(candidate_emb_next_step))

            next_inputs = control_flow_ops.cond(all_finished,
                                                true_fn=lambda: self._zero_inputs,
                                                false_fn=get_next_inputs)

            # Returning
            return (finished, next_inputs, state)


class CustomBeamHelper(BeamHelper):
    """ A helper to feed data in a custom beam search decoder """

    def __init__(self, cell, order_embedding, candidate_embedding, candidates, sequence_length, initial_state,
                 beam_width, input_layer=None, output_layer=None, time_major=False):
        """ Initialize the CustomBeamHelper
            :param cell: An `RNNCell` instance.
            :param order_embedding: The order embedding vector  - Size: (batch, ord_emb_size)
            :param candidate_embedding: The candidate embedding vector - Size: (batch, cand_emb_size)
            :param candidates: The candidates at each time step -- Size: (batch, nb_cand, max_candidates)
            :param sequence_length: The length of each sequence (batch,)
            :param initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            :param beam_width: Python integer, the number of beams.
            :param input_layer: Optional. A layer to apply on the inputs
            :param output_layer: Optional. An instance of `tf.layers.Layer`, i.e., `tf.layers.Dense`. Optional layer
                                 to apply to the RNN output prior to storing the result or sampling.
            :param time_major: If true indicates that the first dimension is time, otherwise it is batch size.
        """
        # pylint: disable=super-init-not-called,too-many-arguments
        rnn_cell_impl.assert_like_rnncell('cell', cell)                                                                 # pylint: disable=protected-access
        assert isinstance(beam_width, int), 'beam_width should be a Python integer'

        self._sequence_length = ops.convert_to_tensor(sequence_length, name='sequence_length')
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError("Expected vector for sequence_length. Shape: %s" % self._sequence_length.get_shape())

        candidates = ops.convert_to_tensor(candidates, name='candidates')
        candidates = nest.map_structure(_transpose_batch_time, candidates) if not time_major else candidates

        self._cell = cell
        self._order_embedding_fn = _get_embedding_fn(order_embedding)
        self._candidate_embedding_fn = _get_embedding_fn(candidate_embedding)
        self._candidate_tas = nest.map_structure(_unstack_ta, candidates)
        self._input_layer = input_layer if input_layer is not None else lambda x: x
        self._output_layer = output_layer

        self._input_size = order_embedding.shape[-1]
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

        # Compute input shape
        self._zero_inputs = \
            CandidateInputs(inputs=
                            array_ops.zeros_like(self._split_batch_beams(
                                self._input_layer(self._order_embedding_fn(self._start_tokens)),
                                self._input_size)),
                            candidates=array_ops.zeros_like(candidates[0, :]),
                            candidates_emb=array_ops.zeros_like(self._candidate_embedding_fn(candidates[0, :])))

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
        finished, zero_inputs = self._finished, self._zero_inputs
        all_finished = math_ops.reduce_all(gen_math_ops.equal(0, self._sequence_length))
        initial_inputs = self._order_embedding_fn(self._start_tokens)
        initial_candidates = self._candidate_tas.read(0)

        # Start Inputs
        start_inputs = control_flow_ops.cond(all_finished,
                                             lambda: zero_inputs,
                                             lambda: CandidateInputs(
                                                 inputs=self._split_batch_beams(self._input_layer(initial_inputs),
                                                                                self._input_size),
                                                 candidates=initial_candidates,
                                                 candidates_emb=self._candidate_embedding_fn(initial_candidates)))

        return finished, start_inputs, self._initial_cell_state

    def step(self, time, inputs, cell_state):
        """ Performs a step using the beam search cell
            :param time: The current time step (scalar)
            :param inputs: A (structure of) input tensors.
            :param state: A (structure of) state tensors and TensorArrays.
            :return: `(cell_outputs, next_cell_state)`.
        """
        raw_inputs = inputs
        inputs, candidates, candidates_emb = raw_inputs.inputs, raw_inputs.candidates, raw_inputs.candidates_emb

        inputs = nest.map_structure(lambda inp: self._merge_batch_beams(inp, depth_shape=inp.shape[2:]), inputs)
        cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state, self._cell.state_size)
        cell_outputs, next_cell_state = self._cell(inputs, cell_state)                  # [batch * beam, out_sz]
        next_cell_state = nest.map_structure(self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

        # Splitting outputs and adding a bias dimension
        # cell_outputs is [batch, beam, cand_emb_size + 1]
        cell_outputs = self._output_layer(cell_outputs) if self._output_layer is not None else cell_outputs
        cell_outputs = nest.map_structure(lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
        cell_outputs = array_ops.pad(cell_outputs, [(0, 0), (0, 0), (0, 1)], constant_values=1.)

        # Computing candidates
        # cell_outputs is reshaped to   [batch, beam,        1, cand_emb_size + 1]
        # candidates_emb is reshaped to [batch,    1, max_cand, cand_emb_size + 1]
        # output_mask is                [batch,    1, max_cand]
        # cell_outputs is finally       [batch, beam, max_cand]
        cell_outputs = math_ops.reduce_sum(array_ops.expand_dims(cell_outputs, axis=2)
                                           * array_ops.expand_dims(candidates_emb, axis=1), axis=-1)
        output_mask = math_ops.cast(array_ops.expand_dims(gen_math_ops.greater(candidates, 0), axis=1), dtypes.float32)
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

        # Sampling
        next_word_ids = beam_search_output.predicted_ids
        candidates = inputs.candidates
        nb_candidates = array_ops.shape(candidates)[1]
        sample_ids = math_ops.reduce_sum(array_ops.one_hot(next_word_ids, nb_candidates, dtype=dtypes.int32)
                                         * array_ops.expand_dims(candidates, axis=1), axis=-1)

        def get_next_inputs():
            """ Retrieves the inputs for the next time step """
            inputs_next_step = sample_ids
            inputs_emb_next_step = self._input_layer(self._order_embedding_fn(inputs_next_step))
            candidate_next_step = self._candidate_tas.read(next_time)
            candidate_emb_next_step = self._candidate_embedding_fn(candidate_next_step)

            # Prevents this branch from executing eagerly
            with ops.control_dependencies([inputs_emb_next_step, candidate_next_step, candidate_emb_next_step]):
                return CandidateInputs(inputs=array_ops.identity(inputs_emb_next_step),
                                       candidates=array_ops.identity(candidate_next_step),
                                       candidates_emb=array_ops.identity(candidate_emb_next_step))

        # Getting next inputs
        next_inputs = control_flow_ops.cond(all_finished,
                                            true_fn=lambda: self._zero_inputs,
                                            false_fn=get_next_inputs)

        # Rewriting beam search output with the correct sample ids
        beam_search_output = beam_search_decoder.BeamSearchDecoderOutput(scores=beam_search_output.scores,
                                                                         predicted_ids=sample_ids,
                                                                         parent_ids=beam_search_output.parent_ids)

        # Returning
        return beam_search_output, next_inputs
