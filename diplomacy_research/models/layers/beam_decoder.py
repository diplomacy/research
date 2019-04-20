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
""" Diverse Beam Search Decoder
    - Contains a custom implementation of a beam search
"""
# Source: https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder
from abc import ABCMeta, abstractmethod
import math
import sys
import numpy as np
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import gen_array_ops
from diplomacy_research.utils.tensorflow import beam_search_decoder
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import nn_ops
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import tensor_array_ops
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import tensor_util


def _check_maybe(tensor):
    """ Checks that the tensor has a known rank. """
    if tensor.shape.ndims is None:
        raise ValueError('Expected tensor (%s) to have known rank, but ndims == None.' % tensor)

class BeamHelper(metaclass=ABCMeta):
    """ A helper to feed data in a custom beam search decoder """

    @property
    def beam_width(self):
        """ Returns the beam width """
        raise NotImplementedError()

    @property
    def batch_size(self):
        """ Returns the batch size """
        raise NotImplementedError()

    @property
    def output_size(self):
        """ Returns the size of the RNN output """
        raise NotImplementedError()

    @abstractmethod
    def initialize(self):
        """ Initialize the beam helper - Called in beam_decoder.initialize()
            :return: `(finished, start_inputs, initial_cell_state)`.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, time, inputs, cell_state):
        """ Performs a step using the beam search cell
            :param time: The current time step (scalar)
            :param inputs: A (structure of) input tensors.
            :param state: A (structure of) state tensors and TensorArrays.
            :return: `(cell_outputs, next_cell_state)`.
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    def _merge_batch_beams(self, tensor, depth_shape=None):
        """ Merges the tensor from a batch of beam into a batch by beams
            More exactly, `tensor` is a tensor of dimension [batch, beam, depth_shape].
            We reshape this into [batch * beam, depth_shape]

            :param tensor: Tensor of dimension [batch_size, beam_width, depth_shape]
            :param depth_shape: (Possibly known) depth shape.
            :return: A reshaped version of tensor with dimension [batch_size * beam_width, depth_shape]
        """
        if isinstance(depth_shape, ops.Tensor):
            depth_shape = tensor_shape.as_shape(tensor_util.constant_value(depth_shape))
        else:
            depth_shape = tensor_shape.TensorShape(depth_shape)
        tensor_full_shape = array_ops.shape(tensor)
        static_batch_size = tensor_util.constant_value(self.batch_size)
        batch_size_beam_width = None if static_batch_size is None else static_batch_size * self.beam_width
        reshaped_tensor = gen_array_ops.reshape(tensor,
                                                array_ops.concat(([self.batch_size * self.beam_width],
                                                                  tensor_full_shape[2:]),
                                                                 0))
        reshaped_tensor.set_shape((tensor_shape.TensorShape([batch_size_beam_width]).concatenate(depth_shape)))
        return reshaped_tensor

    def _split_batch_beams(self, tensor, depth_shape=None):
        """ Splits the tensor from a batch by beams into a batch of beams.
            More exactly, `tensor` is a tensor of dimension [batch_size * beam_width, depth_shape].
            We reshape this into [batch_size, beam_width, depth_shape]

            :param tensor: Tensor of dimension [batch_size * beam_width, depth_shape]
            :param depth_shape: (Possible known) depth shape.
            :return: A reshaped version of tensor with dimension [batch_size, beam_width, depth_shape]
        """
        if isinstance(depth_shape, ops.Tensor):
            depth_shape = tensor_shape.TensorShape(tensor_util.constant_value(depth_shape))
        else:
            depth_shape = tensor_shape.TensorShape(depth_shape)
        tensor_full_shape = array_ops.shape(tensor)
        reshaped_tensor = gen_array_ops.reshape(tensor,
                                                array_ops.concat(([self.batch_size, self.beam_width],
                                                                  tensor_full_shape[1:]),
                                                                 0))
        static_batch_size = tensor_util.constant_value(self.batch_size)
        expected_reshaped_shape = tensor_shape.TensorShape([static_batch_size,
                                                            self.beam_width]).concatenate(depth_shape)
        if not reshaped_tensor.shape.is_compatible_with(expected_reshaped_shape):
            raise ValueError('Unexpected behavior when reshaping between beam width and batch size.  The reshaped '
                             'tensor has shape: %s. We expected it to have shape (batch_size, beam_width, depth) == %s.'
                             'Perhaps you forgot to create a zero_state with batch_size=encoder_batch_size * beam_width'
                             '?' % (reshaped_tensor.shape, expected_reshaped_shape))
        reshaped_tensor.set_shape(expected_reshaped_shape)
        return reshaped_tensor

    def _maybe_split_batch_beams(self, tensor, depth_shape):
        """ Maybe splits the tensor from a batch by beams into a batch of beams
            We do this so that we can use nest and not run into problems with shapes.

            :param tensor: `Tensor`, either scalar or shaped `[batch_size * beam_width] + depth_shape`.
            :param depth_shape: `Tensor`, Python int, or `TensorShape`.
            :return: If `tensor` is a matrix or higher order tensor, then the return value is `tensor` reshaped
                     to [batch_size, beam_width] + depth_shape. Otherwise, `tensor` is returned unchanged.
        """
        if isinstance(tensor, tensor_array_ops.TensorArray):
            return tensor
        _check_maybe(tensor)
        if tensor.shape.ndims >= 1:
            return self._split_batch_beams(tensor, depth_shape)
        return tensor

    def _maybe_merge_batch_beams(self, tensor, depth_shape):
        """ Splits the tensor from a batch by beams into a beam of beams
            More exactly, `tensor` is a tensor of dimension `[batch_size * beam_width] + depth_shape`, then
            we reshape it to `[batch_size, beam_width] + depth_shape`

            :param tensor: `Tensor` of dimension `[batch_size * beam_width] + depth_shape`.
            :param depth_shape: `Tensor`, Python int, or `TensorShape`.
            :return: A reshaped version of `tensor` with shape `[batch_size, beam_width] + depth_shape`
        """
        if isinstance(tensor, tensor_array_ops.TensorArray):
            return tensor
        _check_maybe(tensor)
        if tensor.shape.ndims >= 2:
            return self._merge_batch_beams(tensor, depth_shape)
        return tensor


class DiverseBeamSearchDecoder(beam_search_decoder.BeamSearchDecoder):
    """ Diverse BeamSearch sampling decoder.

        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in `AttentionWrapper`,
        then you must ensure that:

        - The encoder output has been tiled to `beam_width` via @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
        - The `batch_size` argument passed to the `zero_state` method of this wrapper is equal to
          `true_batch_size * beam_width`.
        - The initial state created with `zero_state` above contains a `cell_state` value containing properly tiled
          final state from the encoder.

        An example:
        ```
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(sequence_length, multiplier=beam_width)
            attention_mechanism = MyFavoriteAttentionMechanism(num_units=attention_depth,
                                                               memory=tiled_inputs,
                                                               memory_sequence_length=tiled_sequence_length)
            attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
            decoder_initial_state = attention_cell.zero_state(dtype, batch_size=true_batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
        ```
    """
    def __init__(self, beam_helper, sequence_length, nb_groups=5, similarity_penalty=1., reorder_tensor_arrays=True):
        """ Initialize the BeamSearchDecoder
            :param beam_helper: A `BeamHelper` class to feed data in the BeamSearchDecoder
            :param sequence_length: The length of each sequence (batch,)
            :param nb_groups: Python integer. The number of groups to use (for diverse beam search - 1610.02424)
            :param similarity_penalty: The penalty to apply to tokens similar to previous groups (ArXiV 1610.02424)
            :param reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell state will be reordered
                                        according to the beam search path. If the `TensorArray` can be reordered, the
                                        stacked form will be returned. Otherwise, the `TensorArray` will be returned as
                                        is. Set this flag to `False` if the cell state contains `TensorArray`s that are
                                        not amenable to reordering.
            :type beam_helper: BeamHelper
        """
        # pylint: disable=super-init-not-called,too-many-arguments
        assert isinstance(nb_groups, int), 'nb_groups should be a Python integer'

        self._sequence_length = ops.convert_to_tensor(sequence_length, name='sequence_length')
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError("Expected vector for sequence_length. Shape: %s" % self._sequence_length.get_shape())

        # Getting beam width and batch size from helper
        self._beam_width = beam_helper.beam_width
        self._batch_size = beam_helper.batch_size

        # Finding group splits
        nb_groups = min(max(nb_groups, 1), self._beam_width)        # Cannot have more groups than beams
        group_ids = [group_id % nb_groups for group_id in range(self._beam_width)]
        self._group_splits = [len([item for item in group_ids if item == x]) for x in range(nb_groups)]

        # Setting variables
        self._beam_helper = beam_helper
        self._similarity_penalty = similarity_penalty
        self._reorder_tensor_arrays = reorder_tensor_arrays
        self._end_token = -1

    @property
    def output_size(self):
        """ Returns the cell output and the id """
        return beam_search_decoder.BeamSearchDecoderOutput(scores=tensor_shape.TensorShape([self._beam_width]),
                                                           predicted_ids=tensor_shape.TensorShape([self._beam_width]),
                                                           parent_ids=tensor_shape.TensorShape([self._beam_width]))

    @property
    def output_dtype(self):
        """ Returns the cell dtype """
        return beam_search_decoder.BeamSearchDecoderOutput(scores=nest.map_structure(lambda _: dtypes.float32,
                                                                                     self._rnn_output_size()),
                                                           predicted_ids=dtypes.int32,
                                                           parent_ids=dtypes.int32)

    def _rnn_output_size(self):
        """ Computes the RNN cell size """
        return self._beam_helper.output_size

    def initialize(self, name=None):
        """ Initialize the decoder.
            :param name: Name scope for any created operations.
            :return: `(finished, start_inputs, initial_state)`.
        """
        finished, start_inputs, initial_cell_state = self._beam_helper.initialize()

        # Initial state
        dtype = nest.flatten(initial_cell_state)[0].dtype
        log_probs = array_ops.one_hot(array_ops.zeros([self._batch_size], dtype=dtypes.int32),  # (batch_sz, beam_sz)
                                      depth=self._beam_width,
                                      on_value=ops.convert_to_tensor(0.0, dtype=dtype),
                                      off_value=ops.convert_to_tensor(-np.Inf, dtype=dtype),
                                      dtype=dtype)
        initial_state = beam_search_decoder.BeamSearchDecoderState(cell_state=initial_cell_state,
                                                                   log_probs=log_probs,
                                                                   finished=finished,
                                                                   lengths=array_ops.zeros([self._batch_size,
                                                                                            self._beam_width],
                                                                                           dtype=dtypes.int64),
                                                                   accumulated_attention_probs=())

        # Returning
        return finished, start_inputs, initial_state

    def step(self, time, inputs, state, name=None):
        """ Performs a decoding step.
            :param time: scalar `int32` tensor.
            :param inputs: A (structure of) input tensors.
            :param state: A (structure of) state tensors and TensorArrays.
            :param name: Name scope for any created operations.
            :return: `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, 'BeamSearchDecoderStep', (time, inputs, state)):
            # Stepping through beam helper
            cell_state = state.cell_state
            cell_outputs, next_cell_state = self._beam_helper.step(time, inputs, cell_state)

            # Performing beam search step
            beam_search_output, beam_search_state = _beam_search_step(time=time,
                                                                      logits=cell_outputs,
                                                                      group_splits=self._group_splits,
                                                                      next_cell_state=next_cell_state,
                                                                      beam_state=state,
                                                                      batch_size=self._batch_size,
                                                                      beam_width=self._beam_width,
                                                                      similarity_penalty=self._similarity_penalty,
                                                                      sequence_length=self._sequence_length)

            # Computing next inputs
            beam_search_output, next_inputs = self._beam_helper.next_inputs(time,
                                                                            inputs,
                                                                            beam_search_output,
                                                                            beam_search_state)

        # Returning
        return beam_search_output, beam_search_state, next_inputs, beam_search_state.finished

def _beam_search_step(time, logits, group_splits, next_cell_state, beam_state, batch_size, beam_width,
                      similarity_penalty, sequence_length):
    """ Performs a single step of Beam Search Decoding.
        :param time: Beam search time step, should start at 0. At time 0 we assume that all beams are equal and
                     consider only the first beam for continuations.
        :param logits: Logits at the current time step. A tensor of shape `[batch_size, beam_width, vocab_size]`
        :param group_splits: The size of each group (e.g. [2, 2, 2, 1, 1] for splitting beam of 8 in 5 groups)
        :param next_cell_state: The next state from the cell, e.g. AttentionWrapperState if the cell is attentional.
        :param beam_state: Current state of the beam search. An instance of `BeamSearchDecoderState`.
        :param batch_size: The batch size for this input.
        :param beam_width: Python int. The size of the beams.
        :param similarity_penalty: The penalty to apply to scores to account for similarity to previous beams.
        :param sequence_length: The length of each input (b,)
        :return: A new beam state.
    """
    # pylint: disable=too-many-arguments
    time = ops.convert_to_tensor(time, name='time')
    static_batch_size = tensor_util.constant_value(batch_size)

    # Calculate the current lengths of the predictions
    prediction_lengths = beam_state.lengths
    previously_finished = beam_state.finished

    # Calculate the total log probs for the new hypotheses
    # Final Shape: [batch_size, beam_width, vocab_size]
    step_log_probs = nn_ops.log_softmax(logits)
    step_log_probs = _mask_probs(step_log_probs, 0, previously_finished)
    total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs

    # Calculate the continuation lengths by adding to all continuing beams.
    vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
    lengths_to_add = gen_math_ops.cast(gen_math_ops.less(time + 1, sequence_length), dtypes.int64)
    lengths_to_add = gen_array_ops.tile(lengths_to_add[:, None, None], multiples=[1, 1, vocab_size])
    add_mask = math_ops.cast(gen_math_ops.logical_not(previously_finished), dtypes.int64)
    lengths_to_add *= array_ops.expand_dims(add_mask, 2)
    new_prediction_lengths = lengths_to_add + array_ops.expand_dims(prediction_lengths, 2)

    # Calculate the scores for each beam
    scores = _get_scores(log_probs=total_probs, sequence_lengths=new_prediction_lengths)
    scores_flat = gen_array_ops.reshape(scores, [batch_size, -1])

    # Adjusting for similarity (arXiv:1610.02424 - Diverse Beam Search)
    next_beam_scores = array_ops.zeros(shape=[batch_size, 0], dtype=dtypes.float32)
    word_indices = array_ops.zeros(shape=[batch_size, 0], dtype=dtypes.int32)
    next_word_ids = array_ops.zeros(shape=[batch_size, 0], dtype=dtypes.int32, name='next_beam_word_ids')
    next_beam_ids = array_ops.zeros(shape=[batch_size, 0], dtype=dtypes.int32, name='next_beam_parent_ids')

    # For each group, selecting the top `group_size` candidates, and penalizing those candidates in future groups
    for group_size in group_splits:
        t_group_size = ops.convert_to_tensor(group_size, dtype=dtypes.int32)

        # Computing the best candidates for the current group
        next_group_scores, group_indices = nn_ops.top_k(scores_flat, k=t_group_size)
        next_group_scores.set_shape([static_batch_size, group_size])
        group_indices.set_shape([static_batch_size, group_size])

        # Storing the best scores and the indices
        next_beam_scores = array_ops.concat([next_beam_scores, next_group_scores], axis=1)
        word_indices = array_ops.concat([word_indices, group_indices], axis=1)

        # Decoding the selected positions in the group
        group_next_word_ids = math_ops.cast(math_ops.mod(group_indices, vocab_size), dtypes.int32)
        group_next_beam_ids = math_ops.cast(group_indices / vocab_size, dtypes.int32)
        next_word_ids = array_ops.concat([next_word_ids, group_next_word_ids], axis=1)
        next_beam_ids = array_ops.concat([next_beam_ids, group_next_beam_ids], axis=1)

        # Masking word indices so the next groups doesn't reselect them
        word_indices_mask = array_ops.one_hot(word_indices,                     # [batch, group_size, vocab * beam]
                                              depth=vocab_size * beam_width,
                                              on_value=ops.convert_to_tensor(-np.Inf, dtype=dtypes.float32),
                                              off_value=ops.convert_to_tensor(0.0, dtype=dtypes.float32),
                                              dtype=dtypes.float32)

        # Reducing the probability of selecting the same word in the next groups
        same_word_mask = gen_array_ops.tile(array_ops.one_hot(group_next_word_ids,  # [batch, group_sz, vocab * beam]
                                                              depth=vocab_size,
                                                              on_value=math.log(0.5),
                                                              off_value=0.,
                                                              dtype=dtypes.float32), [1, 1, beam_width])

        # Adding mask to scores flat
        scores_flat = scores_flat \
                      + math_ops.reduce_sum(word_indices_mask, axis=1) \
                      + similarity_penalty * math_ops.reduce_sum(same_word_mask, axis=1)

    # Pick out the probs, beam_ids, and states according to the chosen predictions
    next_beam_probs = beam_search_decoder._tensor_gather_helper(gather_indices=word_indices,                            # pylint: disable=protected-access
                                                                gather_from=total_probs,
                                                                batch_size=batch_size,
                                                                range_size=beam_width * vocab_size,
                                                                gather_shape=[-1],
                                                                name='next_beam_probs')

    # Append new ids to current predictions
    previously_finished = beam_search_decoder._tensor_gather_helper(gather_indices=next_beam_ids,                       # pylint: disable=protected-access
                                                                    gather_from=previously_finished,
                                                                    batch_size=batch_size,
                                                                    range_size=beam_width,
                                                                    gather_shape=[-1])
    next_finished = gen_math_ops.logical_or(previously_finished,
                                            array_ops.expand_dims(gen_math_ops.greater_equal(time + 1, sequence_length),
                                                                  axis=-1),
                                            name='next_beam_finished')

    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged.
    # 2. Beams that are now finished (EOS predicted) have their length increased by 1.
    # 3. Beams that are not yet finished have their length increased by 1.
    lengths_to_add = math_ops.cast(gen_math_ops.logical_not(previously_finished), dtypes.int64)
    next_prediction_len = beam_search_decoder._tensor_gather_helper(gather_indices=next_beam_ids,                       # pylint: disable=protected-access
                                                                    gather_from=beam_state.lengths,
                                                                    batch_size=batch_size,
                                                                    range_size=beam_width,
                                                                    gather_shape=[-1])
    next_prediction_len += lengths_to_add

    # Pick out the cell_states according to the next_beam_ids. We use a different gather_shape here because the
    # cell_state tensors, i.e. the tensors that would be gathered from, all have dimension greater than two and we
    # need to preserve those dimensions.
    gather_helper = beam_search_decoder._maybe_tensor_gather_helper                                                     # pylint: disable=protected-access

    def _get_gather_shape(tensor):
        """ Computes the gather shape """
        # We can use -1 if the entire shape is static
        # Otherwise, we need to define specifically the gather shape (because we have multiple dynamic dims)
        if None not in tensor.shape.as_list()[1:]:
            return [batch_size * beam_width, -1]
        if len([1 for dim in tensor.shape.as_list()[2:] if dim is None]) == 1:
            return [batch_size * beam_width] + [-1 if dim.value is None else dim.value for dim in tensor.shape[2:]]
        raise ValueError('Cannot gather shape with more than 2 dynamic dims - %s' % tensor.shape.as_list())

    next_cell_state = nest.map_structure(lambda gather_from: gather_helper(gather_indices=next_beam_ids,
                                                                           gather_from=gather_from,
                                                                           batch_size=batch_size,
                                                                           range_size=beam_width,
                                                                           gather_shape=_get_gather_shape(gather_from)),
                                         next_cell_state)

    next_state = beam_search_decoder.BeamSearchDecoderState(cell_state=next_cell_state,
                                                            log_probs=next_beam_probs,
                                                            lengths=next_prediction_len,
                                                            finished=next_finished,
                                                            accumulated_attention_probs=())
    output = beam_search_decoder.BeamSearchDecoderOutput(scores=next_beam_scores,
                                                         predicted_ids=next_word_ids,
                                                         parent_ids=next_beam_ids)
    return output, next_state

def _get_scores(log_probs, sequence_lengths):
    """ Calculates scores for beam search hypotheses.
        :param log_probs: The log probabilities with shape `[batch_size, beam_width, vocab_size]`.
        :param sequence_lengths: The array of sequence lengths.
        :return: The scores for beam search hypotheses
    """
    del sequence_lengths            # Unused args
    return log_probs

def _mask_probs(probs, end_token, finished):
    """ Masks log probabilities.
        The result is that finished beams allocate all probability mass to PAD_ID and unfinished beams remain unchanged.
        :param probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
        :param end_token: An int32 id corresponding to the token to allocate probability to when finished.
        :param finished: A boolean tensor of shape `[batch_size, beam_width]` that specifies which elements in the
                         beam are finished already.
        :return: A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished beams stay unchanged and
                 finished beams are replaced with a tensor with all probability on the EOS token.
    """
    vocab_size = array_ops.shape(probs)[2]

    # All finished examples are replaced with a vector that has all probability on end_token
    finished_row = array_ops.one_hot(end_token,
                                     vocab_size,
                                     dtype=probs.dtype,
                                     on_value=ops.convert_to_tensor(0., dtype=probs.dtype),
                                     off_value=probs.dtype.min)
    finished_probs = gen_array_ops.tile(gen_array_ops.reshape(finished_row, [1, 1, -1]),
                                        array_ops.concat([array_ops.shape(finished), [1]], 0))
    finished_mask = gen_array_ops.tile(array_ops.expand_dims(finished, 2), [1, 1, vocab_size])
    return array_ops.where(finished_mask, finished_probs, probs)
