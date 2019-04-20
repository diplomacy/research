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
""" Decoders
    - Contains derived tensorflow decoders that can apply a decoder mask at every time step
"""
import collections
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import seq2seq
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import nest

LARGE_NEGATIVE = -1e20

class MaskedInputs(
        collections.namedtuple('MaskedInputs', ('inputs', 'mask'))):
    """ Basic masked inputs (where a mask is also provided for every output """

class CandidateInputs(
        collections.namedtuple('CandidateInputs', ('inputs', 'candidates', 'candidates_emb'))):
    """ Candidate Input type. Contains a list of candidates (b, nb_candidate) and their corresponding embedding """

class BasicDecoderWithStateOutput(
        collections.namedtuple('BasicDecoderWithStateOutput', ('rnn_output', 'rnn_state', 'sample_id'))):
    """ Basic Decoder Named Tuple with rnn_output, rnn_state, and sample_id """


class MaskedBasicDecoder(seq2seq.BasicDecoder):
    """ Basic sampling decoder with mask applied at each step. """

    def __init__(self, cell, helper, initial_state, output_layer=None, extract_state=False):
        """ Constructor
            :param cell: An `RNNCell` instance.
            :param helper: A `Helper` instance.
            :param initial_state: A (nested tuple of...) tensors and TensorArrays. Initial state of the RNNCell.
            :param output_layer: Optional. An instance of `tf.layers.Layer`, i.e., `tf.layers.Dense`. Optional layer
                                 to apply to the RNN output prior to storing the result or sampling.
            :param extract_state: Optional. Boolean. If set, will also return the RNN state at each time step.
            :type cell: tensorflow.python.ops.rnn_cell_impl.RNNCell
            :type helper: tensorflow.contrib.seq2seq.python.ops.helper.Helper
            :type output_layer: tensorflow.python.layers.base.Layer
        """
        super(MaskedBasicDecoder, self).__init__(cell=cell,
                                                 helper=helper,
                                                 initial_state=initial_state,
                                                 output_layer=output_layer)
        self.extract_state = extract_state

    @property
    def output_size(self):
        # Return the cell output and the id
        if self.extract_state:
            return BasicDecoderWithStateOutput(rnn_output=self._rnn_output_size(),
                                               rnn_state=tensor_shape.TensorShape([self._cell.output_size]),
                                               sample_id=self._helper.sample_ids_shape)
        return seq2seq.BasicDecoderOutput(rnn_output=self._rnn_output_size(),
                                          sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        if self.extract_state:
            return BasicDecoderWithStateOutput(nest.map_structure(lambda _: dtype, self._rnn_output_size()),
                                               dtype,
                                               self._helper.sample_ids_dtype)
        return seq2seq.BasicDecoderOutput(nest.map_structure(lambda _: dtype, self._rnn_output_size()),
                                          self._helper.sample_ids_dtype)

    def step(self, time, inputs, state, name=None):
        """ Performs a decoding step
            :param time: scalar `int32` tensor.
            :param inputs: A (structure of) input tensors.  (** This is a MaskedInputs tuple **)
            :param state: A (structure of) state tensors and TensorArrays.
            :param name: Name scope for any created operations.
            :return: (outputs, next_state, next_inputs, finished)
        """
        assert isinstance(inputs, (MaskedInputs, ops.Tensor)), 'Expected "MaskedInputs" or a Tensor.'
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            inputs, output_mask = inputs, None
            if isinstance(inputs, MaskedInputs):
                inputs, output_mask = inputs.inputs, inputs.mask
            cell_outputs, cell_state = self._cell(inputs, state)
            cell_state_output = cell_outputs                        # Corresponds to cell_state.h (before output layer)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            if output_mask is not None:
                cell_outputs = gen_math_ops.add(cell_outputs, (1. - output_mask) * LARGE_NEGATIVE)
            sample_ids = self._helper.sample(time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(time=time,
                                                                           outputs=cell_outputs,
                                                                           state=cell_state,
                                                                           sample_ids=sample_ids)
        if self.extract_state:
            outputs = BasicDecoderWithStateOutput(cell_outputs, cell_state_output, sample_ids)
        else:
            outputs = seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        """ Finalize function """
        return outputs, final_state

class CandidateBasicDecoder(seq2seq.BasicDecoder):
    """ Basic sampling decoder that chooses among candidates at each step. """

    def __init__(self, cell, helper, initial_state, max_candidate_length, output_layer=None, extract_state=False):
        """ Constructor
            :param cell: An `RNNCell` instance.
            :param helper: A `Helper` instance.
            :param initial_state: A (nested tuple of...) tensors and TensorArrays. Initial state of the RNNCell.
            :param max_candidate_length: The maximum number of candidates
            :param output_layer: Optional. An instance of `tf.layers.Layer`, i.e., `tf.layers.Dense`. Optional layer
                                 to apply to the RNN output prior to storing the result or sampling.
            :param extract_state: Optional. Boolean. If set, will also return the RNN state at each time step.
            :type cell: tensorflow.python.ops.rnn_cell_impl.RNNCell
            :type helper: tensorflow.contrib.seq2seq.python.ops.helper.Helper
            :type output_layer: tensorflow.python.layers.base.Layer
        """
        super(CandidateBasicDecoder, self).__init__(cell=cell,
                                                    helper=helper,
                                                    initial_state=initial_state,
                                                    output_layer=output_layer)
        self.extract_state = extract_state
        self.max_candidate_length = max_candidate_length

    @property
    def output_size(self):
        # Return the cell output and the id
        if self.extract_state:
            return BasicDecoderWithStateOutput(rnn_output=self.max_candidate_length,
                                               rnn_state=tensor_shape.TensorShape([self._cell.output_size]),
                                               sample_id=self._helper.sample_ids_shape)
        return seq2seq.BasicDecoderOutput(rnn_output=self.max_candidate_length,
                                          sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        if self.extract_state:
            return BasicDecoderWithStateOutput(dtype,
                                               dtype,
                                               self._helper.sample_ids_dtype)
        return seq2seq.BasicDecoderOutput(dtype,
                                          self._helper.sample_ids_dtype)

    def step(self, time, inputs, state, name=None):
        """ Performs a decoding step
            :param time: scalar `int32` tensor.
            :param inputs: A (structure of) input tensors.  (** This is a MaskedInputs tuple **)
            :param state: A (structure of) state tensors and TensorArrays.
            :param name: Name scope for any created operations.
            :return: (outputs, next_state, next_inputs, finished)
        """
        assert isinstance(inputs, CandidateInputs), 'The inputs must be of type "CandidateInputs"'
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            inputs, candidates, candidates_emb = inputs.inputs, inputs.candidates, inputs.candidates_emb
            cell_outputs, cell_state = self._cell(inputs, state)
            cell_state_output = cell_outputs                        # Corresponds to cell_state.h (before output layer)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            # Adding a bias dimension, then computing candidate logits and masking PAD_IDs
            cell_outputs = array_ops.pad(cell_outputs, [(0, 0), (0, 1)], constant_values=1.)
            cell_outputs = math_ops.reduce_sum(cell_outputs[:, None, :] * candidates_emb, axis=-1)
            output_mask = math_ops.cast(gen_math_ops.greater(candidates, 0), dtypes.float32)
            cell_outputs = gen_math_ops.add(cell_outputs, (1. - output_mask) * LARGE_NEGATIVE)

            # Sampling and computing next inputs
            sample_ids = self._helper.sample(time=time, outputs=(cell_outputs, candidates), state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(time=time,
                                                                           outputs=cell_outputs,
                                                                           state=cell_state,
                                                                           sample_ids=sample_ids)
        if self.extract_state:
            outputs = BasicDecoderWithStateOutput(cell_outputs, cell_state_output, sample_ids)
        else:
            outputs = seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        """ Finalize function """
        return outputs, final_state
