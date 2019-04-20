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
""" Wrappers
        - Contains various tensorflow decoder wrappers
"""
import collections
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import _transpose_batch_time
from diplomacy_research.utils.tensorflow import _unstack_ta
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import contrib_framework
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import embedding_lookup
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import rnn_cell_impl


def _get_embedding_fn(embedding):
    """ Returns a callable embedding function """
    if embedding is None:
        return lambda ids: ids
    if callable(embedding):
        return embedding
    return lambda ids: embedding_lookup(embedding, ids)

class ConcatenationWrapper(rnn_cell_impl.RNNCell):
    """ Wraps another `RNNCell` and concatenates the same input at each time step. """

    def __init__(self, cell, concat_inputs, name=None):
        """ Constructs an ConcatenationWrapper

            :param cell: An instance of `RNNCell`.
            :param concat_inputs: The inputs to concatenate at each time step [batch, input_size]
            :param name: name: Name to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(ConcatenationWrapper, self).__init__(name=name)
        rnn_cell_impl.assert_like_rnncell('cell', cell)

        # Setting values
        self._cell = cell
        self._concat_inputs = concat_inputs
        self._cell_input_fn = lambda input_1, input_2: array_ops.concat([input_1, input_2], axis=-1)

    @property
    def output_size(self):
        """ Returns the cell output size """
        return self._cell.output_size

    @property
    def state_size(self):
        """ The `state_size` property of the parent cell. """
        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this cell.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: A tuple containing zeroed out tensors and, possibly, empty TA objects.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):                                                          # pylint: disable=arguments-differ
        """ Performs a time-step (i.e. concatenation)
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: The state from the previous time step.
            :return: The cell output and the next state
        """
        cell_inputs = self._cell_input_fn(inputs, self._concat_inputs)
        cell_output, next_state = self._cell(cell_inputs, state)
        return cell_output, next_state

    def update_state(self, time, cell_outputs, state):
        """ Update the state of a feeder cell after it has been processed
            :param time: The current time step
            :param cell_outputs: The output of the main cell
            :param state: The current next_state for the feeder cell
            :return: The updated next_state
        """
        if not hasattr(self._cell, 'update_state'):
            return state
        return getattr(self._cell, 'update_state')(time, cell_outputs, state)

class ArrayConcatWrapperState(
        collections.namedtuple('ArrayConcatWrapperState', ('cell_state',        # The underlying cell state
                                                           'time'))):           # The current time step
    """ `namedtuple` storing the state of a `ArrayConcatWrapper`. """

    def clone(self, **kwargs):
        """ Clone this object, overriding components provided by kwargs. """
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return contrib_framework.with_same_shape(old, new)
            return new
        return nest.map_structure(with_same_shape, self, super(ArrayConcatWrapperState, self)._replace(**kwargs))

class ArrayConcatWrapper(rnn_cell_impl.RNNCell):
    """ Wraps another `RNNCell` and concatenates the input[i] of the array at each time step. """

    def __init__(self, cell, concat_inputs, mask_inputs=None, embedding=None, name=None):
        """ Constructs an ArrayConcatWrapper

            If embedding is provided, the concat_inputs is expected to be [batch, time]
            If embedding is not provided, the concat_inputs is expected to be [batch, time, input_size]

            mask_inputs of True will mask (zero-out) the given input (or embedded input)

            :param cell: An instance of `RNNCell`.
            :param concat_inputs: The inputs to concatenate [batch, time] or [batch, time, input_size]
            :param mask_inputs: Optional. Boolean [batch, time] that indicates if the concat_inputs is to be masked
            :param embedding: Optional. Embedding fn or embedding vector to embed the concat_inputs at each time step
            :param name: name: Name to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(ArrayConcatWrapper, self).__init__(name=name)
        rnn_cell_impl.assert_like_rnncell('cell', cell)

        # Setting values
        self._cell = cell
        self._cell_input_fn = lambda input_1, input_2: array_ops.concat([input_1, input_2], axis=-1)
        self._embedding_fn = _get_embedding_fn(embedding)
        self._mask_inputs_ta = None

        # Converting mask inputs to a tensor array
        if mask_inputs is not None:
            mask_inputs = nest.map_structure(_transpose_batch_time, mask_inputs)
            self._mask_inputs_ta = nest.map_structure(_unstack_ta, mask_inputs)         # [time, batch]

        # Converting concat_inputs to a tensor array
        concat_inputs = nest.map_structure(_transpose_batch_time, concat_inputs)
        self._concat_inputs_ta = nest.map_structure(_unstack_ta, concat_inputs)         # [time, batch] / [t, b, inp_sz]

    @property
    def output_size(self):
        """ Returns the cell output size """
        return self._cell.output_size

    @property
    def state_size(self):
        """ The `state_size` property of `ArrayConcatWrapper`.
            :return: An `ArrayConcatWrapperState` tuple containing shapes used by this object.
        """
        return ArrayConcatWrapperState(cell_state=self._cell.state_size,
                                       time=tensor_shape.TensorShape([]))

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `ArrayConcatWrapper`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: An `ArrayConcatWrapperState` tuple containing zeroed out tensors and, possibly, empty TA objects.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return ArrayConcatWrapperState(cell_state=self._cell.zero_state(batch_size, dtype),
                                           time=array_ops.zeros([], dtype=dtypes.int32))

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):                                                          # pylint: disable=arguments-differ
        """ Performs a time-step (i.e. concatenation of input[i])
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: The state from the previous time step.
            :return: The cell output and the next state
        """
        if not isinstance(state, ArrayConcatWrapperState):
            raise TypeError('Expected state to be instance of ArrayConcatWrapperState. Received type %s instead.'
                            % type(state))

        next_time = state.time + 1
        concat_inputs_i = self._embedding_fn(self._concat_inputs_ta.read(state.time))          # [batch, emb_size]

        # Masking - Values of True are zeroed-out
        if self._mask_inputs_ta is not None:
            mask_i = self._mask_inputs_ta.read(state.time)
            concat_inputs_i = concat_inputs_i * math_ops.cast(gen_math_ops.logical_not(mask_i)[:, None], dtypes.float32)

        # Concatenating to real input
        cell_inputs = self._cell_input_fn(inputs, concat_inputs_i)
        cell_state = state.cell_state
        cell_output, cell_state = self._cell(cell_inputs, cell_state)
        next_state = ArrayConcatWrapperState(cell_state=cell_state, time=next_time)
        return cell_output, next_state

    def update_state(self, time, cell_outputs, state):
        """ Update the state of a feeder cell after it has been processed
            :param time: The current time step
            :param cell_outputs: The output of the main cell
            :param state: The current next_state for the feeder cell
            :return: The updated next_state
        """
        if not hasattr(self._cell, 'update_state'):
            return state
        return ArrayConcatWrapperState(cell_state=getattr(self._cell, 'update_state')(time,
                                                                                      cell_outputs,
                                                                                      state.cell_state),
                                       time=state.time)

class IdentityCell(rnn_cell_impl.RNNCell):
    """ RNN cell that returns its inputs as outputs """

    def __init__(self, output_size, name=None):
        """ IdentityCell - Returns its inputs as outputs
            :param output_size: The size of the input / output
            :param name: name: Name to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(IdentityCell, self).__init__(name=name)

        # Setting values
        self._output_size = output_size

    @property
    def output_size(self):
        """ Returns the cell output size """
        return self._output_size

    @property
    def state_size(self):
        """ The `state_size` property of `IdentityCell`. """
        return tensor_shape.TensorShape([])

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `IdentityCell`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: A zeroed out scalar representing the initial state of the cell.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return array_ops.zeros([], dtype=dtypes.int32)

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):  # pylint: disable=arguments-differ
        """ Runs the identity cell
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: The state from the previous time step.
            :return: The cell output and the next state
        """
        outputs, next_state = inputs, state
        assert outputs.shape[1].value == self._output_size, \
            'Expected output size of %d - Got %d' % (self._output_size, outputs.shape[1].value)
        return outputs, next_state
