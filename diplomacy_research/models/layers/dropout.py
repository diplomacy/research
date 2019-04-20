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
""" Dropout
    - Contains tensorflow class to apply seeded dropout
"""
import collections
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.models.layers.seeded_random import seeded_random
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import gen_array_ops
from diplomacy_research.utils.tensorflow import constant_op
from diplomacy_research.utils.tensorflow import context
from diplomacy_research.utils.tensorflow import contrib_framework
from diplomacy_research.utils.tensorflow import control_flow_ops
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import gen_math_ops
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import nn_ops
from diplomacy_research.utils.tensorflow import rnn_cell_impl
from diplomacy_research.utils.tensorflow import tensor_shape


def seeded_dropout(inputs, seeds, keep_probs, offset=None, noise_shape=None, seed=None, name=None):
    """ Computes dropout (with a deterministic mask).
        Every item in the batch has a deterministic seed to compute the deterministic mask

        With probability `keep_probs`, outputs the input element scaled up by `1 / keep_prob`, otherwise outputs `0`.
        The scaling is so that the expected sum is unchanged.

        By default, each element is kept or dropped independently. If `noise_shape` is specified, it must be
        broadcastable to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]` will make
        independent decisions.

        For example, if `shape(x) = [k, l, m, n]` and `noise_shape = [k, 1, 1, n]`, each batch and channel component
        will be kept independently and each row and column will be kept or not kept together.

        :param inputs: A floating point tensor.
        :param seeds: A tensor representing the seed for each item in the batch. (Size: (batch,))
        :param keep_probs: A scalar or vector of size (batch,). The probability that each element is kept.
        :param offset: Integer. Alternative offset to apply to compute the deterministic mask (e.g. in a loop).
        :param noise_shape: A 1-D `Tensor` of type `int32`, represents the shape for randomly generated keep/drop flags.
        :param seed: A Python integer. Used to create a default seed for the operation.
        :param name: name: A name for this operation (optional).
        :return: A Tensor of the same shape of `x`.
    """
    if offset is None:
        seeded_dropout.offset += 40555607

    # If inputs is a scalar, this is likely the 'time' attribute in a state, we don't want to mask it
    # Same thing for integers - We can safely ignore them
    # So we don't want to mask it
    if not inputs.shape or inputs.dtype.is_integer:
        return inputs

    with ops.name_scope(name, 'seeded_dropout', [inputs]):
        inputs = ops.convert_to_tensor(inputs, name='x')
        if not inputs.dtype.is_floating:
            raise ValueError('Expected a floating point tensor. Got a %s tensor instead.' % inputs.dtype)
        if isinstance(keep_probs, float) and not 0 < keep_probs <= 1:
            raise ValueError('keep_probs must be a scalar tensor or a float in the range (0, 1], got %g' % keep_probs)

        # Early return if nothing needs to be dropped.
        if isinstance(keep_probs, float) and keep_probs == 1:
            return inputs

        # Not supported in eager mode
        if context.executing_eagerly():
            raise ValueError('This function is not supported in eager mode.')

        # Converting to tensor
        keep_probs = ops.convert_to_tensor(keep_probs, dtype=inputs.dtype, name='keep_probs')
        keep_probs = gen_math_ops.maximum(0., gen_math_ops.minimum(1., keep_probs))
        keep_probs = gen_array_ops.reshape(keep_probs, [-1] + [1] * (len(inputs.shape) - 1))
        all_keep_probs_are_one = math_ops.reduce_all(gen_math_ops.equal(keep_probs, 1.))

        # Computing noise shape
        noise_shape = nn_ops._get_noise_shape(inputs, noise_shape)  # pylint: disable=protected-access

        def get_dropout_mask():
            """ Computes the dropout mask """
            # random_tensor = uniform [keep_probs, 1.0 + keep_probs)
            random_tensor = keep_probs
            random_tensor += seeded_random(seeds,
                                           offset=offset if offset is not None else seeded_dropout.offset,
                                           shape=noise_shape[1:],
                                           dtype=inputs.dtype,
                                           seed=seed)

            # 0. if [keep_probs, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = gen_math_ops.floor(random_tensor)
            ret = math_ops.divide(inputs, keep_probs) * binary_tensor
            ret.set_shape(inputs.get_shape())

            # Setting control flow ops to avoid computing this function if not required
            with ops.control_dependencies([ret]):
                return array_ops.identity(ret)

        # Returning the dropout mask
        return control_flow_ops.cond(all_keep_probs_are_one,
                                     true_fn=lambda: inputs,
                                     false_fn=get_dropout_mask)

class SeededDropoutWrapperState(
        collections.namedtuple('SeededDropoutWrapperState', ('cell_state',      # The underlying cell state
                                                             'time'))):         # The current time
    """ `namedtuple` storing the state of a `SeededDropoutWrapper`. """
    def clone(self, **kwargs):
        """ Clone this object, overriding components provided by kwargs. """
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return contrib_framework.with_same_shape(old, new)
            return new
        return nest.map_structure(with_same_shape,
                                  self,
                                  super(SeededDropoutWrapperState, self)._replace(**kwargs))

class SeededDropoutWrapper(rnn_cell_impl.DropoutWrapper):
    """Operator adding seeded dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, seeds, input_keep_probs=1.0, output_keep_probs=1.0, state_keep_probs=1.0,
                 variational_recurrent=False, input_size=None, dtype=None, seed=None,
                 dropout_state_filter_visitor=None):
        """ Create a cell with added input, state, and/or output seeded dropout.

            If `variational_recurrent` is set to `True` (**NOT** the default behavior), then the same dropout mask is
            applied at every step, as described in:

                Y. Gal, Z Ghahramani.    "A Theoretically Grounded Application of Dropout in
                Recurrent Neural Networks".    https://arxiv.org/abs/1512.05287

            Otherwise a different dropout mask is applied at every time step.

            Note, by default (unless a custom `dropout_state_filter` is provided), the memory state (`c` component
            of any `LSTMStateTuple`) passing through a `DropoutWrapper` is never modified.    This behavior is
            described in the above article.

            :param cell: an RNNCell, a projection to output_size is added to it.
            :param seeds: A tensor representing the seed for each item in the batch. (Size: (batch,))
            :param input_keep_probs: float, scalar tensor, or batch vector (b,). Input keep probabilities.
            :param output_keep_probs: float, scalar tensor, or batch vector (b,). Output keep probabilities.
            :param state_keep_probs: float, scalar tensor, or batch vector (b,). State keep probabilities (excl 'c')
            :param variational_recurrent:  If `True`, same dropout pattern is applied across all time steps per run call
            :param input_size: (optional) (possibly nested tuple of) `TensorShape` objects containing the depth(s) of
                               the input tensors expected to be passed in to the `DropoutWrapper`.
                               Required and used **iff** `variational_recurrent = True` and `input_keep_prob < 1`.
            :param dtype: (optional) The `dtype` of the input, state, and output tensors.
            :param seed: (optional) integer, the default randomness seed to use if one of the seeds is 0.
            :param dropout_state_filter_visitor: Optional. See DropoutWrapper for description.
        """
        # pylint: disable=too-many-arguments
        SeededDropoutWrapper.offset += 11828683
        super(SeededDropoutWrapper, self).__init__(cell=cell,
                                                   input_keep_prob=1.,
                                                   output_keep_prob=1.,
                                                   state_keep_prob=1.,
                                                   variational_recurrent=False,
                                                   input_size=input_size,
                                                   dtype=dtype,
                                                   seed=seed,
                                                   dropout_state_filter_visitor=dropout_state_filter_visitor)

        def _convert_to_probs_tensor(keep_probs):
            """ Converts a keep_probs tensor to its broadcastable shape """
            probs_tensor = ops.convert_to_tensor(keep_probs)
            probs_tensor = gen_math_ops.maximum(0., gen_math_ops.minimum(1., probs_tensor))
            return gen_array_ops.reshape(probs_tensor, [-1, 1])

        # Converting to tensor
        self._input_keep_probs = _convert_to_probs_tensor(input_keep_probs)
        self._output_keep_probs = _convert_to_probs_tensor(output_keep_probs)
        self._state_keep_probs = _convert_to_probs_tensor(state_keep_probs)

        # Detecting if we skip computing those probs
        self._skip_input_keep_probs = isinstance(input_keep_probs, float) and input_keep_probs == 1.
        self._skip_output_keep_probs = isinstance(output_keep_probs, float) and output_keep_probs == 1.
        self._skip_state_keep_probs = isinstance(state_keep_probs, float) and state_keep_probs == 1.

        # Generating variational recurrent
        self._seeds = seeds
        self._variational_recurrent = variational_recurrent

        enum_map_up_to = rnn_cell_impl._enumerated_map_structure_up_to                                                  # pylint: disable=protected-access

        def batch_noise(input_dim, inner_offset, inner_seed):
            """ Generates noise for variational dropout """
            if not isinstance(input_dim, int):              # Scalar tensor - We can ignore it safely
                return None
            return seeded_random(seeds,
                                 offset=SeededDropoutWrapper.offset + inner_offset,
                                 shape=[input_dim],
                                 dtype=dtype,
                                 seed=inner_seed)

        # Computing deterministic recurrent noise
        if variational_recurrent:
            if dtype is None:
                raise ValueError('When variational_recurrent=True, dtype must be provided')

            input_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 127602767, self._gen_seed('input', index))
            state_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 31248361, self._gen_seed('state', index))
            output_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 71709719, self._gen_seed('output', index))

            if not self._skip_input_keep_probs:
                if input_size is None:
                    raise ValueError("When variational_recurrent=True and input_keep_prob < 1.0 or "
                                     "is unknown, input_size must be provided")
                self._recurrent_input_noise = enum_map_up_to(input_size, input_map_fn, input_size)
            self._recurrent_state_noise = enum_map_up_to(cell.state_size, state_map_fn, cell.state_size)
            self._recurrent_output_noise = enum_map_up_to(cell.output_size, output_map_fn, cell.output_size)

    @property
    def state_size(self):
        """ The `state_size` property of `SeededDropoutWrapper`.
            :return: A `SeededDropoutWrapperState` tuple containing shapes used by this object.
        """
        return SeededDropoutWrapperState(cell_state=self._cell.state_size,
                                         time=tensor_shape.TensorShape([]))

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `AttentionWrapper`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: An `SelfAttentionWrapperState` tuple containing zeroed out tensors.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return SeededDropoutWrapperState(cell_state=self._cell.zero_state(batch_size, dtype),
                                             time=array_ops.zeros([], dtype=dtypes.int32))

    def _do_dropout(self, values, offset, salt_prefix, recurrent_noise, keep_probs, filtered_structure=None):
        """ Decides whether to perform standard dropout or recurrent dropout
            :param values: (Possibly nested) tensor on which to perform dropout.
            :param offset: Integer. Offset to apply to compute the mask (should be different at each time step).
            :param salt_prefix: Salt prefix to compute the default op seed.
            :param recurrent_noise: The recurrent noise to apply for variational dropout.
            :param keep_probs: The probabilities to keep the inputs (no dropout). - Size: (batch,)
            :param filtered_structure: Tree-like structure used to decide where to apply dropout.
            :return: The values with dropout applied.
        """
        enum_map_up_to = rnn_cell_impl._enumerated_map_structure_up_to                                                  # pylint: disable=protected-access

        def reg_dropout_map_fn(index, do_dropout, value):
            """ Applies regular dropout """
            if not isinstance(do_dropout, bool) or do_dropout:
                return seeded_dropout(value, self._seeds, keep_probs,
                                      offset=SeededDropoutWrapper.offset + offset,
                                      seed=self._gen_seed(salt_prefix, index))
            return value

        def var_dropout_map_fn(index, do_dropout, value, noise):
            """ Applies variational dropout """
            if noise is None:
                return value
            if not isinstance(do_dropout, bool) or do_dropout:
                return self._variational_recurrent_dropout_value(index, value, noise, keep_probs)
            return value

        # Shallow filtered substructure
        # To traverse the entire structure; inside the dropout fn, we check to see if leafs of this are bool or not
        if filtered_structure is None:
            filtered_structure = values

        # Regular Dropout
        if not self._variational_recurrent:
            return enum_map_up_to(filtered_structure, reg_dropout_map_fn, *[filtered_structure, values])

        # Variational Dropout
        return enum_map_up_to(filtered_structure, var_dropout_map_fn, *[filtered_structure, values, recurrent_noise])

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if not isinstance(state, SeededDropoutWrapperState):
            raise TypeError('Expected state to be instance of SeededDropoutWrapperState. Received type %s instead.'
                            % type(state))

        # Dropout on inputs
        if not self._skip_input_keep_probs:
            inputs = self._do_dropout(inputs,
                                      offset=state.time * 46202587,
                                      salt_prefix='input',
                                      recurrent_noise=self._recurrent_input_noise,
                                      keep_probs=self._input_keep_probs)

        # Dropout on state
        output, new_state = self._cell(inputs, state.cell_state, scope=scope)
        if not self._skip_state_keep_probs:
            shallow_filtered_substructure = nest.get_traverse_shallow_structure(self._dropout_state_filter, new_state)
            new_state = self._do_dropout(new_state,
                                         offset=state.time * 79907039,
                                         salt_prefix='state',
                                         recurrent_noise=self._recurrent_state_noise,
                                         keep_probs=self._state_keep_probs,
                                         filtered_structure=shallow_filtered_substructure)

        # Dropout on outputs
        if not self._skip_output_keep_probs:
            output = self._do_dropout(output,
                                      offset=state.time * 16676461,
                                      salt_prefix='output',
                                      recurrent_noise=self._recurrent_output_noise,
                                      keep_probs=self._output_keep_probs)

        # Returning
        return output, SeededDropoutWrapperState(cell_state=new_state, time=state.time + 1)

# Settings offsets, so that a different offset is generated per call
seeded_dropout.offset = constant_op.constant(0)
SeededDropoutWrapper.offset = constant_op.constant(0)
