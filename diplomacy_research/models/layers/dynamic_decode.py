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
""" Dynamic Decoders
    - Contains an implementation of dynamic_decode that uses shape invariants
    Source: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/seq2seq/python/ops/decoder.py
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import _create_zero_outputs, _transpose_batch_time
from diplomacy_research.utils.tensorflow import seq2seq
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops, gen_array_ops
from diplomacy_research.utils.tensorflow import constant_op
from diplomacy_research.utils.tensorflow import context
from diplomacy_research.utils.tensorflow import control_flow_ops
from diplomacy_research.utils.tensorflow import control_flow_util
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import math_ops, gen_math_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import tensor_array_ops
from diplomacy_research.utils.tensorflow import tensor_util
from diplomacy_research.utils.tensorflow import variable_scope

def dynamic_decode(decoder, output_time_major=False, impute_finished=False, maximum_iterations=None,
                   parallel_iterations=32, invariants_map=None, swap_memory=False, scope=None):
    """ Performs dynamic decoding with `decoder`.
        :param decoder: A `Decoder` instance.
        :param output_time_major: If True, outputs [time, batch, ...], otherwise outputs [batch, time, ...]
        :param impute_finished: If true, finished states are copied through the end of the game
        :param maximum_iterations: Int or None. The maximum number of steps (otherwise decode until it's done)
        :param parallel_iterations: Argument passed to tf.while_loop
        :param invariants_map: Optional. Dictionary of tensor path (in initial_state) to its shape invariant.
        :param swap_memory: Argument passed to `tf.while_loop`.
        :param scope: Optional variable scope to use.
        :return: A tuple of 1) final_outputs, 2) final_state, 3) final_sequence_length
    """
    if not isinstance(decoder, seq2seq.Decoder):
        raise TypeError('Expected decoder to be type Decoder, but saw: %s' % type(decoder))

    with variable_scope.variable_scope(scope, 'decoder') as varscope:

        # Determine context types.
        ctxt = ops.get_default_graph()._get_control_flow_context()                                                      # pylint: disable=protected-access
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        in_while_loop = control_flow_util.GetContainingWhileContext(ctxt) is not None

        # Properly cache variable values inside the while_loop.
        # Don't set a caching device when running in a loop, since it is possible that train steps could be wrapped
        # in a tf.while_loop. In that scenario caching prevents forward computations in loop iterations from re-reading
        # the updated weights.
        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        # Setting maximum iterations
        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(maximum_iterations,
                                                       dtype=dtypes.int32,
                                                       name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError('maximum_iterations must be a scalar')

        def _inv_shape(maybe_ta):
            """ Returns the invariatns shape """
            if isinstance(maybe_ta, tensor_array_ops.TensorArray):
                return maybe_ta.flow.shape
            return maybe_ta.shape

        def _invariants(structure):
            """ Returns the invariants of a structure """
            return nest.map_structure(_inv_shape, structure)

        def _map_invariants(structure):
            """ Returns the invariants of a structure, but replaces the invariant using the value in invariants_map """
            return nest.map_structure_with_paths(lambda path, tensor: (invariants_map or {}).get(path,
                                                                                                 _inv_shape(tensor)),
                                                 structure)

        # Initializing decoder
        initial_finished, initial_inputs, initial_state = decoder.initialize()
        zero_outputs = _create_zero_outputs(decoder.output_size, decoder.output_dtype, decoder.batch_size)

        if is_xla and maximum_iterations is None:
            raise ValueError('maximum_iterations is required for XLA compilation.')
        if maximum_iterations is not None:
            initial_finished = gen_math_ops.logical_or(initial_finished, maximum_iterations <= 0)
        initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        # Creating initial output TA
        def _shape(batch_size, from_shape):
            """ Returns the batch_size concatenated with the from_shape """
            if (not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0):
                return tensor_shape.TensorShape(None)
            batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name='batch_size'))
            return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla

        def _create_ta(shape, dtype):
            """ Creates a tensor array"""
            return tensor_array_ops.TensorArray(dtype=dtype,
                                                size=0 if dynamic_size else maximum_iterations,
                                                dynamic_size=dynamic_size,
                                                element_shape=_shape(decoder.batch_size, shape))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs, finished, unused_sequence_lengths):
            """ While loop condition"""
            return gen_math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            """ Internal while_loop body. """
            (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs, state)
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = gen_math_ops.logical_or(decoder_finished, finished)
            next_sequence_lengths = array_ops.where(gen_math_ops.logical_not(finished),
                                                    gen_array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                                                    sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(lambda out, zero: array_ops.where(finished, zero, out),
                                          next_outputs,
                                          zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays, multiple dynamic dims, and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                elif None in new.shape.as_list()[1:]:
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(_maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths)

        res = control_flow_ops.while_loop(condition,
                                          body,
                                          loop_vars=(initial_time,
                                                     initial_outputs_ta,
                                                     initial_state,
                                                     initial_inputs,
                                                     initial_finished,
                                                     initial_sequence_lengths),
                                          shape_invariants=(_invariants(initial_time),
                                                            _invariants(initial_outputs_ta),
                                                            _map_invariants(initial_state),
                                                            _invariants(initial_inputs),
                                                            _invariants(initial_finished),
                                                            _invariants(initial_sequence_lengths)),
                                          parallel_iterations=parallel_iterations,
                                          maximum_iterations=maximum_iterations,
                                          swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths
