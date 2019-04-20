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
# Source: https://github.com/openai/gradient-checkpointing/blob/master/memory_saving_gradients.py
# MIT License
""" Provides gradient checkpoint to reduce memory usage of backward propagation """
import contextlib
import logging
import sys
import time
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

import numpy as np
from toposort import toposort
from diplomacy_research.utils.tensorflow import tf, gradients as gradients_lib, graph_editor

# Increasing recursion limit
sys.setrecursionlimit(10000)

# Constants
LOGGER = logging.getLogger(__name__)
DEBUG_LOGGING = False
MIN_CHECKPOINT_NODE_SIZE = 1024
TF_GRADIENTS = gradients_lib.gradients
UTIL = sys.modules[__name__]


def gradients(ys, xs, grad_ys=None, checkpoints='collection', aggregation_method=None, **kwargs):
    """ Authors: Tim Salimans and Yaroslav Bulatov

        Memory efficient gradient implementation inspired by "Training Deep Nets with Sublinear Memory Cost"
        by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

        :param ys: Tensor or list of tensors.
        :param xs: Tensor or list of tensors
        :param grad_ys: List of tensors holding the gradients received by the ys. Same length as ys.
        :param checkpoints: One of
            1) a list consisting of tensors from the forward pass of the neural net that we should re-use when
               calculating the gradients in the backward pass all other tensors that do not appear in this list will
               be re-computed
            2) The string 'speed': checkpoint all outputs of convolutions and matmuls. these ops are usually the most
                                    expensive, so checkpointing them maximizes the running speed (this is a good option
                                    if nonlinearities, concats, batchnorms, etc are taking up a lot of memory)
            3) The string 'memory': try to minimize the memory usage (currently using a very simple strategy that
                                    identifies a number of bottleneck tensors in the graph to checkpoint)
            4) The string 'collection': look for a tensorflow collection named 'checkpoints', which holds the tensors
                                    to checkpoint
        :param aggregation_method:
        :param kwargs: Optional kwargs to pass to tf.gradients
        :return: The gradients
    """
    # pylint: disable=invalid-name
    # Computes forwards and backwards ops
    # Forward ops are all ops that are candidates for recomputation
    ys = [ys] if not isinstance(ys, list) else ys
    xs = [xs] if not isinstance(xs, list) else xs
    bwd_ops = graph_editor.get_backward_walk_ops([y.op for y in ys], inclusive=True)
    fwd_ops = graph_editor.get_forward_walk_ops([x.op for x in xs], inclusive=True, within_ops=bwd_ops)

    debug_print("bwd_ops: %s", bwd_ops)
    debug_print("fwd_ops: %s", fwd_ops)

    # Exclude ops with no inputs, or ops linked to xs or variables
    xs_ops = _to_ops(xs)
    fwd_ops = [op for op in fwd_ops if (op.inputs
                                        and not op in xs_ops
                                        and not '/assign' in op.name
                                        and not '/Assign' in op.name
                                        and not '/read' in op.name)]

    # Computes the list of tensors that can be recomputed from fw_ops
    ts_all = graph_editor.filter_ts(fwd_ops, True)
    ts_all = [t for t in ts_all if '/read' not in t.name]
    ts_all = set(ts_all) - set(xs) - set(ys)

    # Construct list of tensors to checkpoint during forward pass, if not given as input
    # At this point automatic selection happened and checkpoints is list of nodes
    if not isinstance(checkpoints, list):
        checkpoints = {'collection': _get_collection_checkpoints,
                       'speed': _get_speed_checkpoints,
                       'memory': _get_memory_checkpoints}[checkpoints](fwd_ops, ts_all,
                                                                       ys=ys, xs=xs, grad_ys=grad_ys,
                                                                       aggregation_method=aggregation_method,
                                                                       **kwargs)
    checkpoints = list(set(checkpoints).intersection(ts_all))
    assert isinstance(checkpoints, list)
    debug_print("Checkpoint nodes used: %s", checkpoints)

    # Better error handling of special cases
    # xs are already handled as checkpoint nodes, so no need to include them
    xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
    if xs_intersect_checkpoints:
        debug_print('Warning, some input nodes are also checkpoint nodes: %s', xs_intersect_checkpoints)
    ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
    debug_print('ys: %s, checkpoints: %s, intersect: %s', ys, checkpoints, ys_intersect_checkpoints)

    # Saving an output node (ys) gives no benefit in memory while creating new edge cases, exclude them
    if ys_intersect_checkpoints:
        debug_print('Warning, some output nodes are also checkpoints nodes: %s', ys_intersect_checkpoints)

    # Remove initial and terminal nodes from checkpoints list if present
    # Only keeping checkpoints not in a control flow context
    checkpoints = list(set(checkpoints) - set(ys) - set(xs))
    checkpoints = [ckpt for ckpt in checkpoints if ckpt._op._control_flow_context is None]                              # pylint: disable=protected-access

    # Check that we have some nodes to checkpoint
    if not checkpoints:
        raise RuntimeError('No checkpoints nodes found or given as input!')

    # Disconnect dependencies between checkpointed tensors
    checkpoints_disconnected = {}
    for ckpt in checkpoints:
        if ckpt.op and ckpt.op.name is not None:
            grad_node = tf.stop_gradient(ckpt, name=ckpt.op.name + '_stop_grad')
        else:
            grad_node = tf.stop_gradient(ckpt)
        checkpoints_disconnected[ckpt] = grad_node

    # Partial derivatives to the checkpointed tensors and xs
    ops_to_copy = fast_backward_ops(seed_ops=[y.op for y in ys], stop_at_ts=checkpoints, within_ops=fwd_ops)
    debug_print('Found %s ops to copy within fwd_ops %s, seed %s, stop_at %s',
                len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints)
    debug_print('ops_to_copy = %s', ops_to_copy)
    debug_print('Processing list %s', ys)

    _, info = graph_editor.copy_with_input_replacements(graph_editor.sgv(ops_to_copy), {})
    for origin_op, op in info._transformed_ops.items():                                                                 # pylint: disable=protected-access
        op._set_device(origin_op.node_def.device)                                                                       # pylint: disable=protected-access
    copied_ops = info._transformed_ops.values()                                                                         # pylint: disable=protected-access
    debug_print('Copied %s to %s', ops_to_copy, copied_ops)

    graph_editor.reroute_ts(checkpoints_disconnected.values(), checkpoints_disconnected.keys(), can_modify=copied_ops)
    debug_print('Rewired %s in place of %s restricted to %s',
                checkpoints_disconnected.values(), checkpoints_disconnected.keys(), copied_ops)

    # Get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]                                                   # pylint: disable=protected-access
    boundary = list(checkpoints_disconnected.values())
    dv = TF_GRADIENTS(ys=copied_ys,
                      xs=boundary + xs,
                      grad_ys=grad_ys,
                      aggregation_method=aggregation_method,
                      **kwargs)
    debug_print('Got gradients %s', dv)
    debug_print('for %s', copied_ys)
    debug_print('with respect to %s', boundary + xs)

    # Adding control inputs to the graph
    inputs_to_do_before = [y.op for y in ys]
    if grad_ys is not None:
        inputs_to_do_before += grad_ys
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # Partial derivatives to the checkpointed nodes
    # Dictionary of "node: backprop" for nodes in the boundary
    # Partial derivatives to xs (usually the params of the neural net)
    d_checkpoints = {r: dr for r, dr in zip(checkpoints_disconnected.keys(), dv[:len(checkpoints_disconnected)])}
    d_xs = dv[len(checkpoints_disconnected):]

    # Incorporate derivatives flowing through the checkpointed nodes
    checkpoints_sorted_lists = tf_toposort(checkpoints, within_ops=fwd_ops)
    for ts in checkpoints_sorted_lists[::-1]:
        debug_print('Processing list %s', ts)
        checkpoints_other = [r for r in checkpoints if r not in ts]
        checkpoints_disconnected_other = [checkpoints_disconnected[r] for r in checkpoints_other]

        # Copy part of the graph below current checkpoint node, stopping at other checkpoints nodes
        ops_to_copy = fast_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=checkpoints_other)
        debug_print('Found %s ops to copy within %s, seed %s, stop_at %s',
                    len(ops_to_copy), fwd_ops, [r.op for r in ts], checkpoints_other)
        debug_print('ops_to_copy = %s', ops_to_copy)

        # We are done!
        if not ops_to_copy:
            break

        _, info = graph_editor.copy_with_input_replacements(graph_editor.sgv(ops_to_copy), {})
        for origin_op, op in info._transformed_ops.items():                                                             # pylint: disable=protected-access
            op._set_device(origin_op.node_def.device)                                                                   # pylint: disable=protected-access
        copied_ops = info._transformed_ops.values()                                                                     # pylint: disable=protected-access
        debug_print('Copied %s to %s', ops_to_copy, copied_ops)
        graph_editor.reroute_ts(checkpoints_disconnected_other, checkpoints_other, can_modify=copied_ops)
        debug_print('Rewired %s in place of %s restricted to %s',
                    checkpoints_disconnected_other, checkpoints_other, copied_ops)

        # Gradient flowing through the checkpointed node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]                                                # pylint: disable=protected-access
        substitute_backprops = [d_checkpoints[r] for r in ts]
        dv = TF_GRADIENTS(ys=boundary,
                          xs=checkpoints_disconnected_other + xs,
                          grad_ys=substitute_backprops,
                          aggregation_method=aggregation_method,
                          **kwargs)
        debug_print("Got gradients %s", dv)
        debug_print("for %s", boundary)
        debug_print("with respect to %s", checkpoints_disconnected_other + xs)
        debug_print("with boundary backprop substitutions %s", substitute_backprops)

        # Adding control inputs
        inputs_to_do_before = [d_checkpoints[r].op for r in ts]
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

        # Partial derivatives to the checkpointed nodes
        for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
            if dr is not None:
                if d_checkpoints[r] is None:
                    d_checkpoints[r] = dr
                else:
                    d_checkpoints[r] += dr

        # Partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(checkpoints_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = _unsparsify(d_xs_new[j])
                else:
                    d_xs[j] += _unsparsify(d_xs_new[j])

    # Returning the new gradients
    return d_xs

def tf_toposort(ts, within_ops=None):
    """ Performs a topologic sorts of all tensors
        :param ts: Tensors to sort
        :param within_ops: Optional. Only from these ops.
    """
    # pylint: disable=invalid-name
    all_ops = graph_editor.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)

    deps = {}
    for op in all_ops:
        for o in op.outputs:
            deps[o] = set(op.inputs)
    sorted_ts = toposort(deps)

    # Only keep the tensors from our original list
    ts_sorted_lists = []
    for tensor in sorted_ts:
        keep = list(set(tensor).intersection(ts))
        if keep:
            ts_sorted_lists.append(keep)

    # Returning
    return ts_sorted_lists

def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
    """ Computes ops in a backward pass in a certain part of the graph """
    bwd_ops = set(graph_editor.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)

def _unsparsify(tensor):
    """ Properly processes gradient of IndexedSlices """
    if not isinstance(tensor, tf.IndexedSlices):
        return tensor
    assert tensor.dense_shape is not None, 'Got sparse gradients of unknown shape'
    indices = tensor.indices
    while indices.shape.ndims < tensor.values.shape.ndims:
        indices = tf.expand_dims(indices, -1)
    return tf.scatter_nd(indices, tensor.values, tensor.dense_shape)

def _get_collection_checkpoints(fwd_ops, ts_all, **grad_kwargs):
    """ Returns checkpoints from a collection """
    del fwd_ops, ts_all, grad_kwargs            # Unused args
    return tf.get_collection('checkpoints')

def _get_speed_checkpoints(fwd_ops, ts_all, **grad_kwargs):
    """ Returns checkpoints to speed up performance """
    del ts_all, grad_kwargs                     # Unused args
    return graph_editor.filter_ts_from_regex(fwd_ops, 'conv2d|Conv|MatMul')

def _get_memory_checkpoints(fwd_ops, ts_all, **grad_kwargs):
    """ Returns checkpoints to improve memory usage """
    # pylint: disable=invalid-name
    # Remove very small tensors and some weird ops
    def fixdims(t):                         # tf.Dimension values are not compatible with int, convert manually
        try:
            return [int(e if e.value is not None else 64) for e in t]
        except (TypeError, ValueError):
            return [0]                      # unknown shape

    ts_all = [t for t in ts_all if (np.prod(fixdims(t.shape)) > MIN_CHECKPOINT_NODE_SIZE
                                    and 'L2Loss' not in t.name
                                    and 'entropy' not in t.name
                                    and 'FusedBatchNorm' not in t.name
                                    and 'Switch' not in t.name
                                    and 'dropout' not in t.name
                                    and 'Cast' not in t.name)]

    # Filter out all tensors that are inputs of the backward graph
    with UTIL.capture_ops() as bwd_ops:
        TF_GRADIENTS(**grad_kwargs)
    bwd_inputs = [t for op in bwd_ops for t in op.inputs]

    # List of tensors in forward graph that is in input to bwd graph
    ts_filtered = list(set(bwd_inputs).intersection(ts_all))
    debug_print('Using tensors %s', ts_filtered)

    # Trying two slightly different ways of getting bottlenecks tensors to checkpoint
    for ts in [ts_filtered, ts_all]:

        # Get all bottlenecks in the graph
        bottleneck_ts = []
        for t in ts:
            b_ops = set(graph_editor.get_backward_walk_ops(t.op, inclusive=True, within_ops=fwd_ops))
            f_ops = set(graph_editor.get_forward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))

            # Check that there are not shortcuts
            b_inp = {inp for op in b_ops for inp in op.inputs}.intersection(ts_all)
            f_inp = {inp for op in f_ops for inp in op.inputs}.intersection(ts_all)
            if not set(b_inp).intersection(f_inp) and len(b_inp) + len(f_inp) >= len(ts_all):
                bottleneck_ts.append(t)                                 # we have a bottleneck!
            else:
                debug_print('Rejected bottleneck candidate and ops %s',
                            [t] + list(set(ts_all) - set(b_inp) - set(f_inp)))

        # Success? or try again without filtering?
        if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)):             # Yes, enough bottlenecks found!
            break

    # No bottleneck tensors found
    if not bottleneck_ts:
        raise RuntimeError('Unable to find bottleneck tensors. Please provide a list of use the "speed" variant.')

    # Sort the bottlenecks
    bottlenecks_sorted_lists = tf_toposort(bottleneck_ts, within_ops=fwd_ops)
    sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

    # Save an approximately optimal number ~ sqrt(N)
    N = len(ts_filtered)
    if len(bottleneck_ts) <= np.ceil(np.sqrt(N)):
        checkpoints = sorted_bottlenecks
    else:
        step = int(np.ceil(len(bottleneck_ts) / np.sqrt(N)))
        checkpoints = sorted_bottlenecks[step::step]

    # Returning checkpoints
    return checkpoints

@contextlib.contextmanager
def capture_ops():
    """ Decorator function to capture ops creted in the block

        with capture_ops() as ops:
            # create some ops
        print(ops) # => prints ops created.
    """
    op_list = []
    scope_name = str(int(time.time() * 10 ** 6))
    with tf.name_scope(scope_name):
        yield op_list

    graph = tf.get_default_graph()
    op_list.extend(graph_editor.select_ops(scope_name + '/.*', graph=graph))

def _to_op(tensor_or_op):
    """ Returns the operation on a node if available, otherwise just the tensor withot the op """
    if hasattr(tensor_or_op, 'op'):
        return tensor_or_op.op
    return tensor_or_op

def _to_ops(iterable):
    """ Calls _to_op() on an iterable """
    if not _is_iterable(iterable):
        return iterable
    return [_to_op(i) for i in iterable]

def _is_iterable(maybe_iterable):
    """ Checks if the object is iterable or not """
    try:
        iter(maybe_iterable)
    except TypeError:
        return False
    return True

def debug_print(message, *args):
    """ Method similar to LOGGER.log, but also replaces all TF ops/tensors with their names
        e.g. debug_print('see tensors %s for %s', tensor_list, [1,2,3])
    """
    if not DEBUG_LOGGING:
        return
    LOGGER.debug(message, *[format_ops(arg) for arg in args])

def format_ops(ops, sort_outputs=True):
    """ Helper method for printing ops.
        Converts Tensor/Operation op to op.name, rest to str(op)
    """
    if hasattr(ops, '__iter__') and not isinstance(ops, str):
        formatted_ops = [(op.name if hasattr(op, 'name') else str(op)) for op in ops]
        if sort_outputs:
            return sorted(formatted_ops)
        return formatted_ops
    return ops.name if hasattr(ops, 'name') else str(ops)

def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
    """ Adds control inputs (inputs to do before) as control dependencies to wait_to_do_ops """
    for wait_op in wait_to_do_ops:
        ctrl_inp = [i for i in inputs_to_do_before if wait_op.control_inputs is None or i not in wait_op.control_inputs]
        graph_editor.add_control_inputs(wait_op, ctrl_inp)
