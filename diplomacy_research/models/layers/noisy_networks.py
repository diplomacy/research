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
""" Noisy Networks
    - Converts variables in a graph to their noisy equivalent
"""
from math import sqrt
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import tf
from diplomacy_research.utils.tensorflow import graph_editor


def convert_to_noisy_variables(variables, activation=None):
    """ Converts a list of variables to noisy variables
        :param variables: A list of variables to make noisy
        :param activation: Optional. The activation function to use on the linear noisy transformation
        :return: Nothing, but modifies the graph in-place

        Reference: 1706.10295 - Noisy Networks for exploration
    """
    if tf.get_collection(tf.GraphKeys.TRAIN_OP):
        raise RuntimeError('You must call convert_to_noisy_variables before applying an optimizer on the graph.')

    graph = tf.get_default_graph()
    if not isinstance(variables, list):
        variables = list(variables)

    # Replacing each variable
    for variable in variables:
        variable_read_op = _get_variable_read_op(variable, graph)
        variable_outputs = _get_variable_outputs(variable_read_op, graph)
        variable_scope = variable.name.split(':')[0]
        variable_shape = variable.shape.as_list()
        fan_in = variable_shape[0]

        # Creating noisy variables
        with tf.variable_scope(variable_scope + '_noisy'):
            with tf.device(variable.device):
                s_init = tf.constant_initializer(0.5 / sqrt(fan_in))

                noisy_u = tf.identity(variable, name='mu')
                noisy_s = tf.get_variable(name='sigma',
                                          shape=variable.shape,
                                          dtype=tf.float32,
                                          initializer=s_init,
                                          caching_device=variable._caching_device)      # pylint: disable=protected-access
                noise = tf.random.normal(shape=variable_shape)

                replaced_var = noisy_u + noisy_s * noise
                replaced_var = activation(replaced_var) if activation else replaced_var

        # Replacing in-place
        inputs_index = [var_index for var_index, var_input in enumerate(graph_editor.sgv(*variable_outputs).inputs)
                        if var_input.name.split(':')[0] == variable_read_op.name.split(':')[0]]
        graph_editor.connect(graph_editor.sgv(replaced_var.op),
                             graph_editor.sgv(*variable_outputs).remap_inputs(inputs_index),
                             disconnect_first=True)

def _get_variable_read_op(variable, graph):
    """ Returns the /read operation for a variable """
    return graph.get_operation_by_name(variable.name.split(':')[0] + '/read')

def _get_variable_outputs(variable_read_op, graph):
    """ Returns the list of tensors that have the variable as input """
    outputs = []
    for graph_op in graph.get_operations():
        for var_input in graph_op.inputs._inputs:                                       # pylint: disable=protected-access
            if var_input in variable_read_op.outputs:
                outputs += [graph_op]
    return outputs
