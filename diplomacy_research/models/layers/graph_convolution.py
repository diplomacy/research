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
""" Graph Convolution
    - Implements the graph convolution algorithm (ArXiv 1609.02907)
    - Partially adapted from https://github.com/tkipf/gcn/ (MIT License)
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

import numpy as np
from diplomacy_research.models.layers.initializers import he, zeros
from diplomacy_research.models.state_space import NB_NODES
from diplomacy_research.utils.tensorflow import tf, batch_norm

# Method described in - SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
# ArXiv 1609.02907
def preprocess_adjacency(adjacency_matrix):
    """ Symmetrically normalize the adjacency matrix for graph convolutions.
        :param adjacency_matrix: A NxN adjacency matrix
        :return: A normalized NxN adjacency matrix
    """
    # Computing A^~ = A + I_N
    adj = adjacency_matrix
    adj_tilde = adj + np.eye(adj.shape[0])

    # Calculating the sum of each row
    sum_of_row = np.array(adj_tilde.sum(1))

    # Calculating the D tilde matrix ^ (-1/2)
    d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculating the normalized adjacency matrix
    norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return np.array(norm_adj, dtype=np.float32)

class GraphConvolution():
    """ Performs a graph convolution (ArXiV 1609.02907) """
    # pylint: disable=too-few-public-methods, too-many-arguments

    def __init__(self, input_dim, output_dim, norm_adjacency, activation_fn=tf.nn.relu, residual=False, bias=False,
                 scope=None, reuse=None):
        """ Initializes the graph convolutional network
            :param input_dim: The number of features per node in the input
            :param output_dim: The number of features per node desired in the output
            :param norm_adjacency: [PLACEHOLDER] The sparse normalized adjacency matrix (NxN matrix)
            :param activation_fn: The activation function to use after the graph convolution
            :param residual: Use residual connection or not.
            :param bias: Boolean flag that indicates we also want to include a bias term
            :param scope: Optional. The scope to use for this layer
            :param reuse: Optional. Boolean. Whether or not the layer and its variables should be reused.
        """
        self.activation_fn = activation_fn if activation_fn is not None else lambda x: x
        self.norm_adjacency = norm_adjacency
        self.bias = bias
        self.var_w, self.var_b = None, None
        self.residual = residual

        # Initializing variables
        with tf.variable_scope(scope, 'GraphConv', reuse=reuse):
            self.var_w = he('W', [NB_NODES, input_dim, output_dim])
            if self.bias:
                self.var_b = zeros('b', [output_dim])

    def __call__(self, inputs):
        """ Actually performs the graph convolution
            :param inputs: The input feature matrix
            :return: The activated output matrix
        """
        # Performs the convolution
        pre_act = tf.transpose(inputs, perm=[1, 0, 2])              # (b, N, in )               => (N, b, in )
        pre_act = tf.matmul(pre_act, self.var_w)                    # (N, b, in) * (N, in, out) => (N, b, out)
        pre_act = tf.transpose(pre_act, perm=[1, 0, 2])             # (N, b, out)               => (b, N, out)
        pre_act = tf.matmul(self.norm_adjacency, pre_act)           # (b, N, N) * (b, N, out)   => (b, N, out)

        # Adds the bias
        if self.bias:
            pre_act += self.var_b                                   # (b, N, out) + (1,1,out) => (b, N, out)

        # Applying activation fn and residual connection
        post_act = self.activation_fn(pre_act)
        if self.residual:
            post_act += inputs
        return post_act                                             # (b, N, out)

def film_gcn_res_block(inputs, gamma, beta, gcn_out_dim, norm_adjacency, is_training, residual=True,
                       activation_fn=tf.nn.relu):
    """ Following the design here https://arxiv.org/pdf/1709.07871.pdf
        :param inputs: A tensor of [b, NB_NODES, gcn_in_dim]
        :param gamma: A tensor of [b, 1, gcn_out_dim]. Used for film
        :param beta: A tensor of [b, 1, gcn_out_dim]. Used for film
        :param gcn_out_dim: number of output channels of graph conv
        :param norm_adjacency: The adjacency matrix for graph conv
        :param is_training: a flag for train/test behavior.
        :param residual: Use residual connection or not.
        :param activation_fn: The activation function on the output. default: relu
        :return: A tensor of [b, NB_NODES, gcn_out_dim]
    """
    gcn_in_dim = inputs.shape.as_list()[-1]
    assert gcn_in_dim == gcn_out_dim or not residual, 'For residual blocks, the in and out dims must be equal'

    gcn_result = GraphConvolution(input_dim=gcn_in_dim,
                                  output_dim=gcn_out_dim,
                                  norm_adjacency=norm_adjacency,
                                  activation_fn=None,
                                  residual=False,
                                  bias=True)(inputs)
    gcn_bn_result = batch_norm(gcn_result, is_training=is_training, fused=True)
    film_result = gamma * gcn_bn_result + beta

    # Applying activation fn and residual connection
    if activation_fn is not None:
        film_result = activation_fn(film_result)
    if residual:
        film_result += inputs
    return film_result
