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
""" Layers
    - Contains custom TensorFlow Layers objects
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import base
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import nn
from diplomacy_research.utils.tensorflow import tensor_shape


class Identity(base.Layer):
    """ Identity Layer """

    def __init__(self, dtype, name=None, **kwargs):
        """ Constructor
            :param dtype: The output dtype
            :param name: The name of the layer (string).
            :param kwargs: Optional keyword arguments.
        """
        super(Identity, self).__init__(dtype=dtype, name=name, **kwargs)

    def call(self, inputs, **kwargs):
        """ Returns inputs """
        return inputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shape given the input shape """
        return input_shape


class MultiLayers(base.Layer):
    """ Multi Layer Class - Applies multiple layers sequentially """

    def __init__(self, layers, **kwargs):
        """ Constructor
            :param layers: A list of layers to apply
            :param kwargs: Optional. Keyword arguments
            :type layers: List[base.Layer]
        """
        super(MultiLayers, self).__init__(**kwargs)
        self.layers = layers

    def call(self, inputs, **kwargs):
        """ Sequentially calls each layer with the output of the previous one """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shape given the input shape """
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape


class Dropout(base.Layer):
    """ Modified Dropout Class """

    def __init__(self, keep_prob=0.5, noise_shape=None, seed=None, name=None, **kwargs):
        """ Constructor
            :param keep_prob: The keep probability, between 0 and 1.
            :param noise_shape: 1D tensor of type `int32` representing the shape of the binary dropout mask that will
                                be multiplied with the input.
            :param seed: A Python integer. Used to create random seeds.
            :param name: The name of the layer (string).
            :param kwargs: Optional keyword arguments.
        """
        super(Dropout, self).__init__(name=name, **kwargs)
        self.keep_prob = keep_prob
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, **kwargs):
        """ Applies dropout """
        return nn.dropout(inputs, self.keep_prob, noise_shape=self.noise_shape, seed=self.seed)

    def compute_output_shape(self, input_shape):
        """ Computes the output shape given the input shape """
        return input_shape

class MatMul(base.Layer):
    """ Matrix Multiplication class """

    def __init__(self, proj_matrix, transpose_a=False, transpose_b=False, name=None, **kwargs):
        """ Constructor
            :param proj_matrix: The proj_matrix
            :param transpose_a: Boolean that indicates to transpose the first matrix in the matrix multiplication.
            :param transpose_b: Boolean that indicates to transpose the second matrix in the matrix multiplication.
            :param name: The name of the layer (string).
            :param kwargs: Optional keyword arguments.
        """
        super(MatMul, self).__init__(name=name, **kwargs)
        self.proj_matrix = proj_matrix
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def call(self, inputs, **kwargs):
        """ Performs the matrix multiplication """
        return math_ops.matmul(inputs, self.proj_matrix, transpose_a=self.transpose_a, transpose_b=self.transpose_b)

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        output_size = self.proj_matrix.shape[0].value if self.transpose_b else self.proj_matrix.shape[1].value
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(output_size)
