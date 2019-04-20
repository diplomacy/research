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
""" Tensorflow Utilities
    - Contains various utility functions for Tensorflow
    Adapted from: https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py (MIT License)
"""
import glob
import logging
import os
import sys
import numpy as np

# ----- Only place where Tensorflow should be loaded -----
import tensorflow as tf
from tensorflow.contrib import graph_editor, seq2seq                                                                    # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib import framework as contrib_framework                                                           # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables                               # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.layers import batch_norm                                                                        # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.rnn.python.ops import lstm_ops                                                                  # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq import sequence_loss as seq2seq_sequence_loss                                           # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq import AttentionWrapperState                                                            # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score, _bahdanau_score                  # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, AttentionMechanism         # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _zero_state_tensors, _compute_attention             # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.rnn import DropoutWrapper                                                                       # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder                                                   # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops.decoder import _create_zero_outputs                                          # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.seq2seq.python.ops.helper import Helper, _transpose_batch_time, _unstack_ta                     # pylint: disable=unused-import,no-name-in-module
from tensorflow.contrib.training import GreedyLoadBalancingStrategy, byte_size_load_fn                                  # pylint: disable=unused-import,no-name-in-module
from tensorflow.python import debug as tf_debug                                                                         # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.client import timeline                                                                           # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.eager import context                                                                             # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.framework import constant_op, dtypes, graph_util, ops, random_seed, tensor_shape, tensor_util    # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.framework.sparse_tensor import SparseTensor                                                      # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.layers import base, core                                                                         # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops import array_ops, gen_array_ops, check_ops, clip_ops, control_flow_ops, gen_control_flow_ops # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops import control_flow_util, data_flow_ops, embedding_ops, init_ops, math_ops, gen_math_ops     # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops import gradients, logging_ops, nn, nn_impl, nn_ops, random_ops, rnn_cell_impl, sparse_ops    # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops import state_ops, gen_user_ops, tensor_array_ops                                             # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops import variables, variable_scope                                                             # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.ops.embedding_ops import embedding_lookup                                                        # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.platform import tf_logging                                                                       # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.profiler.model_analyzer import ALL_ADVICE                                                        # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.saved_model.utils import build_tensor_info                                                       # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.saved_model import builder as saved_model_builder, signature_def_utils, tag_constants            # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME                                       # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.training import optimizer, session_run_hook                                                      # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.util import nest                                                                                 # pylint: disable=unused-import,no-name-in-module
from tensorflow.python.util.tf_export import tf_export                                                                  # pylint: disable=unused-import,no-name-in-module
import tensorflow_probability as tfp                                                                                    # pylint: disable=unused-import,no-name-in-module
from tensorflow_probability.python.distributions import categorical                                                     # pylint: disable=unused-import,no-name-in-module
# --------------------------------------------------------
from diplomacy_research.models.training.supervised.common import decay_rates
from diplomacy_research.utils.model import int_prod
from diplomacy_research.settings import BUILD_DIR, LOG_INF_NAN_TENSORS

# Constants
LOGGER = logging.getLogger(__name__)
LOGGER.info('TensorFlow has been successfully loaded.')
SO_CACHE = {}

# ----------------------------------------------------------------------------
# ----------                   SESSION HOOKS                  ----------------
# ----------------------------------------------------------------------------
class SupervisedDatasetSessionRunHook(tf.train.SessionRunHook):
    """ Performs dataset initialization and cleanup when using a MonitoredTrainingSession """

    def __init__(self, supervised_dataset):
        """ Constructor
            :param supervised_dataset: Reference to the supervised dataset object
            :type supervised_dataset: diplomacy_research.models.datasets.supervised_dataset.SupervisedDataset
        """
        self.supervised_dataset = supervised_dataset

    def after_create_session(self, session, coord):
        """ Called when a new TensorFlow session is created. (graph is already finalized)
            :param session: The TensorFlow session that has been created
            :param coord: A coordinator object which keeps track of all threads.
            :type session: tensorflow.python.client.session.Session
            :type coord: tensorflow.python.training.coordinator.Coordinator
        """
        # Initializing dataset to start of training (might be changed when status is loaded from disk)
        del coord       # Unused argument
        self.supervised_dataset.start_training_mode(session)

    def end(self, session):
        """ Called at the end of the session.
            :param session: A TensorFlow Session that will soon closed.
            :type session: tensorflow.python.client.session.Session
        """
        del session     # Unused argument
        self.supervised_dataset.save_status()
        self.supervised_dataset.close()

class QueueDatasetSessionRunHook(tf.train.SessionRunHook):
    """ Performs dataset initialization and cleanup when using a MonitoredTrainingSession """

    def __init__(self, queue_dataset):
        """ Constructor
            :param queue_dataset: Reference to the queue dataset object
            :type queue_dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
        """
        self.queue_dataset = queue_dataset

    def end(self, session):
        """ Called at the end of the session.
            :param session: A TensorFlow Session that will soon closed.
            :type session: tensorflow.python.client.session.Session
        """
        del session     # Unused argument
        self.queue_dataset.close()

class ReinforcementStartTrainingHook(tf.train.SessionRunHook):
    """ Resets the barrier and initializes variables at session creation """

    def __init__(self, trainer):
        """ Constructor
            :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
        """
        self.trainer = trainer

    def after_create_session(self, session, coord):
        """ Called when a new TensorFlow session is created. (graph is already finalized)
            :param session: The TensorFlow session that has been created
            :param coord: A coordinator object which keeps track of all threads.
            :type session: tensorflow.python.client.session.Session
            :type coord: tensorflow.python.training.coordinator.Coordinator
        """
        del coord       # Unused args

        # Creating directory
        if not os.path.exists(os.path.join(self.trainer.flags.save_dir, 'serving')):
            os.makedirs(os.path.join(self.trainer.flags.save_dir, 'serving'), exist_ok=True)

class SupervisedStartTrainingHook(tf.train.SessionRunHook):
    """ Resets the barrier and initializes variables at session creation """

    def __init__(self, trainer):
        """ Constructor
            :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
        """
        self.trainer = trainer

    def after_create_session(self, session, coord):
        """ Called when a new TensorFlow session is created. (graph is already finalized)
            :param session: The TensorFlow session that has been created
            :param coord: A coordinator object which keeps track of all threads.
            :type session: tensorflow.python.client.session.Session
            :type coord: tensorflow.python.training.coordinator.Coordinator
        """
        del coord       # Unused args

        # Initiating learning rate
        decay_rates(self.trainer, session)

        # Creating validation directory
        if not os.path.exists(os.path.join(self.trainer.flags.save_dir, 'validation')):
            os.makedirs(os.path.join(self.trainer.flags.save_dir, 'validation'), exist_ok=True)

class RestoreVariableHook(tf.train.SessionRunHook):
    """ Restores the variable using a saver """

    def __init__(self, restore_saver, checkpoint_dir, latest_filename):
        """ Constructor """
        self.restore_saver = restore_saver
        self.checkpoint_dir = checkpoint_dir
        self.latest_filename = latest_filename

    def after_create_session(self, session, coord):
        """ Called when a new TensorFlow session is created. (graph is already finalized)
            :param session: The TensorFlow session that has been created
            :param coord: A coordinator object which keeps track of all threads.
            :type session: tensorflow.python.client.session.Session
            :type coord: tensorflow.python.training.coordinator.Coordinator
        """
        del coord       # Unused args
        if not self.restore_saver or not os.path.exists(os.path.join(self.checkpoint_dir, self.latest_filename)):
            return

        # Loading using saver
        checkpoint_path = tf.train.get_checkpoint_state(self.checkpoint_dir, self.latest_filename).model_checkpoint_path        # pylint: disable=no-member
        self.restore_saver.restore(session, checkpoint_path)


# ----------------------------------------------------------------------------
# ----------                   FUNCTIONS                      ----------------
# ----------------------------------------------------------------------------
def bytes_feature(value):
    """ Builds a tf.train.Example feature with bytes
        :param value: A list of bytes to save in the feature (or a numpy array)
        :return: The tf.train.Feature of bytes
    """
    value = [value] if isinstance(value, int) else value
    try:
        value_list = bytes(value)
    except TypeError:
        value_list = bytes(np.array(value, dtype=np.uint8))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_list]))

def ints_feature(value):
    """ Builds a tf.train.Example feature with integers (int64)
        :param value: A list of integers to save in the feature
        :return: The tf.train.Feature of integers
    """
    value_list = [value] if not isinstance(value, list) else value
    value_list = np.array(value_list, np.int64).flatten().tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def floats_feature(value):
    """ Builds a tf.train.Example feature with floating point integers (floats)
        :param value: A list of floats to save in the feature
        :return: The tf.train.Feature of floats
    """
    value_list = [value] if not isinstance(value, list) else value
    value_list = np.array(value_list, np.float32).flatten().tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

def strings_feature(value):
    """ Builds a tf.train.Example feature by converting string to a list of bytes
        :param value: A string to save
        :return: The tf.train.Feature of bytes
    """
    return bytes_feature(bytes(value, 'utf-8'))

def generic_feature(value, dtype):
    """ Builds a tf.train.Example feature by converting a generic object to a list of bytes
        :param value: An object to save
        :param dtype: A numpy or tensorflow dtype
        :return: The tf.train.Feature of bytes
    """
    if dtype is None:
        return strings_feature(value)
    if not isinstance(dtype, dtypes.DType):
        dtype = np_to_tf(dtype)
    value_list = [value] if not isinstance(value, list) else value
    value_list = np.array(value_list, dtype.as_numpy_dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_list]))

def to_float(tensor):
    """ Casts the tensor to float32 """
    return tf.cast(tensor, tf.float32)

def to_int32(tensor):
    """ Casts the tensor to int32 """
    return tf.cast(tensor, tf.int32)

def to_int64(tensor):
    """ Casts the tensor to int64 """
    return tf.cast(tensor, tf.int64)

def to_uint8(tensor):
    """ Casts the tensor to uint8 """
    return tf.cast(tensor, tf.uint8)

def to_bool(tensor):
    """ Casts the tensor to bool """
    return tf.cast(tensor, tf.bool)

def gelu(tensor):
    """ Gaussian Error Linear Unit - https://arxiv.org/abs/1606.08415 """
    return 0.5 * tensor * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (tensor + 0.044715 * tf.pow(tensor, 3))))

def ensure_finite(input_tensor, name=None):
    """ Replaces NaN and Inf in the input tensor
        :param input_tensor: The input tensor to check
        :return: The tensor with NaN and Inf replaced with `0`
    """
    if not LOG_INF_NAN_TENSORS:
        return input_tensor

    def get_ix_slices_values_without_nan():
        """ Gets the indexed slice values without NaN """
        ix_slice_values_without_nans = tf.where(gen_math_ops.is_finite(input_tensor.values),
                                                input_tensor.values,
                                                gen_array_ops.zeros_like(input_tensor.values))
        print_op = logging_ops.print_v2('WARNING - Tensor %s has NaN or Inf values. %s' %
                                        (input_tensor.name, name or ''))
        with ops.control_dependencies([ix_slice_values_without_nans, print_op]):
            return array_ops.identity(ix_slice_values_without_nans)

    def get_tensor_without_nan():
        """ Gets the tensor without NaN """
        tensor_without_nans = tf.where(tf.is_finite(input_tensor), input_tensor, tf.zeros_like(input_tensor))
        print_op = logging_ops.print_v2('WARNING - Tensor %s has NaN or Inf values. %s' %
                                        (input_tensor.name, name or ''))
        with ops.control_dependencies([tensor_without_nans, print_op]):
            return array_ops.identity(tensor_without_nans)

    # Tensor
    if isinstance(input_tensor, ops.Tensor):
        return control_flow_ops.cond(math_ops.reduce_all(gen_math_ops.is_finite(input_tensor)),
                                     true_fn=lambda: input_tensor,
                                     false_fn=get_tensor_without_nan)

    # Indexed Slices
    if isinstance(input_tensor, ops.IndexedSlices):
        values = control_flow_ops.cond(math_ops.reduce_all(gen_math_ops.is_finite(input_tensor.values)),
                                       true_fn=lambda: input_tensor.values,
                                       false_fn=get_ix_slices_values_without_nan)
        return ops.IndexedSlices(values=values,
                                 indices=input_tensor.indices,
                                 dense_shape=input_tensor.dense_shape)

    # Unknown type
    return input_tensor

def sequence_loss(logits, targets, weights, average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """ Weighted cross-entropy loss for a sequence of logits. """
    return seq2seq_sequence_loss(logits=ensure_finite(logits),
                                 targets=targets,
                                 weights=weights,
                                 average_across_timesteps=average_across_timesteps,
                                 average_across_batch=average_across_batch,
                                 softmax_loss_function=softmax_loss_function,
                                 name=name)

def cross_entropy(labels, logits, name=None):
    """ Computes the cross entropy between the labels and logits
        This is a safe version that adds epsilon to logits to prevent log(0)
    """
    return nn_ops.sparse_softmax_cross_entropy_with_logits(logits=ensure_finite(logits),
                                                           labels=labels,
                                                           name=name)

def binary_cross_entropy(labels, logits, name=None):
    """ Computes the binary cross entropy between the labels and logits
        This is a safe version that adds epsilon to logits to prevent log(0)
    """
    return nn_impl.sigmoid_cross_entropy_with_logits(logits=ensure_finite(logits),
                                                     labels=labels,
                                                     name=name)

def get_empty_tensor(batch_size):
    """ Returns an empty tensor """
    if batch_size is None:
        return array_ops.zeros(shape=(), dtype=dtypes.float32, name='empty')
    return array_ops.zeros(shape=(batch_size,), dtype=dtypes.float32, name='empty')

def is_empty_tensor(tensor):
    """ Determines if the tensor is empty """
    if tensor is None:
        return True
    var_name = tensor.name.split('/')[-1]
    return bool(var_name.split('_')[0] == 'empty')

def _tile_batch(tensor, multiplier):
    """ Core single-tensor implementation of tile_batch. """
    tensor = ops.convert_to_tensor(tensor, name='t')
    shape_tensor = array_ops.shape(tensor)
    if tensor.shape.ndims is None:
        raise ValueError('tensor must have statically known rank')
    if tensor.shape.ndims == 0:             # We can't tile scalars (e.g. time)
        return tensor
    tiling = [1] * (tensor.shape.ndims + 1)
    tiling[1] = multiplier
    tiled_static_batch_size = (tensor.shape[0].value * multiplier if tensor.shape[0].value is not None else None)
    tiled = gen_array_ops.tile(array_ops.expand_dims(tensor, 1), tiling)
    tiled = gen_array_ops.reshape(tiled, array_ops.concat(([shape_tensor[0] * multiplier], shape_tensor[1:]), 0))
    tiled.set_shape(tensor_shape.TensorShape([tiled_static_batch_size]).concatenate(tensor.shape[1:]))
    return tiled

def tile_batch(tensor, multiplier, name=None):
    """ Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

        For each tensor t in a (possibly nested structure) of tensors, this function takes a tensor t
        shaped `[batch_size, s0, s1, ...]` composed of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it
        to have a shape `[batch_size * multiplier, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated `multiplier` times.

        :param tensor: `Tensor` shaped `[batch_size, ...]`.
        :param multiplier: Python int.
        :param name: Name scope for any created operations.
        :return: A (possibly nested structure of) `Tensor` shaped `[batch_size * multiplier, ...]`.
    """
    if tensor is None:
        return None
    flat_tensor = nest.flatten(tensor)
    with ops.name_scope(name, 'tile_batch', flat_tensor + [multiplier]):
        return nest.map_structure(lambda t_: _tile_batch(t_, multiplier), tensor)

def get_tile_beam(multiplier):
    """ Returns a function to tile a tensor """
    return lambda tensor: tile_batch(tensor, multiplier)

def get_tile_tensor_array(multiplier):
    """ Returns a function to tile a tensor array """
    def _tile_tensor_array(tensor_array):
        """ Function to return
            :type tensor_array: tensor_array_ops.TensorArray
        """
        if tensor_array is None:
            return None

        counter = tf.constant(0)
        tensor_size = tensor_array.size()
        init_tensor_array = tf.TensorArray(dtype=tensor_array.dtype,
                                           size=0,
                                           dynamic_size=True,
                                           clear_after_read=False,
                                           infer_shape=True,
                                           element_shape=tensor_array._element_shape[0])                                # pylint: disable=protected-access

        def cond(time, loop_ta):
            """ Checks if we need to stop """
            del loop_ta  # Unused arg
            return tf.less(time, tensor_size)

        def body(time, loop_ta):
            """ Tile the item in the tensor array """
            loop_ta = loop_ta.write(time, tile_batch(tensor_array.read(time), multiplier))
            return time + 1, loop_ta

        _, new_tensor_array = control_flow_ops.while_loop(cond, body,
                                                          loop_vars=(counter, init_tensor_array),
                                                          back_prop=False)
        return new_tensor_array

    # Returning the function
    return _tile_tensor_array

def np_to_tf(np_dtype):
    """ Converts numpy dtype to tf dtype """
    if isinstance(np_dtype, dtypes.DType):
        raise TypeError('Numpy dtype %s is already a Tensorflow type.' % np_dtype)
    if np_dtype is None:
        return tf.string

    dtype_name = np.dtype(np_dtype).name
    if dtype_name == 'object':
        return tf.string

    if dtype_name in ['float16', 'float32', 'float64', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                      'int16', 'int8', 'complex64', 'complex128', 'bool']:
        return getattr(tf, dtype_name)
    raise TypeError('Unsupported Numpy DType %s' % np_dtype)

def normalize(inputs):
    """ Normalize a tensor by substracting its mean and dividing by its std dev """
    def _normalize(inputs):
        """ Normalize inputs """
        mean, var = tf.nn.moments(inputs, axes=[0])
        normalized = tf.cond(tf.equal(var, 0.),
                             true_fn=lambda: inputs,
                             false_fn=lambda: (inputs - mean) / tf.sqrt(var))
        with tf.control_dependencies([normalized]):
            return tf.identity(normalized)

    # No need to normalize if we don't have 8 items in a batch
    return tf.cond(tf.greater_equal(tf.shape(inputs)[0], 8),
                   true_fn=lambda: _normalize(inputs),
                   false_fn=lambda: inputs)

def pad_axis(tensor, axis, min_size):
    """ Pads the tensor so that the specified axis as at least the minimum size
        :param tensor: The tensor to pad.
        :param axis: The axis to pad (e.g. 0, or -1)
        :param min_size: The minimum axis size
        :return: The padded tensor
    """
    nb_dims = len(tensor.shape)
    axis = nb_dims + axis if axis < 0 else axis
    axis_dim = tensor.shape.as_list()[axis]

    # Axis has fixed dimension
    if axis_dim is not None:
        if axis_dim >= min_size:
            return tensor
        paddings = [(0, 0) if axis_ix != axis else (0, min_size - axis_dim) for axis_ix in range(nb_dims)]
        return tf.pad(tensor, paddings, constant_values=tf.cast(0, tensor.dtype))

    # Axis has variable dimension
    def padded_tensor():
        """ Computes the padded tensor """
        paddings = [(0, 0) if axis_ix != axis else (0, min_size - tf.shape(tensor)[axis]) for axis_ix in range(nb_dims)]
        pad_op = tf.pad(tensor, paddings, constant_values=tf.cast(0, tensor.dtype))
        with tf.control_dependencies([pad_op]):
            return tf.identity(pad_op)

    return tf.cond(tf.shape(tensor)[axis] >= min_size,
                   true_fn=lambda: tensor,
                   false_fn=padded_tensor)

def build_sparse_batched_tensor(batch_indices, value, dtype, dense_shape):
    """ Creates a SparsedTensor from batched indices, values, and the shape from proto field.
        :param batch_indices: A tensor of shape [batch, nb_points, nb_dims] - dtype is tf.int64
        :param value: The value to set to all indices
        :param dtype: The dtype of the value
        :param dense_shape: The shape of the unbatched Sparse Tensor - 1D of length nb_dims
        :return: A SparseTensor of dimension nb_dims + 1 with batch_size as the first dimension
        :type proto_field: diplomacy_research.models.datasets.base_builder.SparseProtoField
    """
    # Source: https://stackoverflow.com/questions/42147362/
    batch_size, nb_points, _ = tf.unstack(tf.shape(batch_indices))
    index_range_tiled = tf.tile(tf.range(batch_size)[..., None],
                                tf.stack([1, nb_points]))[..., None]

    # Merging indices
    merged_indices = tf.reshape(tf.concat([tf.cast(index_range_tiled, tf.int64), batch_indices], axis=2),
                                [-1, 1 + tf.size(dense_shape)])

    # Removing rows with (x, 0, 0, 0)
    merged_indices = tf.boolean_mask(merged_indices,
                                     tf.math.greater(tf.reduce_sum(merged_indices[:, 1:], axis=-1), 1))
    merged_values = tf.fill([tf.shape(merged_indices)[0]], tf.cast(value, dtype))

    # Sorting and returning
    sparse_tensor = tf.SparseTensor(merged_indices,
                                    merged_values,
                                    dense_shape=tf.concat([[tf.cast(batch_size, tf.int64)],
                                                           tf.cast(dense_shape, tf.int64)], axis=0))
    return tf.sparse.reorder(sparse_tensor)

def scope_vars(scope, trainable_only=False):
    """ Get the variables inside a scope
        :param scope: The name of the scope or a VariableScope object
        :param trainable_only: Boolean flag that indicates to only return the trainable variables
        :return: All the variables in the specified scope
    """
    graph_keys = tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES
    target_scope = scope if isinstance(scope, str) else scope.name
    return tf.get_collection(graph_keys, scope=target_scope)

def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name

def get_placeholder(name, dtype, shape, for_summary=False):
    """ Creates or returns a placeholder from cache
        :param name: The name of the placeholder
        :param dtype: The dtype of the placeholder
        :param shape: The shape of the placeholder
        :param for_summary: Boolean that indicates that the placeholder is for summary.
        :return: The cached placeholder or a new placeholder if it doesn't exist
    """
    # In cache, returning
    collection_name = 'pholder_summary_{}' if for_summary else 'placeholder_{}'
    cached_placeholders = tf.get_collection(collection_name.format(name))
    if cached_placeholders:
        cached_placeholder = cached_placeholders[0]
        assert cached_placeholder.dtype == dtype and cached_placeholder.shape.ndims == len(shape)
        return cached_placeholder

    # Creating
    placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
    tf.add_to_collection(collection_name.format(name), placeholder)
    return placeholder

def get_placeholder_with_default(name, default_value, shape, dtype=None):
    """ Creates or returns a placeholder (with default value) from cache
        :param name: The name of the placeholder
        :param default_value: The default value the placeholder will have when it is not in the feed dict
        :param shape: The shape of the placeholder
        :param dtype: Optional. The data type of the default value.
        :return: The cached placeholder or a new placeholder if it doesn't exist
    """
    if 'dropout' in name:
        assert default_value == 0, 'The placeholder "%s" must have a default value of 0.' % name

    # In cache, returning
    cached_placeholders = tf.get_collection('placeholder_{}'.format(name))
    if cached_placeholders:
        cached_placeholder = cached_placeholders[0]
        assert cached_placeholder.shape.ndims == len(shape)
        return cached_placeholder

    # Creating
    default_tensor = tf.convert_to_tensor(default_value, dtype=dtype)
    placeholder = tf.placeholder_with_default(default_tensor, shape=shape, name=name)
    tf.add_to_collection('placeholder_{}'.format(name), placeholder)
    return placeholder

def flatten_all_but_first(tensor):
    """ Flattens a tensor to only 2 dimensions (e.g. [1,2,3,4] -> [1, 24])
        :param tensor: The tensor to flatten
        :return: The 2D flatten tensor
    """
    return tf.reshape(tensor, [-1, int_prod(tensor.get_shape().as_list()[1:])])

def load_so_by_name(so_name):
    """ Load a compiled library given library name (without extension). """
    if so_name in SO_CACHE:
        return SO_CACHE[so_name]

    # Looking in the build folder
    for so_path in glob.glob(os.path.join(BUILD_DIR, 'lib*', '%s.*.so' % so_name)):
        LOGGER.info('Loading "%s" from path: "%s"', so_name, so_path)
        SO_CACHE[so_name] = tf.load_op_library(so_path)
        return SO_CACHE[so_name]

    # Otherwise looking in sys.path
    for path in sys.path:
        for so_path in glob.glob(os.path.join(path, '%s.*.so' % so_name)):
            LOGGER.info('Loading "%s" from path: "%s"', so_name, so_path)
            SO_CACHE[so_name] = tf.load_op_library(so_path)
            return SO_CACHE[so_name]

    # Otherwise, raising FileNotFoundError
    raise FileNotFoundError('Unable to find a .so that starts with %s on your path.' % so_name)
