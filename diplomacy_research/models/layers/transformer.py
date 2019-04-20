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
""" Transformer
    - Contains the code to implement a transformer cell
"""
# Adapted from: https://github.com/openai/gpt-2/blob/master/src/model.py - MIT License
import collections
import logging
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import pad_axis
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops, gen_array_ops
from diplomacy_research.utils.tensorflow import contrib_framework
from diplomacy_research.utils.tensorflow import core
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import embedding_lookup
from diplomacy_research.utils.tensorflow import gelu
from diplomacy_research.utils.tensorflow import init_ops
from diplomacy_research.utils.tensorflow import math_ops, gen_math_ops
from diplomacy_research.utils.tensorflow import nn_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import rnn_cell_impl
from diplomacy_research.utils.tensorflow import variable_scope

# Constants
LOGGER = logging.getLogger(__name__)

def _get_embedding_fn(embedding):
    """ Returns a callable embedding function """
    return embedding if callable(embedding) else (lambda ids: embedding_lookup(embedding, ids))

class TransformerCellState(
        collections.namedtuple('TransformerCellState', ('past_attentions',      # Past attentions
                                                        'feeder_state',         # The state of the feeder cell
                                                        'time'))):              # The current time step
    """ `namedtuple` storing the state of a `TransformerCellState`. """

    def clone(self, **kwargs):
        """ Clone this object, overriding components provided by kwargs. """
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return contrib_framework.with_same_shape(old, new)
            return new
        return nest.map_structure(with_same_shape, self, super(TransformerCellState, self)._replace(**kwargs))

class TransformerCell(rnn_cell_impl.RNNCell):
    """ Transformer RNN cell that conditions optionally on a previous context and feeder cell
        This is a uni-directional implementation, where attention is only on the current and previous outputs
    """

    def __init__(self, nb_layers, nb_heads, word_embedding, position_embedding, batch_size, feeder_cell=None,
                 feeder_init_state=None, past_attentions=None, context=None, context_word_embedding=None,
                 past_seq_lengths=None, scope=None, name=None):
        """ Uni-directional transformer cell
            :param nb_layers: The number of layers to use
            :param nb_heads: The number of attention heads to use
            :param word_embedding: The word embedding vector - [vocab_size, emb_size]
            :param position_embedding: The position embedding vector - [context_size, emb_size]
            :param batch_size: The current batch size - Scalar
            :param feeder_cell: Optional. An other RNN cell that returns additional inputs to condition on.
            :param feeder_init_state: Optional. The initial state of the feeder.
            :param past_attentions: Optional. The past_attention tensor from the final state of another cell.
            :param context: Optional. A list of tokens to use as context (initial input of the sequence) - (b, seq_len)
            :param context_word_embedding: The word embedding vector to embed words in the context - [ctxt_voc, emb_sz]
            :param past_seq_lengths: Optional. The len of each item in the past_attentions or context (b,)
            :param scope: Optional. Scope to use to properly share parameters.
            :param name: Optional scope name.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(TransformerCell, self).__init__(name=name)

        # Setting values
        self._nb_layers = nb_layers
        self._nb_heads = nb_heads
        self._word_embedding = word_embedding                                               # [vocab_size, emb_size]
        self._position_embedding_fn = _get_embedding_fn(position_embedding)
        self._feeder_cell = feeder_cell
        self._feeder_init_state = feeder_init_state
        self._past_attns = past_attentions
        self._context = None
        self._context_word_embedding_fn = lambda ids: ids
        self._past_seq_lengths = past_seq_lengths
        self._scope = scope

        # Infering shapes
        self._batch_size = batch_size
        self._vocab_size = word_embedding.shape.as_list()[0]
        self._emb_size = word_embedding.shape.as_list()[1]
        self._position_emb_size = position_embedding.shape.as_list()[0]
        assert self._emb_size % self._nb_heads == 0, 'The embedding size must be perfectly divisible by the nb of heads'

        # Cannot provide both a context and a past_attentions tensor array
        assert past_attentions is None or context is None, 'Cannot provide both a context and past_attentions'

        # Validating context
        if context is not None:
            assert context_word_embedding is not None, 'Arg "context_word_embedding" is required when context provided.'
            if feeder_cell is not None:
                LOGGER.warning('The feeder cell will not be applied on the context.')
                LOGGER.warning('Otherwise, use "past_attentions" on the output of another transformer cell.')
            self._context = context
            self._context_word_embedding_fn = _get_embedding_fn(context_word_embedding)

    @property
    def output_size(self):
        """ Returns the cell output size """
        return self._emb_size

    @property
    def state_size(self):
        """ The `state_size` property of `TransformerCell` """
        past_attns_shape = [self._nb_layers, 2, self._nb_heads, None, self._emb_size // self._nb_heads]
        feeder_state_shape = tensor_shape.TensorShape([]) if self._feeder_cell is None else self._feeder_cell.state_size
        return TransformerCellState(past_attentions=tensor_shape.TensorShape(past_attns_shape),
                                    feeder_state=feeder_state_shape,
                                    time=tensor_shape.TensorShape([]))

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `IdentityCell`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: A zeroed out scalar representing the initial state of the cell.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._feeder_cell is None:
                feeder_init_state = array_ops.zeros([], dtype=dtype)
            elif self._feeder_init_state is not None:
                feeder_init_state = self._feeder_init_state
            else:
                feeder_init_state = self._feeder_cell.zero_state(batch_size, dtype)

            # Empty past attentions
            if self._past_attns is None:
                head_size = self._emb_size // self._nb_heads
                past_attns_shape = [batch_size, self._nb_layers, 2, self._nb_heads, 0 * batch_size, head_size]
                self._past_attns = array_ops.zeros(past_attns_shape, dtype=dtypes.float32)

            # No Context - Returning a zero past attention
            if self._context is None:
                return TransformerCellState(past_attentions=self._past_attns,
                                            feeder_state=feeder_init_state,
                                            time=array_ops.zeros([], dtype=dtypes.int32))

            # Context provided - Computing attention by running a single block step
            _, present_attns, _ = self._step(inputs=self._context_word_embedding_fn(self._context),
                                             past_attns=self._past_attns,
                                             time=0,
                                             feeder_cell=None,
                                             feeder_state=None)
            return TransformerCellState(past_attentions=present_attns,
                                        feeder_state=feeder_init_state,
                                        time=array_ops.zeros([], dtype=dtypes.int32))

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):  # pylint: disable=arguments-differ
        """ Runs the identity cell
            :param inputs: Tensor of shpae - [batch, emb_size]
            :param state: The state from the previous time step.
            :return: The cell output and the next state
        """
        if not isinstance(state, TransformerCellState):
            raise TypeError('Expected state to be instance of TransformerCellState. Received type %s instead.'
                            % type(state))

        past_attns, feeder_state, time = state
        cell_outputs, present_attns, next_feeder_state = self._step(inputs[:, None, :], past_attns, time,
                                                                    feeder_cell=self._feeder_cell,
                                                                    feeder_state=feeder_state)
        cell_outputs = array_ops.squeeze(cell_outputs, axis=1)

        # Updating feeder cell state - 'feeder' processing mode.
        if self._feeder_cell is not None and hasattr(self._feeder_cell, 'update_state'):
            next_feeder_state = getattr(self._feeder_cell, 'update_state')(time, cell_outputs, next_feeder_state)

        # Computing next_state
        next_state = TransformerCellState(past_attentions=array_ops.concat([past_attns, present_attns], axis=-2),
                                          feeder_state=next_feeder_state,
                                          time=time + 1)
        return cell_outputs, next_state

    def _step(self, inputs, past_attns, time, feeder_cell, feeder_state):
        """ Performs the block operation on n-layers
            :param inputs: The tensor inputs (embedding of each word) - [batch, seq_len, emb_size]
            :param past_attns: The past attentions - [batch, nb_layers, 2, nb_heads. past_length, emb_size // nb_heads]
            :param time: A tensor representing the current time step
            :param feeder_cell: None or A feeder cell that returns a RNN cell output to use for conditioning
            :param feeder_state: None or the initial state of the feeder cell
            :param name: Name of the scope - To share weights between calls
            :return: A tuple consisting of:
                        1) The cell outputs - [batch, seq_len, emb_size]
                        2) The present attention - [batch, nb_layers, 2, nb_heads. seq_len, emb_size // nb_heads]
                        3) The new state of the feeder cell
        """
        with variable_scope.variable_scope(self._scope, default_name='step'):
            past_length = array_ops.shape(past_attns)[-2]       # How many past attention steps we have
            seq_len = array_ops.shape(inputs)[-2]               # How many steps are we computing for the current time
            emb_size = inputs.shape[-1].value                   # The size of the embedding
            assert emb_size == self._emb_size, 'Expected an embedding size of %d' % self._emb_size

            # 1) Computing the word embedding of each token
            assert inputs.shape.ndims == 3, 'Expected [batch, seq_len, emb_size]'                   # [bz, seq, emb]
            out_h = inputs

            # 2) Computing the position embedding of each token
            # If we know the context was padded, the effective past length is the context length + nb of time steps
            if self._past_seq_lengths is not None:
                past_length = gen_math_ops.minimum(past_length, self._past_seq_lengths + time)[:, None]     # [bz, 1]
            else:
                past_length = gen_array_ops.fill([self._batch_size, 1], value=past_length)                  # [bz, 1]
            step_ix = math_ops.range(seq_len)[None, :]                                              # [1, seq_len]
            token_positions = gen_math_ops.add(past_length, step_ix)                                # [batch, seq_len]
            token_positions = gen_math_ops.minimum(self._position_emb_size - 1, token_positions)    # [batch, seq_len]
            h_pos = self._position_embedding_fn(token_positions)                                    # [bz, seq, emb]
            out_h = out_h + h_pos

            # 3) If we have a feeder cell, we also need to condition 'h' on it.
            next_feeder_state = feeder_state
            if feeder_cell is not None:
                assert feeder_state is not None, 'A feeder state is required if a feeder cell is provided.'
                assert inputs.shape[1].value == 1, 'The seq dimension must be 1 to use a feeder_cell'
                feeder_outputs, next_feeder_state = feeder_cell(array_ops.squeeze(inputs, axis=1), feeder_state)
                h_feed = feeder_outputs                                                             # [bz, feeder_sz]
                if feeder_outputs.shape[-1].value != emb_size:
                    h_feed = core.Dense(emb_size, activation=None, name='h_feed')(h_feed)           # [bz, emb]
                h_feed = gen_array_ops.tile(h_feed[:, None, :], [1, seq_len, 1])                    # [bz, seq, emb]
                out_h = out_h + h_feed

            # Transformer
            presents = []
            pasts = array_ops.unstack(past_attns, axis=1)                # list of [batch, 2, heads, past_len, head_sz]
            assert len(pasts) == self._nb_layers, 'Expected the past attention to have %d layers.' % self._nb_layers

            for layer_ix, past_attn in enumerate(pasts):
                out_h, present = self._block(out_h, past_attn, 'layer.%d' % layer_ix)
                presents += [present]
            presents = array_ops.stack(presents, axis=1)

            # Normalizing and returning
            cell_outputs = self._norm(out_h, 'norm_h')                                          # [batch, seq, emb]
            return cell_outputs, presents, next_feeder_state

    @staticmethod
    def _conv1d(inputs, output_dim, name=None):
        """ Performs 1d convolution
            :param inputs: The tensor inputs - [axis_0, ..., axis_n-1, input_dim]
            :param output_dim: The size of the 1d convoltuion
            :param name: Name of the scope - To share weights between calls
            :return: The tensors after the convolution - [axis_0, ..., axis_n-1, output_dim]
        """
        with variable_scope.variable_scope(name):
            input_dims = [array_ops.shape(inputs)[axis] if dim.value is None else dim.value
                          for axis, dim in enumerate(inputs.shape)]
            input_prefix_dims, input_last_dim = input_dims[:-1], input_dims[-1]
            weight = variable_scope.get_variable('w',
                                                 shape=[input_last_dim, output_dim],
                                                 initializer=init_ops.random_normal_initializer(0.02))
            beta = variable_scope.get_variable('b',
                                               shape=[output_dim],
                                               initializer=init_ops.constant_initializer(0.))

            inputs = gen_array_ops.reshape(inputs, [-1, input_last_dim])                    # [B, input_last_dim]
            outputs = math_ops.matmul(inputs, weight) + beta                                # [B, output_dim]
            return gen_array_ops.reshape(outputs, input_prefix_dims + [output_dim])         # [..., output_dim]

    @staticmethod
    def _norm(inputs, name, axis=-1):
        """ Applies normalization to the input tensor by normalizing to mean=0, std_dev=1, then applying a gamma, beta
            :param inputs: The tensor inputs to normalize
            :param name: Name of the scope - To share weights between calls
            :param axis: Axis to normalize. Defaults to last one.
            :return: A tensor of the same shape as inputs, but normalized and transformed
        """
        with variable_scope.variable_scope(name):
            axis_dim = inputs.shape[axis].value
            gamma = variable_scope.get_variable('gamma', [axis_dim], initializer=init_ops.constant_initializer(1.))
            beta = variable_scope.get_variable('beta', [axis_dim], initializer=init_ops.constant_initializer(0.))
            mean = math_ops.reduce_mean(inputs, axis=axis, keepdims=True)
            var = math_ops.reduce_mean(gen_math_ops.square(inputs - mean), axis=axis, keepdims=True)
            norm_inputs = (inputs - mean) * gen_math_ops.rsqrt(var + 1e-8)
            outputs = gamma * norm_inputs + beta
            return outputs

    def _mask_attn_weights(self, attn_weights):
        """ Masks the attention weights
            :param attn_weights: The attention weights - [batch, nb_head, seq_len, seq_len + past_length]
            :return: A tensor of 0 and 1. of the same shape and dtype as attn_weights
        """
        seq_len = array_ops.shape(attn_weights)[-2]
        total_len = array_ops.shape(attn_weights)[-1]

        # 1) Creating the attention mask matrix (with the lower triangle set to 1. on the right)
        # e.g. if seq_len == 3, and total_len == 10
        # the attention mask would be:       - [seq_len, total_len]
        # [[1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        num_lower = math_ops.cast(-1, dtypes.int32)
        num_upper = total_len - seq_len
        attn_mask = gen_array_ops.matrix_band_part(array_ops.ones([seq_len, total_len]), num_lower, num_upper)

        # No past_attentions/context - We just add two leading dimensions to attn_mask and can return it
        if self._past_seq_lengths is None:
            return attn_mask[None, None, :, :]

        # If we have a context with varying sequence length, we also need to mask the items after the end of sequence
        # e.g.
        # [[1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],            # => length of 3 (padded to 7) + seq_len of 3
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],            # => length of 7 (padded to 7) + seq_len of 3
        #  [1., 1., 1., 1., 1., 0., 0., 1., 1., 1.]]            # => length of 5 (padded to 7) + seq_len of 3
        #
        # The resulting attention mask would be the product of the two.
        # [[1., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
        #  [1., 1., 1., 1., 1., 0., 0., 1., 1., 1.]]
        seq_mask = array_ops.sequence_mask(self._past_seq_lengths, dtype=dtypes.float32)        # [b, max_len]
        seq_mask = pad_axis(seq_mask, axis=-1, min_size=total_len)                              # [b, total_len]

        # Returning the multiplication of the two masks
        return gen_math_ops.mul(attn_mask[None, None, :, :], seq_mask[:, None, None, :])    # [b, nb_heads, seq, total]

    def _attn(self, inputs, attn_dim, past_attn, name):
        """ Performs multi-head attention inside a transformer block
            :param inputs: The tensor inputs - [batch, seq_len, emb_size]
            :param attn_dim: The dimension of the attention (and output)
            :param past_attn: The past attention - [batch, 2, nb_heads. seq_len, emb_size // nb_heads]
            :param name: Name of the scope - To share weights between calls
            :return: A tuple consisting of:
                    1) The output of the attention - [batch, seq_len, attn_dim]
                    2) The present attention - [batch, 2, nb_heads, seq_len, emb_size // nb_heads]
        """
        assert inputs.shape.ndims == 3, 'Expected [batch, seq_len, emb_size]'
        with variable_scope.variable_scope(name):

            # Computing the query, key, and value vectors
            query_keys_values = self._conv1d(inputs, 3 * attn_dim, 'attn_fc1')      # [batch, seq_len, 3 * attn_dim]
            query, keys, values = array_ops.split(query_keys_values, 3, axis=-1)    # 3x [batch, seq_len, attn_dim]

            # Splitting into nb_heads of size attn_dim // nb_heads
            # Output format is [batch, nb_heads, seq_len, attn_dim // nb_heads]
            query = self._split_in_heads(query)                                     # [bz, nb_heads, seq_len, head_sz]
            keys = self._split_in_heads(keys)                                       # [bz, nb_heads, seq_len, head_sz]
            values = self._split_in_heads(values)                                   # [bz, nb_heads, seq_len, head_sz]
            head_size = query.shape[-1].value

            # Stacking keys and values to get the present_attn
            present_attn = array_ops.stack([keys, values], axis=1)              # [bz, 2, nb_heads, seq_len, head_sz]

            # Adding past_attn to keys and values
            past_keys, past_values = array_ops.unstack(past_attn, 2, axis=1)    # 2x [bz, nb_heads, past_len, head_sz]
            keys = array_ops.concat([past_keys, keys], axis=-2)                 # [bz, nb_heads, total_len, head_sz]
            values = array_ops.concat([past_values, values], axis=-2)           # [bz, nb_heads, total_len, head_sz]

            # Performing multi-head attention
            attn_w = math_ops.matmul(query, keys, transpose_b=True)             # [bz. nb_heads, seq_len, total_len]
            attn_w = attn_w * gen_math_ops.rsqrt(math_ops.cast(head_size, attn_w.dtype) + 1e-8)
            attn_mask = self._mask_attn_weights(attn_w)                         # [bz, 1, seq_len, total_len]
            attn_w = attn_w * attn_mask + math_ops.cast(1e-10, attn_w.dtype) * (1. - attn_mask)
            attn_w = nn_ops.softmax(attn_w)
            attn = math_ops.matmul(attn_w, values)                              # [bz, nb_heads, seq_len, head_sz]

            # Merging attention heads, then 1d conv before returning
            out_attn = self._merge_heads(attn)                                  # [bz, seq_len, attn_dim]
            out_attn = self._conv1d(out_attn, attn_dim, 'attn_fc2')             # [bz, seq_len, attn_dim]

            # Returning
            return out_attn, present_attn

    def _block(self, inputs, past_attn, name):
        """ Computes a transformer block
            :param inputs: The inputs tensor - [batch, seq_len, emb_dim]
            :param past_attn: The past attention - [batch, 2, nb_heads, emb_size // nb_heads]
            :param name: Name of the scope - To share weights between calls
            :return: A tuple consisting of:
                    1) The output of the transformer block - [batch, seq_len, emb_dim]
                    2) The present attention - [batch, 2, nb_heads, 1, emb_size // nb_heads]
        """
        with variable_scope.variable_scope(name):
            input_dim = inputs.shape[-1].value
            h_out = inputs
            h_attn, present_attn = self._attn(self._norm(h_out, 'block_norm1'), input_dim, past_attn, 'block_attn')
            h_out = h_out + h_attn
            h_mlp = self._linear(self._norm(h_out, 'block_norm2'), input_dim * 4, 'block_mlp')
            h_out = h_out + h_mlp
            return h_out, present_attn

    def _linear(self, inputs, proj_dim, name):
        """ Computes a linear unit inside a full block - Projects to 'proj_dim' and back to 'input_dim'
            :param inputs: The inputs tensor -  [axis_0, ..., axis_n-1, input_dim]
            :param proj_dim: The dimension of the projection
            :param name: Name of the scope - To share weights between calls
            :return: A tensor of shape -  [axis_0, ..., axis_n-1, input_dim]
        """
        with variable_scope.variable_scope(name):
            input_dim = inputs.shape[-1].value
            output_h1 = gelu(self._conv1d(inputs, proj_dim, 'mlp_fc1'))
            output_h2 = self._conv1d(output_h1, input_dim, 'mlp_fc2')
            return output_h2

    def _split_in_heads(self, inputs):
        """ Splits the tensor into heads of size attn_dim / heads
            :param inputs: The tensor to split - [batch, seq_len, attn_dim]
            :return: A tensor in the format - [batch, nb_heads, seq_len, attn_dim // nb_heads]
        """
        assert inputs.shape.ndims == 3, 'Expected inputs to be [batch, seq_len, attn_dim]'
        attn_dim = inputs.shape[-1].value
        assert attn_dim % self._nb_heads == 0, 'The attn_dim must be evenly divisible by the nb of heads'

        # Reshaping to [batch, seq_len, nb_heads, head_size]
        batch_size = array_ops.shape(inputs)[0]
        seq_len = array_ops.shape(inputs)[1]
        head_size = attn_dim // self._nb_heads
        inputs = gen_array_ops.reshape(inputs, [batch_size, seq_len, self._nb_heads, head_size])

        # Transposing to [batch, nb_heads, seq_len, head_size]
        return array_ops.transpose(inputs, [0, 2, 1, 3])

    def _merge_heads(self, inputs):
        """ Merges the attn heads of the tensor into a single dimension
            :param inputs: The tensor to merge - [batch, nb_heads, seq_len, head_size]
            :return: A tensor in the format - [batch, seq_len, nb_heads * head_size]
        """
        assert inputs.shape.ndims == 4, 'Expected inputs to be [batch, nb_heads, seq_len, head_size]'
        assert inputs.shape[1].value == self._nb_heads, 'Expected the 2nd dimension to be the number of heads'

        # Transposing to [batch, seq_len, nb_heads, head_size]
        inputs = array_ops.transpose(inputs, [0, 2, 1, 3])

        # Merging last 2 dims
        batch_size = array_ops.shape(inputs)[0]
        seq_len = array_ops.shape(inputs)[1]
        head_size = inputs.shape[-1].value
        return gen_array_ops.reshape(inputs, [batch_size, seq_len, self._nb_heads * head_size])
