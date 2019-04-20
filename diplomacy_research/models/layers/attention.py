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
""" Attention
        - Contains tensorflow class to apply attention
"""
import collections
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'
import numpy as np

from diplomacy_research.models.layers.layers import Identity
from diplomacy_research.utils.tensorflow import _transpose_batch_time
from diplomacy_research.utils.tensorflow import _unstack_ta
from diplomacy_research.utils.tensorflow import AttentionWrapperState, _BaseAttentionMechanism, AttentionMechanism
from diplomacy_research.utils.tensorflow import _bahdanau_score, _maybe_mask_score, _zero_state_tensors
from diplomacy_research.utils.tensorflow import _compute_attention
from diplomacy_research.utils.tensorflow import ops, core
from diplomacy_research.utils.tensorflow import tensor_shape
from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import check_ops
from diplomacy_research.utils.tensorflow import control_flow_ops
from diplomacy_research.utils.tensorflow import contrib_framework
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import math_ops, gen_math_ops
from diplomacy_research.utils.tensorflow import nn_ops
from diplomacy_research.utils.tensorflow import nest
from diplomacy_research.utils.tensorflow import rnn_cell_impl
from diplomacy_research.utils.tensorflow import tensor_array_ops
from diplomacy_research.utils.tensorflow import variable_scope

class BahdanauAttention(_BaseAttentionMechanism):
    # Source: tensorflow/blob/r1.13/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
    """ Implements Bahdanau-style (additive) attention.

        This attention has two forms. The first is Bahdanau attention, as described in:

        Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
            "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015.
            https://arxiv.org/abs/1409.0473

        The second is the normalized form. This form is inspired by the weight normalization article:

        Tim Salimans, Diederik P. Kingma.
            "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks."
            https://arxiv.org/abs/1602.07868

        To enable the second form, construct the object with parameter `normalize=True`.
    """

    def __init__(self, num_units, memory, memory_sequence_length=None, normalize=False, probability_fn=None,
                 score_mask_value=None, dtype=None, name_or_scope='BahdanauAttention'):
        """ Construct the Attention mechanism.
            :param num_units: The depth of the query mechanism.
            :param memory: The memory to query; usually the output of an RNN encoder. This tensor should be
                           shaped `[batch_size, max_time, ...]`.
            :param memory_sequence_length: (optional): Sequence lengths for the batch entries in memory. If provided,
                                           the memory tensor rows are masked with zeros for values past the respective
                                           sequence lengths.
            :param normalize: Python boolean. Whether to normalize the energy term.
            :param probability_fn: (optional) A `callable`. Converts the score to probabilities. The default is
                                   @{tf.nn.softmax}. Other options include @{tf.contrib.seq2seq.hardmax} and
                                   @{tf.contrib.sparsemax.sparsemax}.
                                   Its signature should be: `probabilities = probability_fn(score)`.
            :param score_mask_value: (optional): The mask value for score before passing into `probability_fn`.
                                     The default is -inf. Only used if `memory_sequence_length` is not None.
            :param dtype: The data type for the query and memory layers of the attention mechanism.
            :param name_or_scope: String or VariableScope to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)

        self._num_units = num_units
        self._normalize = normalize
        self._name_or_scope = name_or_scope

        with variable_scope.variable_scope(name_or_scope, default_name='BahdanauAttention'):
            super(BahdanauAttention, self).__init__(query_layer=core.Dense(num_units,
                                                                           name='query_layer',
                                                                           use_bias=False,
                                                                           dtype=dtype),
                                                    memory_layer=core.Dense(num_units,
                                                                            name='memory_layer',
                                                                            use_bias=False,
                                                                            dtype=dtype),
                                                    memory=memory,
                                                    probability_fn=wrapped_probability_fn,
                                                    memory_sequence_length=memory_sequence_length,
                                                    score_mask_value=score_mask_value)

    def __call__(self, query, state):
        """ Score the query based on the keys and values.
            :param query: Tensor of dtype matching `self.values` and shape `[batch_size, query_depth]`.
            :param state: Tensor of dtype matching `self.values` and shape `[batch_size, alignments_size]`
                          (`alignments_size` is memory's `max_time`).
            :return: Tensor of dtype matching `self.values` and shape `[batch_size, alignments_size]`
                     (`alignments_size` is memory's `max_time`).
        """
        with variable_scope.variable_scope(self._name_or_scope, 'bahdanau_attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

class ModifiedBahdanauAttention(BahdanauAttention):
    """ Implements Bahdanau-style (additive) attention.

        This implementation doesn't use a memory layer if the memory already has num_units.
    """
    def __init__(self, num_units, memory, memory_sequence_length=None, normalize=False, probability_fn=None,
                 score_mask_value=None, dtype=None, name_or_scope='ModifiedBahdanauAttention'):
        """ Construct the Attention mechanism.
            :param num_units: The depth of the query mechanism.
            :param memory: The memory to query; usually the output of an RNN encoder. This tensor should be
                           shaped `[batch_size, max_time, ...]`.
            :param memory_sequence_length: (optional): Sequence lengths for the batch entries in memory. If provided,
                                           the memory tensor rows are masked with zeros for values past the respective
                                           sequence lengths.
            :param normalize: Python boolean. Whether to normalize the energy term.
            :param probability_fn: (optional) A `callable`. Converts the score to probabilities. The default is
                                   @{tf.nn.softmax}. Other options include @{tf.contrib.seq2seq.hardmax} and
                                   @{tf.contrib.sparsemax.sparsemax}.
                                   Its signature should be: `probabilities = probability_fn(score)`.
            :param score_mask_value: (optional): The mask value for score before passing into `probability_fn`.
                                     The default is -inf. Only used if `memory_sequence_length` is not None.
            :param dtype: The data type for the query and memory layers of the attention mechanism.
            :param name_or_scope: String or VariableScope to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)

        self._num_units = num_units
        self._normalize = normalize
        self._name_or_scope = name_or_scope

        with variable_scope.variable_scope(name_or_scope, default_name='ModifiedBahdanauAttention'):

            # Only creating memory layer if memory size != num_units
            if num_units != memory.get_shape()[-1].value:
                memory_layer = core.Dense(num_units, name='memory_layer', use_bias=False, dtype=dtype)
            else:
                memory_layer = Identity(dtype=memory.dtype)
            query_layer = core.Dense(num_units, name='query_layer', use_bias=False, dtype=dtype)

            # Building
            super(BahdanauAttention, self).__init__(query_layer=query_layer,                    # pylint: disable=bad-super-call
                                                    memory_layer=memory_layer,
                                                    memory=memory,
                                                    probability_fn=wrapped_probability_fn,
                                                    memory_sequence_length=memory_sequence_length,
                                                    score_mask_value=score_mask_value)

class AttentionWrapper(rnn_cell_impl.RNNCell):
    # Source: tensorflow/blob/r1.13/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
    """ Wraps another `RNNCell` with attention. """

    def __init__(self, cell, attention_mechanism, attention_layer_size=None, alignment_history=False,
                 cell_input_fn=None, output_attention=True, initial_cell_state=None, name_or_scope='AttentionWrapper',
                 attention_layer=None):
        """ Construct the `AttentionWrapper`.

            :param cell: An instance of `RNNCell`.
            :param attention_mechanism: A list of `AttentionMechanism` instances or a singleinstance.
            :param attention_layer_size: A list of Python integers or a single Python integer,
                                         the depth of the attention (output) layer(s).
            :param alignment_history: Python boolean, whether to store alignment history from all time steps in the
                                      final output state
            :param cell_input_fn: (optional) A `callable`. The default is: concat([inputs, attention], axis=-1)
            :param output_attention: Python bool. If `True` (default), the output at each time step is the attn value.
            :param initial_cell_state: The initial state value to use for the cell when the user calls `zero_state()`.
            :param name_or_scope: String or VariableScope to use when creating ops.
            :param attention_layer: A list of `tf.layers.Layer` instances or a single `tf.layers.Layer` instance taking
                                    the context and cell output as inputs to generate attention at each time step.
                                    If None (default), use the context as attention at each time step.

            **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in `AttentionWrapper`,
                     then you must ensure that:

            - The encoder output has been tiled to `beam_width` via `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
            - The `batch_size` argument passed to the `zero_state` method of this wrapper is equal to
                `true_batch_size * beam_width`.
            - The initial state created with `zero_state` above contains a `cell_state` value containing properly
                tiled final state from the encoder.
        """
        # pylint: disable=too-many-arguments
        self._name_or_scope = name_or_scope
        with variable_scope.variable_scope(name_or_scope, 'AttentionWrapper'):
            super(AttentionWrapper, self).__init__()
            rnn_cell_impl.assert_like_rnncell("cell", cell)

            # Attention mechanism
            if isinstance(attention_mechanism, (list, tuple)):
                self._is_multi = True
                attention_mechanisms = attention_mechanism
                for attn_mechanism in attention_mechanisms:
                    if not isinstance(attn_mechanism, AttentionMechanism):
                        raise TypeError('attention_mechanism must contain only instances of AttentionMechanism, saw '
                                        'type: %s' % type(attn_mechanism).__name__)
            else:
                self._is_multi = False
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError('attention_mechanism must be an AttentionMechanism or list of multiple '
                                    'AttentionMechanism instances, saw type: %s' % type(attention_mechanism).__name__)
                attention_mechanisms = (attention_mechanism,)

            # Cell input function
            if cell_input_fn is None:
                cell_input_fn = lambda inputs, attention: array_ops.concat([inputs, attention], -1)
            else:
                if not callable(cell_input_fn):
                    raise TypeError('cell_input_fn must be callable, saw type: %s' % type(cell_input_fn).__name__)

            # Attention layer size
            if attention_layer_size is not None and attention_layer is not None:
                raise ValueError('Only one of attention_layer_size and attention_layer should be set')

            if attention_layer_size is not None:
                attention_layer_sizes = tuple(attention_layer_size
                                              if isinstance(attention_layer_size, (list, tuple))
                                              else (attention_layer_size,))
                if len(attention_layer_sizes) != len(attention_mechanisms):
                    raise ValueError('If provided, attention_layer_size must contain exactly one integer per '
                                     'attention_mechanism, saw: %d vs %d' % (len(attention_layer_sizes),
                                                                             len(attention_mechanisms)))
                self._attention_layers = tuple(core.Dense(attention_layer_size,
                                                          name='attention_layer',
                                                          use_bias=False,
                                                          dtype=attention_mechanisms[i].dtype)
                                               for i, attention_layer_size in enumerate(attention_layer_sizes))
                self._attention_layer_size = sum(attention_layer_sizes)

            elif attention_layer is not None:
                self._attention_layers = tuple(attention_layer
                                               if isinstance(attention_layer, (list, tuple))
                                               else (attention_layer,))
                if len(self._attention_layers) != len(attention_mechanisms):
                    raise ValueError('If provided, attention_layer must contain exactly one layer per '
                                     'attention_mechanism, saw: %d vs %d' % (len(self._attention_layers),
                                                                             len(attention_mechanisms)))
                self._attention_layer_size = \
                    sum(tensor_shape.dimension_value(
                        layer.compute_output_shape([None,
                                                    cell.output_size
                                                    + tensor_shape.dimension_value(mechanism.values.shape[-1])])[-1])
                        for layer, mechanism in zip(self._attention_layers, attention_mechanisms))
            else:
                self._attention_layers = None
                self._attention_layer_size = sum(tensor_shape.dimension_value(attention_mechanism.values.shape[-1])
                                                 for attention_mechanism in attention_mechanisms)

            self._cell = cell
            self._attention_mechanisms = attention_mechanisms
            self._cell_input_fn = cell_input_fn
            self._output_attention = output_attention
            self._alignment_history = alignment_history

            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (tensor_shape.dimension_value(final_state_tensor.shape[0])
                                    or array_ops.shape(final_state_tensor)[0])
                error_message = ('When constructing AttentionWrapper %s: ' % self._base_name +
                                 'Non-matching batch sizes between the memory (encoder output) and initial_cell_state. '
                                 'Are you using the BeamSearchDecoder? You may need to tile your initial state via the '
                                 'tf.contrib.seq2seq.tile_batch function with argument multiple=beam_width.')

                with ops.control_dependencies(self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = \
                        nest.map_structure(lambda state: array_ops.identity(state, name='check_initial_cell_state'),
                                           initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        """ Checks if the batch size of each attention mechanism is correct """
        return [check_ops.assert_equal(batch_size, attention_mechanism.batch_size, message=error_message)
                for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        """ Returns `seq` as tuple or the singular element """
        if self._is_multi:
            return tuple(seq)
        return tuple(seq)[0]

    @property
    def output_size(self):
        """ Returns the cell output size """
        if self._output_attention:
            return self._attention_layer_size
        return self._cell.output_size

    @property
    def state_size(self):
        """ Returns an `AttentionWrapperState` tuple containing shapes used by this object. """
        return AttentionWrapperState(cell_state=self._cell.state_size,
                                     time=tensor_shape.TensorShape([]),
                                     attention=self._attention_layer_size,
                                     alignments=self._item_or_tuple(a.alignments_size
                                                                    for a in self._attention_mechanisms),
                                     attention_state=self._item_or_tuple(a.state_size
                                                                         for a in self._attention_mechanisms),
                                     alignment_history=self._item_or_tuple(a.alignments_size
                                                                           if self._alignment_history else ()
                                                                           for a in self._attention_mechanisms))

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `AttentionWrapper`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: AttentionWrapperState` tuple containing zeroed out tensors and, possibly, empty `TensorArrays`.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)

            error_message = ('When calling zero_state of AttentionWrapper %s: ' % self._base_name +
                             'Non-matching batch sizes between the memory encoder output) and the requested batch '
                             'size. Are you using the BeamSearchDecoder? If so, make sure your encoder output has been '
                             'tiled to beam_width via tf.contrib.seq2seq.tile_batch, and the batch_size= argument '
                             'passed to zero_state is batch_size * beam_width.')
            with ops.control_dependencies(self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(lambda state: array_ops.identity(state, name='checked_cell_state'),
                                                cell_state)
            initial_alignments = [attention_mechanism.initial_alignments(batch_size, dtype)
                                  for attention_mechanism in self._attention_mechanisms]
            return AttentionWrapperState(cell_state=cell_state,
                                         time=array_ops.zeros([], dtype=dtypes.int32),
                                         attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
                                         alignments=self._item_or_tuple(initial_alignments),
                                         attention_state=self._item_or_tuple(
                                             attention_mechanism.initial_state(batch_size, dtype)
                                             for attention_mechanism in self._attention_mechanisms),
                                         alignment_history=self._item_or_tuple(
                                             tensor_array_ops.TensorArray(dtype,
                                                                          size=0,
                                                                          dynamic_size=True,
                                                                          element_shape=alignment.shape)
                                             if self._alignment_history else ()
                                             for alignment in initial_alignments))

    def call(self, inputs, state):
        """ Performs a step of attention-wrapped RNN.

            1) Mix the `inputs` and previous step's `attention` output via `cell_input_fn`.
            2) Call the wrapped `cell` with this input and its previous state.
            3) Score the cell's output with `attention_mechanism`.
            4) Calculate the alignments by passing the score through the `normalizer`.
            5) Calculate the context vector as the inner product between the alignments and the attention_mechanism's
               values (memory).
            6) Calculate the attention output by concatenating the cell output and context through the attention
               layer (a linear layer with `attention_layer_size` outputs).

            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: An instance of `AttentionWrapperState` containing tensors from the previous time step.
            :return: A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState` containing the state calculated at this time step.
        """
        # pylint: disable=arguments-differ
        if not isinstance(state, AttentionWrapperState):
            raise TypeError('Expected state to be instance of AttentionWrapperState. Rcvd %s instead. ' % type(state))

        # Step 1: Calculate the true inputs to the cell based on the previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = tensor_shape.dimension_value(cell_output.shape[0]) or array_ops.shape(cell_output)[0]
        error_message = ('When applying AttentionWrapper %s: ' % self.name + 'Non-matching batch sizes between '
                         'the memory (encoder output) and the query (decoder output). Are you using the '
                         'BeamSearchDecoder? You may need to tile your memory input via the tf.contrib.seq2seq.'
                         'tile_batch function with argument multiple=beam_width.')

        with variable_scope.variable_scope(self._name_or_scope, 'AttentionWrapper', [inputs, state]):

            with ops.control_dependencies(self._batch_size_checks(cell_batch_size, error_message)):
                cell_output = array_ops.identity(cell_output, name='checked_cell_output')

            if self._is_multi:
                previous_attention_state = state.attention_state
                previous_alignment_history = state.alignment_history
            else:
                previous_attention_state = [state.attention_state]
                previous_alignment_history = [state.alignment_history]

            # Computing attention
            all_alignments = []
            all_attentions = []
            all_attention_states = []
            maybe_all_histories = []
            for i, attention_mechanism in enumerate(self._attention_mechanisms):
                attention, alignments, next_attention_state = _compute_attention(attention_mechanism,
                                                                                 cell_output,
                                                                                 previous_attention_state[i],
                                                                                 self._attention_layers[i]
                                                                                 if self._attention_layers else None)
                alignment_history = previous_alignment_history[i].write(state.time,
                                                                        alignments) if self._alignment_history else ()

                all_attention_states.append(next_attention_state)
                all_alignments.append(alignments)
                all_attentions.append(attention)
                maybe_all_histories.append(alignment_history)

            # Building next state
            attention = array_ops.concat(all_attentions, 1)
            next_state = AttentionWrapperState(time=state.time + 1,
                                               cell_state=next_cell_state,
                                               attention=attention,
                                               attention_state=self._item_or_tuple(all_attention_states),
                                               alignments=self._item_or_tuple(all_alignments),
                                               alignment_history=self._item_or_tuple(maybe_all_histories))

            # Returning
            if self._output_attention:
                return attention, next_state
            return cell_output, next_state

class StaticAttentionWrapper(rnn_cell_impl.RNNCell):
    """ Wraps another `RNNCell` with attention. """

    def __init__(self, cell, memory, alignments, sequence_length, probability_fn=None, score_mask_value=None,
                 attention_layer_size=None, cell_input_fn=None, output_attention=False, name=None):
        """ Constructs an AttentionWrapper with static alignments (attention weights)

            :param cell: An instance of `RNNCell`.
            :param memory: The memory to query [batch_size, memory_time, memory_size]
            :param alignments: A tensor of probabilities of shape [batch_size, time_steps, memory_time]
            :param sequence_length: Sequence lengths for the batch entries in memory. Size (b,)
            :param probability_fn: A `callable`.  Converts the score to probabilities.  The default is @{tf.nn.softmax}.
            :param score_mask_value:  The mask value for score before passing into `probability_fn`. Default is -inf.
            :param attention_layer_size: The size of the attention layer. Uses the context if None.
            :param cell_input_fn: (optional) A `callable` to aggregate attention.
                                  Default: `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
            :param output_attention: If true, outputs the attention, if False outputs the cell output.
            :param name: name: Name to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(StaticAttentionWrapper, self).__init__(name=name)
        rnn_cell_impl.assert_like_rnncell('cell', cell)

        # Setting values
        self._cell = cell
        self._memory = memory
        self._attention_layer_size = attention_layer_size
        self._output_attention = output_attention
        self._memory_time = alignments.get_shape()[-1].value
        self._memory_size = memory.get_shape()[-1].value
        self._sequence_length = sequence_length

        # Validating attention layer size
        if self._attention_layer_size is None:
            self._attention_layer_size = self._memory_size

        # Validating cell_input_fn
        if cell_input_fn is None:
            cell_input_fn = lambda inputs, attention: array_ops.concat([inputs, attention], -1)
        else:
            if not callable(cell_input_fn):
                raise TypeError('cell_input_fn must be callable, saw type: %s' % type(cell_input_fn).__name__)
        self._cell_input_fn = cell_input_fn

        # Probability Function
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if score_mask_value is None:
            score_mask_value = dtypes.as_dtype(self._memory.dtype).as_numpy_dtype(-np.inf)
        self._probability_fn = lambda score, _: probability_fn(_maybe_mask_score(score,
                                                                                 sequence_length,
                                                                                 score_mask_value), _)

        # Storing alignments as TA
        # Padding with 1 additional zero, to prevent error on read(0)
        alignments = array_ops.pad(alignments, [(0, 0), (0, 1), (0, 0)])
        alignments = nest.map_structure(_transpose_batch_time, alignments)      # (max_time + 1, b, memory_time)
        self._alignments_ta = nest.map_structure(_unstack_ta, alignments)       # [time_step + 1, batch, memory_time]
        self._initial_alignment = self._alignments_ta.read(0)
        self._initial_attention = self._compute_attention(self._initial_alignment, self._memory)[0]

        # Storing zero inputs
        batch_size = array_ops.shape(memory)[0]
        self._zero_cell_output = array_ops.zeros([batch_size, cell.output_size])
        self._zero_attention = array_ops.zeros([batch_size, self._attention_layer_size])
        self._zero_state = self.zero_state(batch_size, dtypes.float32)
        self._zero_alignment = array_ops.zeros_like(self._initial_alignment)

    def _compute_attention(self, alignments, memory):
        """Computes the attention and alignments for a given attention_mechanism."""
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, 1)

        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is  [batch_size, 1, memory_time]
        # memory is [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.
        context = math_ops.matmul(expanded_alignments, memory)
        context = array_ops.squeeze(context, [1])
        attn_layer = lambda x: x
        if self._attention_layer_size != self._memory_size:
            attn_layer = core.Dense(self._attention_layer_size, name='attn_layer', use_bias=False, dtype=context.dtype)
        attention = attn_layer(context)
        return attention, alignments

    @property
    def output_size(self):
        """ Returns the WrappedCell output size """
        if self._output_attention:
            return self._attention_layer_size
        return self._cell.output_size

    @property
    def state_size(self):
        """ The `state_size` property of `AttentionWrapper`.
            :return: An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(cell_state=self._cell.state_size,
                                     time=tensor_shape.TensorShape([]),
                                     attention=self._attention_layer_size,
                                     alignments=self._memory_time,
                                     attention_state=self._memory_time,
                                     alignment_history=())

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `AttentionWrapper`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: An `AttentionWrapperState` tuple containing zeroed out tensors and, possibly, empty TA objects.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return AttentionWrapperState(cell_state=self._cell.zero_state(batch_size, dtype),
                                         time=array_ops.zeros([], dtype=dtypes.int32),
                                         attention=self._initial_attention,
                                         alignments=self._initial_alignment,
                                         attention_state=self._initial_alignment,
                                         alignment_history=())

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):                                                          # pylint: disable=arguments-differ
        """ Perform a step of attention-wrapped RNN
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: An instance of `AttentionWrapperState` containing tensors from the previous time step.
            :return: A tuple `(attention_or_cell_output, next_state)`, where:
                    - `attention_or_cell_output` depending on `output_attention`.
                    - `next_state` is an instance of `AttentionWrapperState` containing the state calculated at
                       this time step.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError('Expected state to be instance of AttentionWrapperState. Received type %s instead.'
                            % type(state))

        next_time = state.time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)

        def get_next_alignments():
            """ Returns the next alignments """
            next_align = self._alignments_ta.read(next_time)
            with ops.control_dependencies([next_align]):
                return array_ops.identity(next_align)

        # Calculate the true inputs to the cell based on the previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, cell_state = self._cell(cell_inputs, cell_state)

        # Computing context
        next_alignments = control_flow_ops.cond(all_finished,
                                                true_fn=lambda: self._zero_alignment,
                                                false_fn=get_next_alignments)
        attention, _ = self._compute_attention(next_alignments, self._memory)

        next_state = AttentionWrapperState(time=next_time,
                                           cell_state=cell_state,
                                           attention=attention,
                                           alignments=next_alignments,
                                           attention_state=next_alignments,
                                           alignment_history=())

        if self._output_attention:
            return attention, next_state
        return cell_output, next_state


class SelfAttentionWrapperState(
        collections.namedtuple('SelfAttentionWrapperState', ('cell_state',      # The underlying cell state
                                                             'time',            # The current time
                                                             'memory'))):       # The current memory [b, t, mem_size]
    """ `namedtuple` storing the state of a `SelfAttentionWrapper`. """

    def clone(self, **kwargs):
        """ Clone this object, overriding components provided by kwargs. """
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return contrib_framework.with_same_shape(old, new)
            return new

        return nest.map_structure(with_same_shape,
                                  self,
                                  super(SelfAttentionWrapperState, self)._replace(**kwargs))

class SelfAttentionWrapper(rnn_cell_impl.RNNCell):
    """ Wraps another `RNNCell` with attention over the previous outputs. """

    def __init__(self, cell, memory_size, num_units, sequence_length, normalize=False, probability_fn=None,
                 score_mask_value=None, attention_layer_size=None, cell_input_fn=None, input_fn=None,
                 output_attention=False, dtype=None, name=None):
        """ Wrapper that computes attention over previous outputs (i.e. inputs from time 1 to time t)
            :param cell: An instance of `RNNCell`.
            :param memory_size: The size of the memory (inputs to the cell).
            :param num_units: The depth of the query mechanism.
            :param sequence_length: Sequence lengths for the batch entries in memory. Size (b,)
            :param normalize: Python boolean.  Whether to normalize the energy term.
            :param probability_fn: A `callable`.  Converts the score to probabilities.  The default is @{tf.nn.softmax}.
            :param score_mask_value:  The mask value for score before passing into `probability_fn`. Default is -inf.
            :param attention_layer_size: The size of the attention layer. Uses the context if None.
            :param cell_input_fn: A `callable` to aggregate attention. Default: concat([input, attn], axis=-1)
            :param input_fn: A `callable` to apply to the cell inputs before adding to the memory. Default: identity.
            :param output_attention: If true, outputs the attention, if False outputs the cell output.
            :param dtype: The data type for the query and memory layers of the attention mechanism.
            :param name: name: Name to use when creating ops.
        """
        # pylint: disable=too-many-arguments
        # Initializing RNN Cell
        super(SelfAttentionWrapper, self).__init__(name=name)
        rnn_cell_impl.assert_like_rnncell('cell', cell)

        # Setting values
        self._cell = cell
        self._memory_size = memory_size
        self._num_units = num_units
        self._sequence_length = sequence_length
        self._normalize = normalize
        self._attention_layer_size = attention_layer_size
        self._output_attention = output_attention
        self._dtype = dtype if dtype is not None else dtypes.float32
        self._name = name

        # Cell input function
        if cell_input_fn is None:
            cell_input_fn = lambda inputs, attention: array_ops.concat([inputs, attention], -1)
        elif not callable(cell_input_fn):
            raise TypeError('cell_input_fn must be callable, saw type: %s' % type(cell_input_fn).__name__)
        self._cell_input_fn = cell_input_fn

        # Input function
        if input_fn is None:
            input_fn = lambda inputs: inputs
        elif not callable(input_fn):
            raise TypeError('input_fn must be callable, saw type: %s' % type(input_fn).__name__)
        self._input_fn = input_fn

        # Bahdanau Attention
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        self._wrapped_probability_fn = lambda score, _: probability_fn(score)
        if score_mask_value is None:
            score_mask_value = dtypes.as_dtype(self._dtype).as_numpy_dtype(-np.inf)
        self._score_mask_value = score_mask_value

    def _compute_attention(self, query, memory):
        """ Computes the attention and alignments for the Bahdanau attention mechanism .
            :param query: The query (inputs) to use to compute attention. Size [b, input_size]
            :param memory: The memory (previous outputs) used to compute attention [b, time_step, memory_size]
            :return: The attention. Size [b, attn_size]
        """
        assert len(memory.shape) == 3, 'Memory needs to be [batch, time, memory_size]'
        memory_time = array_ops.shape(memory)[1]
        memory_size = memory.shape[2]
        num_units = self._num_units
        assert self._memory_size == memory_size, 'Expected mem size of %s - Got %s' % (self._memory_size, memory_size)

        # Query, memory, and attention layers
        query_layer = core.Dense(num_units, name='query_layer', use_bias=False, dtype=self._dtype)
        memory_layer = lambda x: x
        if memory_size != self._num_units:
            memory_layer = core.Dense(num_units, name='memory_layer', use_bias=False, dtype=self._dtype)
        attn_layer = lambda x: x
        if self._attention_layer_size is not None and memory_size != self._attention_layer_size:
            attn_layer = core.Dense(self._attention_layer_size, name='attn_layer', use_bias=False, dtype=self._dtype)

        # Masking memory
        sequence_length = gen_math_ops.minimum(memory_time, self._sequence_length)
        sequence_mask = array_ops.sequence_mask(sequence_length, maxlen=memory_time, dtype=dtypes.float32)[..., None]
        values = memory * sequence_mask
        keys = memory_layer(values)

        # Computing scores
        processed_query = query_layer(query)
        scores = _bahdanau_score(processed_query, keys, self._normalize)

        # Getting alignments
        masked_scores = _maybe_mask_score(scores, sequence_length, self._score_mask_value)
        alignments = self._wrapped_probability_fn(masked_scores, None)                  # [batch, time]

        # Getting attention
        expanded_alignments = array_ops.expand_dims(alignments, 1)                      # [batch, 1, time]
        context = math_ops.matmul(expanded_alignments, memory)                          # [batch, 1, memory_size]
        context = array_ops.squeeze(context, [1])                                       # [batch, memory_size]
        attention = attn_layer(context)                                                 # [batch, attn_size]

        # Returning attention
        return attention

    @property
    def output_size(self):
        """ Returns the WrappedCell output size """
        if self._output_attention:
            return self._attention_layer_size
        return self._cell.output_size

    @property
    def state_size(self):
        """ The `state_size` property of `AttentionWrapper`.
            :return: An `SelfAttentionWrapperState` tuple containing shapes used by this object.
        """
        return SelfAttentionWrapperState(cell_state=self._cell.state_size,
                                         time=tensor_shape.TensorShape([]),
                                         memory=self._memory_size)

    def zero_state(self, batch_size, dtype):
        """ Return an initial (zero) state tuple for this `AttentionWrapper`.
            :param batch_size: `0D` integer tensor: the batch size.
            :param dtype: The internal state data type.
            :return: An `SelfAttentionWrapperState` tuple containing zeroed out tensors.
        """
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            # Using batch_size * 0, rather than just 0 to have a dynamic dimension
            initial_cell_state = self._cell.zero_state(batch_size, dtype)
            initial_memory = array_ops.zeros([batch_size, batch_size * 0, self._memory_size], dtype=self._dtype)
            return SelfAttentionWrapperState(cell_state=initial_cell_state,
                                             time=array_ops.zeros([], dtype=dtypes.int32),
                                             memory=initial_memory)

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the given layer """
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.output_size)

    def call(self, inputs, state):                                                          # pylint: disable=arguments-differ
        """ Perform a step of attention-wrapped RNN
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: An instance of `SelfAttentionWrapperState` containing tensors from the previous time step.
            :return: A tuple `(attention_or_cell_output, next_state)`, where:
                    - `attention_or_cell_output` depending on `output_attention`.
                    - `next_state` is an instance of `SelfAttentionWrapperState` containing the state calculated at
                       this time step.
        """
        if not isinstance(state, SelfAttentionWrapperState):
            raise TypeError('Expected state to be instance of AttentionWrapperState. Received type %s instead.'
                            % type(state))

        # Getting batch size
        batch_size = array_ops.shape(inputs)[0]
        assert len(inputs.shape) == 2, 'Expected inputs to be of rank 2'

        def get_next_memory_and_attn():
            """ Gets the next memory and attention """
            next_memory = array_ops.concat([state.memory,                                       # [b, t, mem_size]
                                            array_ops.expand_dims(self._input_fn(inputs), axis=1)], axis=1)
            next_attention = self._compute_attention(inputs, next_memory)
            with ops.control_dependencies([next_memory, next_attention]):
                return array_ops.identity(next_memory), array_ops.identity(next_attention)

        def get_zero_memory_and_attn():
            """ Time = 0, we don't concatenate to memory and attention is all 0. """
            next_memory = state.memory
            next_attention = array_ops.zeros([batch_size, self._attention_layer_size], dtype=inputs.dtype)
            with ops.control_dependencies([next_memory, next_attention]):
                return array_ops.identity(next_memory), array_ops.identity(next_attention)

        # Computing memory and attention
        memory, attention = control_flow_ops.cond(gen_math_ops.equal(state.time, 0),
                                                  true_fn=get_zero_memory_and_attn,
                                                  false_fn=get_next_memory_and_attn)

        # Calculate the true inputs to the cell based on the previous attention value.
        cell_inputs = self._cell_input_fn(inputs, attention)
        cell_state = state.cell_state
        cell_output, cell_state = self._cell(cell_inputs, cell_state)

        # Extracting computed context
        next_state = SelfAttentionWrapperState(cell_state=cell_state,
                                               time=state.time + 1,
                                               memory=memory)

        # Returning cell output or attention
        if self._output_attention:
            return attention, next_state
        return cell_output, next_state
