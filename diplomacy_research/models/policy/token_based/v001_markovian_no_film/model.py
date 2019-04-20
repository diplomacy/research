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
""" Policy model (v001_markovian_no_film)
    - Contains the policy model (v001_markovian_no_film), to evaluate the best actions given a state
"""
import logging
from diplomacy_research.models.policy.token_based import TokenBasedPolicyModel, load_args as load_parent_args
from diplomacy_research.models.state_space import get_adjacency_matrix, VOCABULARY_SIZE, TOKENS_PER_ORDER, \
    NB_SUPPLY_CENTERS

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_parent_args() + [
        # Hyperparameters
        ('int', 'nb_graph_conv', 8, 'Number of Graph Conv Layer'),
        ('int', 'word_emb_size', 400, 'Word embedding size.'),
        ('int', 'gcn_size', 80, 'Size of graph convolution outputs.'),
        ('int', 'lstm_size', 400, 'LSTM (Encoder and Decoder) size.'),
        ('int', 'attn_size', 80, 'LSTM decoder attention size.'),
    ]

class PolicyModel(TokenBasedPolicyModel):
    """ Policy Model """

    def _encode_board(self, board_state, name, reuse=None):
        """ Encodes a board state or prev orders state
            :param board_state: The board state / prev orders state to encode - (batch, NB_NODES, initial_features)
            :param name: The name to use for the encoding
            :param reuse: Whether to reuse or not the weights from another encoding operation
            :return: The encoded board state / prev_orders state
        """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.graph_convolution import GraphConvolution, preprocess_adjacency
        from diplomacy_research.utils.tensorflow import batch_norm

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]
        relu = tf.nn.relu

        # Computing norm adjacency
        norm_adjacency = preprocess_adjacency(get_adjacency_matrix())
        norm_adjacency = tf.tile(tf.expand_dims(norm_adjacency, axis=0), [tf.shape(board_state)[0], 1, 1])

        # Building scope
        scope = tf.VariableScope(name='policy/%s' % name, reuse=reuse)
        with tf.variable_scope(scope):

            # Adding noise to break symmetry
            board_state = board_state + tf.random_normal(tf.shape(board_state), stddev=0.01)
            graph_conv = board_state

            # First Layer
            graph_conv = GraphConvolution(input_dim=graph_conv.shape[-1].value,             # (b, NB_NODES, gcn_size)
                                          output_dim=hps('gcn_size'),
                                          norm_adjacency=norm_adjacency,
                                          activation_fn=relu,
                                          bias=True)(graph_conv)
            graph_conv = batch_norm(graph_conv, is_training=pholder('is_training'), fused=True)

            # Intermediate Layers
            for _ in range(1, hps('nb_graph_conv') - 1):
                graph_conv = GraphConvolution(input_dim=hps('gcn_size'),                    # (b, NB_NODES, gcn_size)
                                              output_dim=hps('gcn_size'),
                                              norm_adjacency=norm_adjacency,
                                              activation_fn=relu,
                                              bias=True)(graph_conv)
                graph_conv = batch_norm(graph_conv, is_training=pholder('is_training'), fused=True)

            # Final Layer
            graph_conv = GraphConvolution(input_dim=hps('gcn_size'),                        # (b, NB_NODES, attn_size)
                                          output_dim=hps('attn_size'),
                                          norm_adjacency=norm_adjacency,
                                          activation_fn=None,
                                          bias=True)(graph_conv)

        # Returning
        return graph_conv

    def _get_board_state_conv(self, board_0yr_conv, is_training, prev_ord_conv=None):
        """ Computes the board state conv to use as the attention target (memory)

            :param board_0yr_conv: The board state encoding of the current (present) board state)
            :param is_training: Indicate whether we are doing training or inference
            :param prev_ord_conv: Optional. The encoding of the previous orders state.
            :return: The board state conv to use as the attention target (memory)
        """
        return board_0yr_conv

    def _build_policy_initial(self):
        """ Builds the policy model (initial step) """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.initializers import uniform
        from diplomacy_research.utils.tensorflow import build_sparse_batched_tensor, pad_axis, to_float, to_bool

        if not self.placeholders:
            self.placeholders = self.get_placeholders()

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]

        # Training loop
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):

                # Features
                board_state = to_float(self.features['board_state'])        # tf.flt32 - (b, NB_NODES, NB_FEATURES)
                decoder_inputs = self.features['decoder_inputs']            # tf.int32 - (b, <= 1 + TOK/ORD * NB_SCS)
                decoder_lengths = self.features['decoder_lengths']          # tf.int32 - (b,)
                dropout_rates = self.features['dropout_rate']               # tf.flt32 - (b,)

                # Batch size
                batch_size = tf.shape(board_state)[0]

                # Building decoder mask
                decoder_mask_indices = self.features['decoder_mask_indices']    # tf.int64 - (b, 3 * len)
                decoder_mask_shape = self.proto_fields['decoder_mask'].shape

                # Overriding dropout_rates if pholder('dropout_rate') > 0
                dropout_rates = tf.cond(tf.greater(pholder('dropout_rate'), 0.),
                                        true_fn=lambda: tf.zeros_like(dropout_rates) + pholder('dropout_rate'),
                                        false_fn=lambda: dropout_rates)

                # Padding inputs
                decoder_inputs = pad_axis(decoder_inputs, axis=-1, min_size=2)
                decoder_mask_indices = pad_axis(decoder_mask_indices, axis=-1, min_size=len(decoder_mask_shape))

                # Reshaping to (b, len, 3)
                # decoder_mask is -- tf.bool (batch, TOK/ORD * NB_SC, VOCAB_SIZE, VOCAB_SIZE)
                decoder_mask_indices = tf.reshape(decoder_mask_indices, [batch_size, -1, len(decoder_mask_shape)])
                decoder_mask = build_sparse_batched_tensor(decoder_mask_indices,
                                                           value=True,
                                                           dtype=tf.bool,
                                                           dense_shape=decoder_mask_shape)

                # Making sure all RNN lengths are at least 1
                # No need to trim, because the fields are variable length
                raw_decoder_lengths = decoder_lengths
                decoder_lengths = tf.math.maximum(1, decoder_lengths)

                # Placeholders
                decoder_type = tf.reduce_max(pholder('decoder_type'))
                is_training = pholder('is_training')

                # Creating graph convolution
                with tf.variable_scope('graph_conv_scope'):
                    assert hps('nb_graph_conv') >= 2

                    # Encoding board state
                    board_state_0yr_conv = self.encode_board(board_state, name='board_state_conv')
                    board_state_conv = self.get_board_state_conv(board_state_0yr_conv, is_training)

                # Creating word embedding vector (to embed word_ix)
                # Embeddings needs to be cached locally on the worker, otherwise TF can't compute their gradients
                with tf.variable_scope('word_embedding_scope'):
                    # embedding:    (voc_size, 256)
                    caching_device = self.cluster_config.caching_device if self.cluster_config else None
                    word_embedding = uniform(name='word_embedding',
                                             shape=[VOCABULARY_SIZE, hps('word_emb_size')],
                                             scale=1.,
                                             caching_device=caching_device)

        # Building output tags
        outputs = {'batch_size': batch_size,
                   'decoder_inputs': decoder_inputs,
                   'decoder_mask': decoder_mask,
                   'decoder_type': decoder_type,
                   'raw_decoder_lengths': raw_decoder_lengths,
                   'decoder_lengths': decoder_lengths,
                   'board_state_conv': board_state_conv,
                   'board_state_0yr_conv': board_state_0yr_conv,
                   'word_embedding': word_embedding,
                   'in_retreat_phase': tf.math.logical_and(         # 1) board not empty, 2) disl. units present
                       tf.reduce_sum(board_state[:], axis=[1, 2]) > 0,
                       tf.math.logical_not(to_bool(tf.reduce_min(board_state[:, :, 23], -1))))}

        # Adding to graph
        self.add_meta_information(outputs)

    def _build_policy_final(self):
        """ Builds the policy model (final step) """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.attention import AttentionWrapper, BahdanauAttention
        from diplomacy_research.models.layers.beam_decoder import DiverseBeamSearchDecoder
        from diplomacy_research.models.layers.decoder import MaskedBasicDecoder
        from diplomacy_research.models.layers.dropout import SeededDropoutWrapper
        from diplomacy_research.models.layers.dynamic_decode import dynamic_decode
        from diplomacy_research.models.policy.token_based.helper import CustomHelper, CustomBeamHelper
        from diplomacy_research.utils.tensorflow import cross_entropy, sequence_loss, to_int32, to_float, get_tile_beam

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]

        # Training loop
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):

                # Features
                player_seeds = self.features['player_seed']                 # tf.int32 - (b,)
                temperature = self.features['temperature']                  # tf,flt32 - (b,)
                dropout_rates = self.features['dropout_rate']               # tf.flt32 - (b,)

                # Placeholders
                stop_gradient_all = pholder('stop_gradient_all')

                # Outputs (from initial steps)
                batch_size = self.outputs['batch_size']
                decoder_inputs = self.outputs['decoder_inputs']
                decoder_mask = self.outputs['decoder_mask']
                decoder_type = self.outputs['decoder_type']
                raw_decoder_lengths = self.outputs['raw_decoder_lengths']
                decoder_lengths = self.outputs['decoder_lengths']
                board_state_conv = self.outputs['board_state_conv']
                word_embedding = self.outputs['word_embedding']

                # --- Decoding ---
                with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
                    lstm_cell = tf.contrib.rnn.LSTMBlockCell(hps('lstm_size'))

                    # decoder output to token
                    decoder_output_layer = tf.layers.Dense(units=VOCABULARY_SIZE,
                                                           activation=None,
                                                           kernel_initializer=tf.random_normal_initializer,
                                                           use_bias=True)

                    # ======== Regular Decoding ========
                    # Applying dropout to input + attention and to output layer
                    decoder_cell = SeededDropoutWrapper(cell=lstm_cell,
                                                        seeds=player_seeds,
                                                        input_keep_probs=1. - dropout_rates,
                                                        output_keep_probs=1. - dropout_rates,
                                                        variational_recurrent=hps('use_v_dropout'),
                                                        input_size=hps('word_emb_size') + hps('attn_size'),
                                                        dtype=tf.float32)

                    # apply attention over location
                    # curr_state [batch, NB_NODES, attn_size]
                    attention_scope = tf.VariableScope(name='policy/decoder_scope/Attention', reuse=tf.AUTO_REUSE)
                    attention_mechanism = BahdanauAttention(num_units=hps('attn_size'),
                                                            memory=board_state_conv,
                                                            normalize=True,
                                                            name_or_scope=attention_scope)
                    decoder_cell = AttentionWrapper(cell=decoder_cell,
                                                    attention_mechanism=attention_mechanism,
                                                    output_attention=False,
                                                    name_or_scope=attention_scope)

                    # Setting initial state
                    decoder_init_state = decoder_cell.zero_state(batch_size, tf.float32)
                    decoder_init_state = decoder_init_state.clone(attention=tf.reduce_mean(board_state_conv, axis=1))

                    # ---- Helper ----
                    helper = CustomHelper(decoder_type=decoder_type,
                                          inputs=decoder_inputs[:, :-1],
                                          embedding=word_embedding,
                                          sequence_length=decoder_lengths,
                                          mask=decoder_mask,
                                          time_major=False,
                                          softmax_temperature=temperature)

                    # ---- Decoder ----
                    sequence_mask = tf.sequence_mask(raw_decoder_lengths,
                                                     maxlen=tf.reduce_max(decoder_lengths),
                                                     dtype=tf.float32)
                    maximum_iterations = TOKENS_PER_ORDER * NB_SUPPLY_CENTERS
                    model_decoder = MaskedBasicDecoder(cell=decoder_cell,
                                                       helper=helper,
                                                       initial_state=decoder_init_state,
                                                       output_layer=decoder_output_layer,
                                                       extract_state=True)
                    training_results, _, _ = dynamic_decode(decoder=model_decoder,
                                                            output_time_major=False,
                                                            maximum_iterations=maximum_iterations,
                                                            swap_memory=hps('swap_memory'))
                    global_vars_after_decoder = set(tf.global_variables())

                    # ======== Beam Search Decoding ========
                    tile_beam = get_tile_beam(hps('beam_width'))

                    # Applying dropout to input + attention and to output layer
                    decoder_cell = SeededDropoutWrapper(cell=lstm_cell,
                                                        seeds=tile_beam(player_seeds),
                                                        input_keep_probs=tile_beam(1. - dropout_rates),
                                                        output_keep_probs=tile_beam(1. - dropout_rates),
                                                        variational_recurrent=hps('use_v_dropout'),
                                                        input_size=hps('word_emb_size') + hps('attn_size'),
                                                        dtype=tf.float32)

                    # apply attention over location
                    # curr_state [batch, NB_NODES, attn_size]
                    attention_mechanism = BahdanauAttention(num_units=hps('attn_size'),
                                                            memory=tile_beam(board_state_conv),
                                                            normalize=True,
                                                            name_or_scope=attention_scope)
                    decoder_cell = AttentionWrapper(cell=decoder_cell,
                                                    attention_mechanism=attention_mechanism,
                                                    output_attention=False,
                                                    name_or_scope=attention_scope)

                    # Setting initial state
                    decoder_init_state = decoder_cell.zero_state(batch_size * hps('beam_width'), tf.float32)
                    decoder_init_state = decoder_init_state.clone(attention=tf.reduce_mean(tile_beam(board_state_conv),
                                                                                           axis=1))

                    # ---- Beam Helper and Decoder ----
                    beam_helper = CustomBeamHelper(cell=decoder_cell,
                                                   embedding=word_embedding,
                                                   mask=decoder_mask,
                                                   sequence_length=decoder_lengths,
                                                   output_layer=decoder_output_layer,
                                                   initial_state=decoder_init_state,
                                                   beam_width=hps('beam_width'))
                    beam_decoder = DiverseBeamSearchDecoder(beam_helper=beam_helper,
                                                            sequence_length=decoder_lengths,
                                                            nb_groups=hps('beam_groups'))
                    beam_results, beam_state, _ = dynamic_decode(decoder=beam_decoder,
                                                                 output_time_major=False,
                                                                 maximum_iterations=maximum_iterations,
                                                                 swap_memory=hps('swap_memory'))

                    # Making sure we haven't created new global variables
                    assert not set(tf.global_variables()) - global_vars_after_decoder, 'New global vars were created'

                    # Processing results
                    logits = training_results.rnn_output                            # (b, dec_len, VOCAB_SIZE)
                    logits_length = tf.shape(logits)[1]                             # dec_len
                    decoder_target = decoder_inputs[:, 1:1 + logits_length]

                    # Selected tokens are the token that was actually fed at the next position
                    sample_mask = to_float(tf.math.equal(training_results.sample_id, -1))
                    selected_tokens = to_int32(
                        sequence_mask * (sample_mask * to_float(decoder_target)
                                         + (1. - sample_mask) * to_float(training_results.sample_id)))

                    # Argmax tokens are the most likely token outputted at each position
                    argmax_tokens = to_int32(to_float(tf.argmax(logits, axis=-1)) * sequence_mask)
                    log_probs = -1. * cross_entropy(logits=logits, labels=selected_tokens) * sequence_mask

                # Computing policy loss
                with tf.variable_scope('policy_loss'):
                    policy_loss = sequence_loss(logits=logits,
                                                targets=decoder_target,
                                                weights=sequence_mask,
                                                average_across_batch=True,
                                                average_across_timesteps=True)
                    policy_loss = tf.cond(stop_gradient_all,
                                          lambda: tf.stop_gradient(policy_loss),                                        # pylint: disable=cell-var-from-loop
                                          lambda: policy_loss)                                                          # pylint: disable=cell-var-from-loop

        # Building output tags
        outputs = {'tag/policy/token_based/v001_markovian_no_film': True,
                   'targets': decoder_inputs[:, 1:],
                   'selected_tokens': selected_tokens,
                   'argmax_tokens': argmax_tokens,
                   'logits': logits,
                   'log_probs': log_probs,
                   'beam_tokens': tf.transpose(beam_results.predicted_ids, perm=[0, 2, 1]),     # [batch, beam, steps]
                   'beam_log_probs': beam_state.log_probs,
                   'rnn_states': training_results.rnn_state,
                   'policy_loss': policy_loss,
                   'draw_prob': self.outputs.get('draw_prob', tf.zeros_like(self.features['draw_target'])),
                   'learning_rate': self.learning_rate}

        # Adding features, placeholders and outputs to graph
        self.add_meta_information(outputs)
