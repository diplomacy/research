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
""" Policy model (v002_markovian_film)
    - Contains the policy model (v002_markovian_film), to evaluate the best actions given a state
"""
import logging
from diplomacy_research.models.policy.order_based import OrderBasedPolicyModel, load_args as load_parent_args
from diplomacy_research.models.state_space import get_adjacency_matrix, NB_SUPPLY_CENTERS, NB_POWERS, \
    ORDER_VOCABULARY_SIZE, MAX_CANDIDATES, PAD_ID
from diplomacy_research.settings import NB_PARTITIONS

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_parent_args() + [
        # Hyperparameters
        ('int', 'nb_graph_conv', 8, 'Number of Graph Conv Layer'),
        ('int', 'order_emb_size', 200, 'Order embedding size.'),
        ('int', 'power_emb_size', 80, 'Power embedding size.'),
        ('int', 'gcn_size', 80, 'Size of graph convolution outputs.'),
        ('int', 'lstm_size', 200, 'LSTM (Encoder and Decoder) size.'),
        ('int', 'attn_size', 80, 'LSTM decoder attention size.'),
    ]

class PolicyModel(OrderBasedPolicyModel):
    """ Policy Model """

    def _encode_board(self, board_state, name, reuse=None):
        """ Encodes a board state or prev orders state
            :param board_state: The board state / prev orders state to encode - (batch, NB_NODES, initial_features)
            :param name: The name to use for the encoding
            :param reuse: Whether to reuse or not the weights from another encoding operation
            :return: The encoded board state / prev_orders state
        """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.graph_convolution import film_gcn_res_block, preprocess_adjacency

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]
        relu = tf.nn.relu

        # Getting film gammas and betas
        film_gammas = self.outputs['_%s_film_gammas' % name]
        film_betas = self.outputs['_%s_film_betas' % name]

        # Computing norm adjacency
        norm_adjacency = preprocess_adjacency(get_adjacency_matrix())
        norm_adjacency = tf.tile(tf.expand_dims(norm_adjacency, axis=0), [tf.shape(board_state)[0], 1, 1])

        # Building scope
        scope = tf.VariableScope(name='policy/%s' % name, reuse=reuse)
        with tf.variable_scope(scope):

            # Adding noise to break symmetry
            board_state = board_state + tf.random_normal(tf.shape(board_state), stddev=0.01)
            graph_conv = tf.layers.Dense(units=hps('gcn_size'), activation=relu)(board_state)

            # First and intermediate layers
            for layer_idx in range(hps('nb_graph_conv') - 1):
                graph_conv = film_gcn_res_block(inputs=graph_conv,                          # (b, NB_NODES, gcn_size)
                                                gamma=film_gammas[layer_idx],
                                                beta=film_betas[layer_idx],
                                                gcn_out_dim=hps('gcn_size'),
                                                norm_adjacency=norm_adjacency,
                                                is_training=pholder('is_training'),
                                                residual=True)

            # Last layer
            graph_conv = film_gcn_res_block(inputs=graph_conv,                              # (b, NB_NODES, final_size)
                                            gamma=film_gammas[-1],
                                            beta=film_betas[-1],
                                            gcn_out_dim=hps('attn_size'),
                                            norm_adjacency=norm_adjacency,
                                            is_training=pholder('is_training'),
                                            residual=False)

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
        from diplomacy_research.utils.tensorflow import pad_axis, to_int32, to_float, to_bool

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
                decoder_inputs = self.features['decoder_inputs']            # tf.int32 - (b, <= 1 + NB_SCS)
                decoder_lengths = self.features['decoder_lengths']          # tf.int32 - (b,)
                candidates = self.features['candidates']                    # tf.int32 - (b, nb_locs * MAX_CANDIDATES)
                current_power = self.features['current_power']              # tf.int32 - (b,)
                dropout_rates = self.features['dropout_rate']               # tf.flt32 - (b,)

                # Batch size
                batch_size = tf.shape(board_state)[0]

                # Overriding dropout_rates if pholder('dropout_rate') > 0
                dropout_rates = tf.cond(tf.greater(pholder('dropout_rate'), 0.),
                                        true_fn=lambda: tf.zeros_like(dropout_rates) + pholder('dropout_rate'),
                                        false_fn=lambda: dropout_rates)

                # Padding decoder_inputs and candidates
                decoder_inputs = pad_axis(decoder_inputs, axis=-1, min_size=2)
                candidates = pad_axis(candidates, axis=-1, min_size=MAX_CANDIDATES)

                # Making sure all RNN lengths are at least 1
                # No need to trim, because the fields are variable length
                raw_decoder_lengths = decoder_lengths
                decoder_lengths = tf.math.maximum(1, decoder_lengths)

                # Placeholders
                decoder_type = tf.reduce_max(pholder('decoder_type'))
                is_training = pholder('is_training')

                # Reshaping candidates
                candidates = tf.reshape(candidates, [batch_size, -1, MAX_CANDIDATES])
                candidates = candidates[:, :tf.reduce_max(decoder_lengths), :]      # tf.int32 - (b, nb_locs, MAX_CAN)

                # Computing FiLM Gammas and Betas
                with tf.variable_scope('film_scope'):
                    power_embedding = uniform(name='power_embedding',
                                              shape=[NB_POWERS, hps('power_emb_size')],
                                              scale=1.)
                    current_power_mask = tf.one_hot(current_power, NB_POWERS, dtype=tf.float32)
                    current_power_embedding = tf.reduce_sum(power_embedding[None]
                                                            * current_power_mask[:, :, None], axis=1)  # (b, power_emb)

                    film_output_dims = [hps('gcn_size')] * (hps('nb_graph_conv') - 1) + [hps('attn_size')]
                    film_weights = tf.layers.Dense(units=2 * sum(film_output_dims),         # (b, 1, 750)
                                                   use_bias=True,
                                                   activation=None)(current_power_embedding)[:, None, :]
                    film_gammas, film_betas = tf.split(film_weights, 2, axis=2)             # (b, 1, 750)
                    film_gammas = tf.split(film_gammas, film_output_dims, axis=2)
                    film_betas = tf.split(film_betas, film_output_dims, axis=2)

                    # Storing as temporary output
                    self.add_output('_board_state_conv_film_gammas', film_gammas)
                    self.add_output('_board_state_conv_film_betas', film_betas)

                # Creating graph convolution
                with tf.variable_scope('graph_conv_scope'):
                    assert hps('nb_graph_conv') >= 2

                    # Encoding board state
                    board_state_0yr_conv = self.encode_board(board_state, name='board_state_conv')
                    board_state_conv = self.get_board_state_conv(board_state_0yr_conv, is_training)

                # Creating order embedding vector (to embed order_ix)
                # Embeddings needs to be cached locally on the worker, otherwise TF can't compute their gradients
                with tf.variable_scope('order_embedding_scope'):
                    # embedding:    (order_vocab_size, 64)
                    caching_device = self.cluster_config.caching_device if self.cluster_config else None
                    partitioner = tf.fixed_size_partitioner(NB_PARTITIONS) if hps('use_partitioner') else None
                    order_embedding = uniform(name='order_embedding',
                                              shape=[ORDER_VOCABULARY_SIZE, hps('order_emb_size')],
                                              scale=1.,
                                              partitioner=partitioner,
                                              caching_device=caching_device)

                # Creating candidate embedding
                with tf.variable_scope('candidate_embedding_scope'):
                    # embedding:    (order_vocab_size, 64)
                    caching_device = self.cluster_config.caching_device if self.cluster_config else None
                    partitioner = tf.fixed_size_partitioner(NB_PARTITIONS) if hps('use_partitioner') else None
                    candidate_embedding = uniform(name='candidate_embedding',
                                                  shape=[ORDER_VOCABULARY_SIZE, hps('lstm_size') + 1],
                                                  scale=1.,
                                                  partitioner=partitioner,
                                                  caching_device=caching_device)

                # Trimming to the maximum number of candidates
                candidate_lengths = tf.reduce_sum(to_int32(tf.math.greater(candidates, PAD_ID)), -1)    # int32 - (b,)
                max_candidate_length = tf.math.maximum(1, tf.reduce_max(candidate_lengths))
                candidates = candidates[:, :, :max_candidate_length]

        # Building output tags
        outputs = {'batch_size': batch_size,
                   'decoder_inputs': decoder_inputs,
                   'decoder_type': decoder_type,
                   'raw_decoder_lengths': raw_decoder_lengths,
                   'decoder_lengths': decoder_lengths,
                   'board_state_conv': board_state_conv,
                   'board_state_0yr_conv': board_state_0yr_conv,
                   'order_embedding': order_embedding,
                   'candidate_embedding': candidate_embedding,
                   'candidates': candidates,
                   'max_candidate_length': max_candidate_length,
                   'in_retreat_phase': tf.math.logical_and(             # 1) board not empty, 2) disl. units present
                       tf.reduce_sum(board_state[:], axis=[1, 2]) > 0,
                       tf.math.logical_not(to_bool(tf.reduce_min(board_state[:, :, 23], -1))))}

        # Adding to graph
        self.add_meta_information(outputs)

    def _build_policy_final(self):
        """ Builds the policy model (final step) """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.attention import AttentionWrapper, ModifiedBahdanauAttention
        from diplomacy_research.models.layers.beam_decoder import DiverseBeamSearchDecoder
        from diplomacy_research.models.layers.decoder import CandidateBasicDecoder
        from diplomacy_research.models.layers.dropout import SeededDropoutWrapper
        from diplomacy_research.models.layers.dynamic_decode import dynamic_decode
        from diplomacy_research.models.policy.order_based.helper import CustomHelper, CustomBeamHelper
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
                decoder_type = self.outputs['decoder_type']
                raw_decoder_lengths = self.outputs['raw_decoder_lengths']
                decoder_lengths = self.outputs['decoder_lengths']
                board_state_conv = self.outputs['board_state_conv']
                order_embedding = self.outputs['order_embedding']
                candidate_embedding = self.outputs['candidate_embedding']
                candidates = self.outputs['candidates']
                max_candidate_length = self.outputs['max_candidate_length']

                # --- Decoding ---
                with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
                    lstm_cell = tf.contrib.rnn.LSTMBlockCell(hps('lstm_size'))

                    # ======== Regular Decoding ========
                    # Applying Dropout to input, attention and output
                    decoder_cell = SeededDropoutWrapper(cell=lstm_cell,
                                                        seeds=player_seeds,
                                                        input_keep_probs=1. - dropout_rates,
                                                        output_keep_probs=1. - dropout_rates,
                                                        variational_recurrent=hps('use_v_dropout'),
                                                        input_size=hps('order_emb_size') + hps('attn_size'),
                                                        dtype=tf.float32)

                    # apply attention over location
                    # curr_state [batch, NB_NODES, attn_size]
                    attention_scope = tf.VariableScope(name='policy/decoder_scope/Attention', reuse=tf.AUTO_REUSE)
                    attention_mechanism = ModifiedBahdanauAttention(num_units=hps('attn_size'),
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
                                          order_embedding=order_embedding,
                                          candidate_embedding=candidate_embedding,
                                          sequence_length=decoder_lengths,
                                          candidates=candidates,
                                          time_major=False,
                                          softmax_temperature=temperature)

                    # ---- Decoder ----
                    sequence_mask = tf.sequence_mask(raw_decoder_lengths,
                                                     maxlen=tf.reduce_max(decoder_lengths),
                                                     dtype=tf.float32)
                    maximum_iterations = NB_SUPPLY_CENTERS
                    model_decoder = CandidateBasicDecoder(cell=decoder_cell,
                                                          helper=helper,
                                                          initial_state=decoder_init_state,
                                                          max_candidate_length=max_candidate_length,
                                                          extract_state=True)
                    training_results, _, _ = dynamic_decode(decoder=model_decoder,
                                                            output_time_major=False,
                                                            maximum_iterations=maximum_iterations,
                                                            swap_memory=hps('swap_memory'))
                    global_vars_after_decoder = set(tf.global_variables())

                    # ======== Beam Search Decoding ========
                    tile_beam = get_tile_beam(hps('beam_width'))

                    # Applying Dropout to input, attention and output
                    decoder_cell = SeededDropoutWrapper(cell=lstm_cell,
                                                        seeds=tile_beam(player_seeds),
                                                        input_keep_probs=tile_beam(1. - dropout_rates),
                                                        output_keep_probs=tile_beam(1. - dropout_rates),
                                                        variational_recurrent=hps('use_v_dropout'),
                                                        input_size=hps('order_emb_size') + hps('attn_size'),
                                                        dtype=tf.float32)

                    # apply attention over location
                    # curr_state [batch, NB_NODES, attn_size]
                    attention_mechanism = ModifiedBahdanauAttention(num_units=hps('attn_size'),
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
                                                   order_embedding=order_embedding,
                                                   candidate_embedding=candidate_embedding,
                                                   candidates=candidates,
                                                   sequence_length=decoder_lengths,
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
                    candidate_logits = training_results.rnn_output                  # (b, dec_len, max_cand_len)
                    logits_length = tf.shape(candidate_logits)[1]                   # dec_len
                    decoder_target = decoder_inputs[:, 1:1 + logits_length]

                    # Selected tokens are the token that was actually fed at the next position
                    sample_mask = to_float(tf.math.equal(training_results.sample_id, -1))
                    selected_tokens = to_int32(
                        sequence_mask * (sample_mask * to_float(decoder_target)
                                         + (1. - sample_mask) * to_float(training_results.sample_id)))

                    # Computing ArgMax tokens
                    argmax_id = to_int32(tf.argmax(candidate_logits, axis=-1))
                    max_nb_candidate = tf.shape(candidate_logits)[2]
                    candidate_ids = \
                        tf.reduce_sum(tf.one_hot(argmax_id, max_nb_candidate, dtype=tf.int32) * candidates, axis=-1)
                    argmax_tokens = to_int32(to_float(candidate_ids) * sequence_mask)

                    # Extracting the position of the target candidate
                    tokens_labels = tf.argmax(to_int32(tf.math.equal(selected_tokens[:, :, None], candidates)), -1)
                    target_labels = tf.argmax(to_int32(tf.math.equal(decoder_target[:, :, None], candidates)), -1)

                    # Log Probs
                    log_probs = -1. * cross_entropy(logits=candidate_logits, labels=tokens_labels) * sequence_mask

                # Computing policy loss
                with tf.variable_scope('policy_loss'):
                    policy_loss = sequence_loss(logits=candidate_logits,
                                                targets=target_labels,
                                                weights=sequence_mask,
                                                average_across_batch=True,
                                                average_across_timesteps=True)
                    policy_loss = tf.cond(stop_gradient_all,
                                          lambda: tf.stop_gradient(policy_loss),                                        # pylint: disable=cell-var-from-loop
                                          lambda: policy_loss)                                                          # pylint: disable=cell-var-from-loop

        # Building output tags
        outputs = {'tag/policy/order_based/v002_markovian_film': True,
                   'targets': decoder_inputs[:, 1:],
                   'selected_tokens': selected_tokens,
                   'argmax_tokens': argmax_tokens,
                   'logits': candidate_logits,
                   'log_probs': log_probs,
                   'beam_tokens': tf.transpose(beam_results.predicted_ids, perm=[0, 2, 1]),     # [batch, beam, steps]
                   'beam_log_probs': beam_state.log_probs,
                   'rnn_states': training_results.rnn_state,
                   'policy_loss': policy_loss,
                   'draw_prob': self.outputs.get('draw_prob', tf.zeros_like(self.features['draw_target'])),
                   'learning_rate': self.learning_rate}

        # Adding features, placeholders and outputs to graph
        self.add_meta_information(outputs)
