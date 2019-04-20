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
""" Policy model (Token Based)
    - Contains the parent policy model, to evaluate the best actions given a state
"""
from collections import OrderedDict
import logging
import math
from diplomacy_research.models.policy.base_policy_model import GREEDY_DECODER, TRAINING_DECODER, StatsKey, \
    OrderProbTokenLogProbs, BasePolicyModel, load_args as load_parent_args
from diplomacy_research.models.state_space import EOS_ID, TOKENS_PER_ORDER, POWER_VOCABULARY_IX_TO_KEY, \
    POWER_VOCABULARY_LIST, NB_SUPPLY_CENTERS, ix_to_token, STANDARD_TOPO_LOCS

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_parent_args()

class TokenBasedPolicyModel(BasePolicyModel):
    """ Policy Model """

    def __init__(self, dataset, hparams):
        """ Initialization
            :param dataset: The dataset that is used to iterate over the data.
            :param hparams: A dictionary of hyper parameters with their values
            :type dataset: diplomacy_research.models.datasets.supervised_dataset.SupervisedDataset
            :type dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
        """
        from diplomacy_research.utils.tensorflow import tf
        hps = lambda hparam_name: self.hparams[hparam_name]
        BasePolicyModel.__init__(self, dataset, hparams)

        # Learning rate
        if not hasattr(self, 'learning_rate') or self.learning_rate is None:
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):
                self.learning_rate = tf.Variable(float(hps('learning_rate')), trainable=False, dtype=tf.float32)

        # Optimizer
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = self.make_optimizer(self.learning_rate)

        # Build ops
        self.build_policy()

        # Decay ops
        if not hasattr(self, 'decay_learning_rate') or self.decay_learning_rate is None:
            self.decay_learning_rate = self.learning_rate.assign(self.placeholders['learning_rate'])

    @property
    def _nb_evaluation_loops(self):
        """ Contains the number of different evaluation tags we want to compute
            This also represent the number of loops we should do over the validation set
            Some model wants to calculate different statistics and require multiple pass to do that

            A value of 1 indicates to only run in the main validation loop
            A value > 1 indicates to run additional loops only for this model.
        """
        return 2

    @property
    def _evaluation_tags(self):
        """ List of evaluation tags (1 list of evaluation tag for each evaluation loop)
            e.g. [['Acc_1', 'Acc_5', 'Acc_Tokens'], ['Gr_1', 'Gr_5', 'Gr_Tokens']]
        """
        return [['[TF]X-Ent', '[TF]Perplexity', '[TF]Acc_1', '[TF]Acc_1_NoHold', '[TF]Acc_Tokens', '[TF]Acc_Player'],
                ['[Gr]Acc_1', '[Gr]Acc_1_NoHold', '[Gr]Acc_Tokens', '[Gr]Acc_Player']]

    @property
    def _early_stopping_tags(self):
        """ List of tags to use to detect early stopping
            The tags are a tuple of 1) 'min' or 'max' and 2) the tag's name
            e.g. [('max', '[Gr]Acc_1'), ('min', '[TF]Perplexity')]
        """
        return [('min', '[TF]Perplexity'), ('max', '[Gr]Acc_1')]

    @property
    def _placeholders(self):
        """ Return a dictionary of all placeholders needed by the model """
        from diplomacy_research.utils.tensorflow import tf, get_placeholder, get_placeholder_with_default

        # Note: 'decoder_type' needs to have a batch_dim to be compatible with TF Serving
        # but will be reduced to a scalar with tf.reduce_max
        return {
            'decoder_type': get_placeholder('decoder_type', shape=[None], dtype=tf.uint8),
            'learning_rate': get_placeholder_with_default('learning_rate', 1e-4, shape=(), dtype=tf.float32),
            'dropout_rate': get_placeholder_with_default('dropout_rate', 0., shape=(), dtype=tf.float32),
            'is_training': get_placeholder_with_default('is_training', False, shape=(), dtype=tf.bool),
            'stop_gradient_all': get_placeholder_with_default('stop_gradient_all', False, shape=(), dtype=tf.bool)
        }

    def _build_policy_initial(self):
        """ Builds the policy model (initial step) """
        raise NotImplementedError()

    def _get_session_args(self, decode=False, eval_loop_ix=None):
        """ Returns a dict of kwargs to feed to session.run
            Expected format: {fetches, feed_dict=None}
        """
        hps = lambda hparam_name: self.hparams[hparam_name]

        # Detecting if we are doing validation
        in_validation, our_validation = False, False
        if eval_loop_ix is not None:
            in_validation = True
            our_validation = eval_loop_ix in self.my_eval_loop_ixs

        # --------- Fetches ---------------
        train_fetches = {'optimizer_op': self.outputs['optimizer_op'],
                         'policy_loss': self.outputs['policy_loss']}

        eval_fetches = {'policy_loss': self.outputs['policy_loss'],
                        'argmax_tokens': self.outputs['argmax_tokens'],
                        'log_probs': self.outputs['log_probs'],
                        'targets': self.outputs['targets'],
                        'current_power': self.features['current_power'],
                        'current_season': self.features['current_season'],
                        'in_retreat_phase': self.outputs['in_retreat_phase'],
                        'request_id': self.features['request_id']}

        # --------- Feed dict --------------
        # Building feed dict
        feed_dict = {self.placeholders['decoder_type']: [TRAINING_DECODER],         # Batch size of 1
                     self.placeholders['is_training']: True,
                     self.placeholders['stop_gradient_all']: False}

        # Dropout disabled during debug (batch), validation, or decoding (stats)
        if self.hparams['debug_batch'] or in_validation or decode:
            feed_dict.update({self.placeholders['dropout_rate']: 0.})
        else:
            feed_dict.update({self.placeholders['dropout_rate']: hps('dropout_rate')})

        # --------- Validation Loop --------------
        # Validation Loop - Running one of our validation loops
        if our_validation:
            decoder_type = {0: TRAINING_DECODER, 1: GREEDY_DECODER}[self.my_eval_loop_ixs.index(eval_loop_ix)]
            feed_dict[self.placeholders['decoder_type']] = [decoder_type]           # Batch size of 1
            feed_dict[self.placeholders['is_training']] = False
            return {'fetches': eval_fetches, 'feed_dict': feed_dict}

        # Validation Loop - Running someone else validation loop
        if in_validation:
            return {'feed_dict': feed_dict}

        # --------- Training Loop --------------
        # Training Loop - We want to decode the specific batch to display stats
        if decode:
            decoder_type = TRAINING_DECODER
            feed_dict[self.placeholders['decoder_type']] = [decoder_type]           # Batch size of 1
            feed_dict[self.placeholders['is_training']] = False
            return {'fetches': eval_fetches, 'feed_dict': feed_dict}

        # Training Loop - Training the model
        return {'fetches': train_fetches, 'feed_dict': feed_dict}

    @staticmethod
    def _decode(**fetches):
        """ Performs decoding on the output (token_based model)
            :param fetches: A dictionary of fetches from the model.

            Keys can include:

            - selected_tokens / argmax_tokens: [Required] The tokens from the model (Tensor [batch, decoder_length])
            - log_probs: [Required] The log probs from the model (Tensor [batch, decoder_length])
            - policy_loss: The policy loss for the batch.
            - targets: The targets from the model (Tensor [batch, length]). Required for evaluation.
            - current_power: The current_power from the model (Tensor [batch,]). Required for evaluation.
            - current_season: The current_season from the model (Tensor [batch,]). Required for evaluation.
            - in_retreat_phase: Boolean that indicates dislodged units are on the map. ([b,]). Required for evaluation.
            - request_id: The unique request id for each item in the batch.

            :return: A dictionary of decoded results, including
                - 1) decoded_orders:
                    A list of dictionary (one per batch) where each dict has location as key and a
                      OrderProbTokenLogProbs tuple as value (i.e. an order, its prob, and the token log probs)
                        e.g. [{'PAR': (order, prob, log_probs),'MAR': (order, prob, log_probs)},
                              {'PAR': (order, prob, log_probs),'MAR': (order, prob, log_probs)}]
                - 2) various other keys for evaluation
        """
        # Missing the required fetches, returning an empty decoded results
        if ('selected_tokens' not in fetches and 'argmax_tokens' not in fetches) or 'log_probs' not in fetches:
            return {}

        # tokens:           [batch, dec_len]
        # log_probs:        [batch, dec_len]
        # policy_loss:      ()
        # targets:          [batch, dec_len]
        # current_power:    [batch]
        # current_season:   [batch]
        # in_retreat_phase: [batch]
        # request_ids:      [batch]
        tokens = fetches.get('selected_tokens', fetches.get('argmax_tokens'))
        log_probs = fetches['log_probs']
        policy_loss = fetches.get('policy_loss', None)
        targets = fetches.get('targets', None)
        current_power = fetches.get('current_power', None)
        current_season = fetches.get('current_season', None)
        in_retreat_phase = fetches.get('in_retreat_phase', None)
        request_ids = fetches.get('request_id', None)

        # Decoding orders
        results = []
        result_tokens = []
        nb_batches = len(tokens)

        # Building all batches
        for batch_ix in range(nb_batches):
            batch_results = OrderedDict()
            batch_results_tokens = OrderedDict()
            batch_tokens = tokens[batch_ix]
            batch_log_probs = log_probs[batch_ix]
            nb_locs = len(batch_tokens) // TOKENS_PER_ORDER
            nb_waive = 0

            # We didn't try to predict orders - Skipping
            if not len(batch_tokens) or batch_tokens[0] == [0]:                                                         # pylint: disable=len-as-condition
                results += [batch_results]
                result_tokens += [batch_results_tokens]
                continue

            # Decoding each location
            for loc_ix in range(nb_locs):
                loc_tokens = batch_tokens[loc_ix * TOKENS_PER_ORDER:(loc_ix + 1) * TOKENS_PER_ORDER]
                loc_log_probs = batch_log_probs[loc_ix * TOKENS_PER_ORDER:(loc_ix + 1) * TOKENS_PER_ORDER]
                loc_order = ' '.join([ix_to_token(token_ix) for token_ix in loc_tokens if token_ix > EOS_ID])

                # No loc - skipping
                if loc_tokens[0] <= EOS_ID or not loc_order:
                    continue

                # WAIVE orders
                if loc_order == 'WAIVE':
                    loc = 'WAIVE_{}'.format(nb_waive)
                    nb_waive += 1

                # Use normal location and skip if already stored
                else:
                    loc = loc_order.split()[1]
                    if loc in batch_results:
                        continue
                    loc = loc[:3]

                # Otherwise, storing order
                batch_results[loc] = OrderProbTokenLogProbs(order=loc_order,
                                                            probability=1.,
                                                            log_probs=loc_log_probs)
                batch_results_tokens[loc] = loc_tokens

            # Done with batch
            results += [batch_results]
            result_tokens += [batch_results_tokens]

        # Returning
        return {'decoded_orders': results,
                'policy_loss': policy_loss,
                'targets': targets,
                'tokens': result_tokens,
                'current_power': current_power,
                'current_season': current_season,
                'in_retreat_phase': in_retreat_phase,
                'request_id': request_ids,
                'log_probs': log_probs}

    def _evaluate(self, decoded_results, feed_dict, eval_loop_ix, incl_detailed):
        """ Calculates the accuracy of the model
            :param decoded_results: The decoded results (output of _decode() function)
            :param feed_dict: The feed dictionary that was given to session.run()
            :param eval_loop_ix: The current evaluation loop index (-1 for training)
            :param incl_detailed: is true if training is over, more statistics can be computed
            :return: A tuple consisting of:
                        1) An ordered dictionary with result_name as key and (weight, value) as value  (Regular results)
                        2) An ordered dictionary with result_name as key and a list of result values  (Detailed results)
        """
        # Detecting if it's our evaluation or not
        if eval_loop_ix == -1:
            eval_loop_ix = 0
        else:
            our_validation = eval_loop_ix in self.my_eval_loop_ixs
            if not our_validation:
                return OrderedDict(), OrderedDict()
            eval_loop_ix = self.my_eval_loop_ixs.index(eval_loop_ix)

        # Evaluating
        policy_loss = decoded_results['policy_loss'] * TOKENS_PER_ORDER                 # Avg X-Ent per unit-order
        perplexity = math.exp(policy_loss) if policy_loss <= 100 else float('inf')
        targets = decoded_results['targets']
        batch_size = targets.shape[0]
        nb_locs_per_target = targets.shape[1] // TOKENS_PER_ORDER
        decoded_orders = decoded_results['decoded_orders']

        # Logging an error if perplexity is inf
        if perplexity == float('inf'):
            for request_id, log_probs in zip(decoded_results['request_id'], decoded_results['log_probs']):
                if sum(log_probs) <= -100:
                    LOGGER.error('Request %s has log probs that causes a -inf perplexity.', request_id)

        # Accuracy
        acc_1_num, denom = 0., 0.
        acc_1_no_hold_num, denom_no_hold = 0., 0.
        nb_tokens_match, nb_tokens_total = 0., 0.
        acc_player_num, denom_player = 0., 0.

        # Decoding batch by batch, loc by loc
        for batch_ix in range(batch_size):
            player_order_mismatch = False
            nb_waive = 0

            # We didn't learn a policy - Skipping
            if not len(targets[batch_ix]) or targets[batch_ix][0] == 0:                                                 # pylint: disable=len-as-condition
                continue

            for loc_ix in range(nb_locs_per_target):
                start, stop = loc_ix * TOKENS_PER_ORDER, (loc_ix + 1) * TOKENS_PER_ORDER
                decoded_target = ' '.join([ix_to_token(ix) for ix in targets[batch_ix][start:stop] if ix > EOS_ID])
                if not decoded_target:
                    break
                nb_tokens_total += TOKENS_PER_ORDER

                if decoded_target == 'WAIVE':
                    loc = 'WAIVE_{}'.format(nb_waive)
                    is_hold_order = False
                    nb_waive += 1
                else:
                    loc = decoded_target.split()[1][:3]
                    is_hold_order = len(decoded_target.split()) <= 2 or decoded_target.split()[2] == 'H'

                # Computing Acc 1
                denom += 1.
                if not is_hold_order:
                    denom_no_hold += 1.

                # Checking if the target is in the decoded results
                if loc in decoded_orders[batch_ix] and decoded_orders[batch_ix][loc].order == decoded_target:
                    acc_1_num += 1.
                    if not is_hold_order:
                        acc_1_no_hold_num += 1.
                else:
                    player_order_mismatch = True

                # Computing Acc Tokens
                tokenized_targets = targets[batch_ix][start:stop]
                tokenized_results = decoded_results['tokens'][batch_ix].get(loc, [-1] * TOKENS_PER_ORDER)
                nb_tokens_match += sum([1. for i in range(TOKENS_PER_ORDER)
                                        if tokenized_targets[i] == tokenized_results[i]])

            # Compute accuracy for this phase
            if not player_order_mismatch:
                acc_player_num += 1
            denom_player += 1

        # No orders at all
        if not denom:
            acc_1 = 1.
            acc_1_no_hold = 1.
            acc_tokens = 1.
            acc_player = 1.
        else:
            acc_1 = acc_1_num / (denom + 1e-12)
            acc_1_no_hold = acc_1_no_hold_num / (denom_no_hold + 1e-12)
            acc_tokens = nb_tokens_match / (nb_tokens_total + 1e-12)
            acc_player = acc_player_num / (denom_player + 1e-12)

        # Computing detailed statistics
        detailed_results = OrderedDict()
        if incl_detailed:
            detailed_results = self._get_detailed_results(decoded_results, feed_dict, eval_loop_ix)

        # Validating decoder type
        decoder_type = [value for tensor, value in feed_dict.items() if 'decoder_type' in tensor.name]
        decoder_type = '' if not decoder_type else decoder_type[0][0]

        # 0 - Teacher Forcing results
        if eval_loop_ix == 0:
            assert decoder_type == TRAINING_DECODER
            return OrderedDict({'[TF]X-Ent': (denom, policy_loss),
                                '[TF]Perplexity': (denom, perplexity),
                                '[TF]Acc_1': (denom, 100. * acc_1),
                                '[TF]Acc_1_NoHold': (denom_no_hold, 100. * acc_1_no_hold),
                                '[TF]Acc_Tokens': (nb_tokens_total, 100. * acc_tokens),
                                '[TF]Acc_Player': (denom_player, 100. * acc_player)}), detailed_results

        # 1 - Greedy Results
        if eval_loop_ix == 1:
            assert decoder_type == GREEDY_DECODER
            return OrderedDict({'[Gr]Acc_1': (denom, 100. * acc_1),
                                '[Gr]Acc_1_NoHold': (denom_no_hold, 100. * acc_1_no_hold),
                                '[Gr]Acc_Tokens': (nb_tokens_total, 100. * acc_tokens),
                                '[Gr]Acc_Player': (denom_player, 100. * acc_player)}), detailed_results

        # Otherwise, invalid evaluation_loop_ix
        raise RuntimeError('Invalid evaluation_loop_ix - Got "%s"' % eval_loop_ix)

    @staticmethod
    def _get_detailed_results(decoded_results, feed_dict, evaluation_loop_ix):
        """ Computes detailed accuracy statistics for the batch
            :param decoded_results: The decoded results (output of _decode() function)
            :param feed_dict: The feed dictionary that was given to session.run()
            :param eval_loop_ix: The current evaluation loop index
            :return: An ordered dictionary with result_name as key and a list of result values (Detailed results)
        """
        del feed_dict                                           # Unused args

        targets = decoded_results['targets']
        log_probs = decoded_results['log_probs']
        request_ids = decoded_results['request_id']
        batch_size = targets.shape[0]
        nb_locs_per_target = targets.shape[1] // TOKENS_PER_ORDER
        decoded_orders = decoded_results['decoded_orders']

        # Extracting from additional info
        for field_name in ['current_power', 'current_season', 'in_retreat_phase']:
            if field_name not in decoded_results:
                LOGGER.warning('The field "%s" is missing. Cannot compute stats', field_name)
                return OrderedDict()
        current_power_name = [POWER_VOCABULARY_IX_TO_KEY[current_power]
                              for current_power in decoded_results['current_power']]
        current_season_name = ['SFW'[current_season] for current_season in decoded_results['current_season']]
        in_retreat_phase = decoded_results['in_retreat_phase']

        # Prefix
        prefix = '[TF]' if evaluation_loop_ix == 0 else '[Gr]'

        # Building results dict
        results = OrderedDict()
        results[prefix + 'Accuracy'] = []
        results[prefix + 'LogProbsDetails'] = [{}]                          # {request_id: (log_probs, mismatch)}
        for power_name in POWER_VOCABULARY_LIST:
            results[prefix + power_name] = []
        for order_type in ['H', '-', '- VIA', 'S', 'C', 'R', 'B', 'D', 'WAIVE']:
            results[prefix + 'Order %s' % order_type] = []
        for season in 'SFW':                                                # Spring, Fall, Winter
            results[prefix + 'Season %s' % season] = []
        for phase in 'MRA':                                                 # Movement, Retreats, Adjustments
            results[prefix + 'Phase %s' % phase] = []
        for position in range(-1, NB_SUPPLY_CENTERS):                       # Position -1 is used for Adjustment phases
            results[prefix + 'Position %d' % position] = []
        for order_loc in sorted(STANDARD_TOPO_LOCS):                        # Order location
            results[prefix + 'Loc %s' % order_loc] = []

        # Computing accuracy
        for batch_ix in range(batch_size):
            request_id = request_ids[batch_ix]
            player_orders_mismatch = False
            nb_waive = 0

            # We didn't learn a policy - Skipping
            if not len(targets[batch_ix]) or targets[batch_ix][0] == 0:                                                 # pylint: disable=len-as-condition
                continue

            for loc_ix in range(nb_locs_per_target):
                start, stop = loc_ix * TOKENS_PER_ORDER, (loc_ix + 1) * TOKENS_PER_ORDER
                decoded_target = ' '.join([ix_to_token(ix) for ix in targets[batch_ix][start:stop] if ix > EOS_ID])
                if not decoded_target:
                    break

                if decoded_target == 'WAIVE':
                    loc = 'WAIVE_{}'.format(nb_waive)
                    order_type = 'WAIVE'
                    nb_waive += 1
                else:
                    loc = decoded_target.split()[1][:3]
                    order_type = decoded_target.split()[2] if len(decoded_target.split()) > 2 else 'H'
                    if order_type == '-' and decoded_target.split()[-1] == 'VIA':
                        order_type = '- VIA'

                # Determining categories
                power_name = current_power_name[batch_ix]
                season = current_season_name[batch_ix]
                if in_retreat_phase[batch_ix]:
                    phase = 'R'
                    order_type = 'R' if order_type in ['-', '- VIA'] else order_type
                else:
                    phase = {'H': 'M', '-': 'M', '- VIA': 'M', 'S': 'M', 'C': 'M',
                             'R': 'R',
                             'D': 'A', 'B': 'A', 'WAIVE': 'A'}[order_type]

                # Use -1 as position for A phase
                position = -1 if phase == 'A' else loc_ix
                stats_key = StatsKey(prefix, power_name, order_type, season, phase, position)

                # Computing accuracies
                success = int(loc in decoded_orders[batch_ix] and decoded_orders[batch_ix][loc].order == decoded_target)
                if not success:
                    player_orders_mismatch = True

                results[prefix + 'Accuracy'] += [success]
                results[prefix + power_name] += [success]
                results[prefix + 'Order %s' % order_type] += [success]
                results[prefix + 'Season %s' % season] += [success]
                results[prefix + 'Phase %s' % phase] += [success]
                results[prefix + 'Position %d' % position] += [success]
                if order_type != 'WAIVE':
                    results[prefix + 'Loc %s' % loc] += [success]
                results[stats_key] = results.get(stats_key, []) + [success]

            # Storing (log_probs, mismatch)
            results[prefix + 'LogProbsDetails'][0][request_id] = (log_probs[batch_ix].sum(),
                                                                  int(player_orders_mismatch))

        # Returning results
        return results

    @staticmethod
    def _post_process_results(detailed_results):
        """ Perform post-processing on the detailed results
            :param detailed_results: An dictionary which contains detailed evaluation statistics
            :return: A dictionary with the post-processed statistics.
        """
        # Adding [Gr]SearchFailure (== 1. iff. logprob(label) > logprob(greedy) and greedy != label)
        # Adding [TF]Acc_Player and [Gr]Acc_Player
        # Removing LogProbsDetails

        # Make sure the detailed results have the correct key (i.e. they have not yet been post-processed)
        for prefix in ['[TF]', '[Gr]']:
            assert prefix + 'LogProbsDetails' in detailed_results

        # Building a dictionary {request_id: (log_probs, mismatch)}
        tf_items, gr_items = {}, {}
        for tf_item in detailed_results['[TF]LogProbsDetails']:
            tf_items.update(tf_item)
        for gr_item in detailed_results['[Gr]LogProbsDetails']:
            gr_items.update(gr_item)

        # Making sure we have processed the same number of TF items and Gr items
        tf_nb_items = len(tf_items)
        gr_nb_items = len(gr_items)
        if tf_nb_items != gr_nb_items:
            LOGGER.warning('Got a different number of items between [TF] (%d items) and [Gr] (%d items)',
                           tf_nb_items, gr_nb_items)

        # Computing search failure and mismatch
        search_failure, gr_acc_player, tf_acc_player = [], [], []
        for request_id in tf_items:
            if request_id not in gr_items:
                LOGGER.warning('Item %s was computed using [TF], but is missing for [Gr]. Skipping.', request_id)
                continue

            tf_logprobs, tf_mismatch = tf_items[request_id]
            gr_logprobs, gr_mismatch = gr_items[request_id]

            # Computing stats
            if gr_mismatch:
                search_failure += [int(tf_logprobs > gr_logprobs)]
            tf_acc_player += [int(not tf_mismatch)]
            gr_acc_player += [int(not gr_mismatch)]

        # Removing extra keys and adding new keys
        detailed_results['[Gr]SearchFailure'] = search_failure
        detailed_results['[TF]Acc_Player'] = tf_acc_player
        detailed_results['[Gr]Acc_Player'] = gr_acc_player
        del detailed_results['[TF]LogProbsDetails']
        del detailed_results['[Gr]LogProbsDetails']

        # Returning post-processed results
        return detailed_results
