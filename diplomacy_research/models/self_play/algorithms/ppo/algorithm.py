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
""" PPO Algorithm
    - Application of Proximal Policy Optimization (1707.06347)
"""
import logging
import numpy as np
from tornado import gen
from diplomacy_research.models.datasets.base_builder import FixedProtoField, VarProtoField
from diplomacy_research.models.policy.base_policy_model import TRAINING_DECODER, GREEDY_DECODER
from diplomacy_research.models.state_space import TOKENS_PER_ORDER, NB_SUPPLY_CENTERS
from diplomacy_research.models.self_play.advantages import MonteCarlo, NStep, GAE, VTrace
from diplomacy_research.models.self_play.algorithms.base_algorithm import BaseAlgorithm, load_args as load_parent_args
from diplomacy_research.models.training.memory_buffer.barrier import set_barrier_status, wait_for_barrier

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    # Default settings
    args = load_parent_args() + [
        ('bool', 'avg_gradients', False, 'For distributed rl training, uses AvgGradOptimizer'),
        ('str', 'advantage_fn', 'gae', 'The advantage function to use (valid: monte_carlo, n_step, gae)'),
        ('int', 'nb_steps', 15, 'The number of steps (phases) for the advantage function'),
        ('float', 'lambda_', 0.9, 'The value of the lambda parameter to use for GAE'),
        ('float', 'c_bar', 1., 'The value of the c_bar parameter for V-Trace'),
        ('float', 'p_bar', 1., 'The value of the p_bar parameter for V-Trace'),

        ('int', 'nb_mini_epochs', 3, 'The number of mini-epochs to use for parameter updates'),
        ('float', 'epsilon', 0.2, 'The epsilon to use to clip the importance sampling ratio.'),
        ('float', 'value_coeff', 0.5, 'The coefficient to apply to the value loss'),
        ('float', 'draw_coeff', 0.5, 'The coefficient to apply to the draw policy loss'),
        ('float', 'entropy_coeff', 0.1, 'The coefficient to apply to the entropy loss'),
    ]
    return args

class PPOAlgorithm(BaseAlgorithm):
    """ PPO Algorithm - Implements the PPO algorithm (1707.06347) """

    @staticmethod
    def get_proto_fields():
        """ Returns the proto fields used by this algorithm """
        return {'draw_action': FixedProtoField([], np.bool),
                'draw_target': FixedProtoField([], np.float32),
                'reward_target': FixedProtoField([], np.float32),
                'value_target': FixedProtoField([], np.float32),
                'old_log_probs': VarProtoField([TOKENS_PER_ORDER * NB_SUPPLY_CENTERS], np.float32)}

    @staticmethod
    def get_evaluation_tags():
        """ Returns a list of fields that are computed during .update() to display in the stats"""
        return ['rl_policy_loss', 'rl_value_loss', 'rl_draw_loss', 'rl_entropy_loss', 'rl_total_loss']

    @staticmethod
    def create_advantage_function(hparams, *, gamma, penalty_per_phase=0.):
        """ Returns the advantage function to use with this algorithm
            :type adapter: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter
            :type reward_fn: diplomacy_research.models.self_play.reward_functions.AbstractRewardFunction
        """
        # Monte-Carlo
        if hparams['advantage_fn'] == 'monte_carlo':
            return MonteCarlo(gamma=gamma,
                              penalty_per_phase=penalty_per_phase)

        # N-Step
        if hparams['advantage_fn'] in ['n_step', 'nstep']:
            return NStep(nb_steps=hparams['nb_steps'],
                         gamma=gamma,
                         penalty_per_phase=penalty_per_phase)

        # GAE
        if hparams['advantage_fn'] == 'gae':
            return GAE(lambda_=hparams['lambda_'],
                       gamma=gamma,
                       penalty_per_phase=penalty_per_phase)

        # V-Trace
        if hparams['advantage_fn'] in ('v_trace', 'vtrace'):
            return VTrace(lambda_=hparams['lambda_'],
                          c_bar=hparams['c_bar'],
                          p_bar=hparams['p_bar'],
                          gamma=gamma,
                          penalty_per_phase=penalty_per_phase)

        # Invalid advantage fn
        raise ValueError('Invalid advantage function. Got %s. Valid values: %s' %
                         (hparams['advantage_fn'], ['monte_carlo', 'n_step', 'gae', 'v_trace']))

    def build(self):
        """ Builds the RL model using the correct optimizer """
        from diplomacy_research.utils.tensorflow import tf, tfp, normalize, to_float
        from diplomacy_research.models.layers.avg_grad_optimizer import AvgGradOptimizer

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.model.hparams[hparam_name]

        # Training loop
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):

                # Placeholders
                stop_gradient_all = self.model.placeholders['stop_gradient_all']

                # Features
                decoder_lengths = self.model.features['decoder_lengths']            # tf.int32   - (b,)
                draw_action = self.model.features['draw_action']                    # tf.bool    - (b,)
                reward_target = self.model.features['reward_target']                # tf.float32 - (b,)
                value_target = self.model.features['value_target']                  # tf.float32 - (b,)
                old_log_probs = self.model.features['old_log_probs']                # tf.float32 - (b, dec_len)
                # current_power = self.model.features['current_power']              # tf.int32   - (b,)

                # Making sure all RNN lengths are at least 1
                # Trimming to the maximum decoder length in the batch
                raw_decoder_lengths = decoder_lengths
                decoder_lengths = tf.math.maximum(1, decoder_lengths)

                # Retrieving model outputs
                baseline = values = self.model.outputs['state_value']                               # (b,)
                logits = self.model.outputs['logits']                                               # (b, dec, VOCAB)
                sequence_mask = tf.sequence_mask(raw_decoder_lengths,                               # (b, dec)
                                                 maxlen=tf.reduce_max(decoder_lengths),
                                                 dtype=tf.float32)

                # Computing Baseline Mean Square Error Loss
                with tf.variable_scope('baseline_scope'):
                    baseline_mse_loss = tf.minimum(tf.square(value_target - values), hps('clip_value_threshold'))
                    baseline_mse_loss = tf.reduce_sum(baseline_mse_loss)                                # ()

                # Calculating surrogate loss
                with tf.variable_scope('policy_gradient_scope'):
                    new_policy_log_probs = self.model.outputs['log_probs'] * sequence_mask              # (b, dec_len)
                    old_policy_log_probs = old_log_probs * sequence_mask                                # (b, dec_len)

                    new_sum_log_probs = tf.reduce_sum(new_policy_log_probs, axis=-1)                    # (b,)
                    old_sum_log_probs = tf.reduce_sum(old_policy_log_probs, axis=-1)                    # (b,)

                    ratio = tf.math.exp(new_sum_log_probs - old_sum_log_probs)                          # (b,)
                    clipped_ratio = tf.clip_by_value(ratio, 1. - hps('epsilon'), 1. + hps('epsilon'))   # (b,)
                    advantages = tf.stop_gradient(normalize(reward_target - baseline))                  # (b,)

                    surrogate_loss_1 = ratio * advantages                                                   # (b,)
                    surrogate_loss_2 = clipped_ratio * advantages                                           # (b,)
                    surrogate_loss = -tf.reduce_mean(tf.math.minimum(surrogate_loss_1, surrogate_loss_2))   # ()

                # Calculating policy gradient for draw action
                with tf.variable_scope('draw_gradient_scope'):
                    draw_action = to_float(draw_action)                                                 # (b,)
                    draw_prob = self.model.outputs['draw_prob']                                         # (b,)
                    log_prob_of_draw = draw_action * tf.log(draw_prob) + (1. - draw_action) * tf.log(1. - draw_prob)
                    draw_gradient_loss = -1. * log_prob_of_draw * advantages                            # (b,)
                    draw_gradient_loss = tf.reduce_mean(draw_gradient_loss)                             # ()

                # Calculating entropy loss
                with tf.variable_scope('entropy_scope'):
                    entropy = tfp.distributions.Categorical(logits=logits).entropy()
                    entropy_loss = -tf.reduce_mean(entropy)                                             # ()

                # Scopes
                scope = ['policy', 'value', 'draw']
                global_ignored_scope = None if not hps('ignored_scope') else hps('ignored_scope').split(',')

                # Creating PPO loss
                ppo_loss = surrogate_loss \
                           + hps('value_coeff') * baseline_mse_loss \
                           + hps('draw_coeff') * draw_gradient_loss \
                           + hps('entropy_coeff') * entropy_loss
                ppo_loss = tf.cond(stop_gradient_all,
                                   lambda: tf.stop_gradient(ppo_loss),              # pylint: disable=cell-var-from-loop
                                   lambda: ppo_loss)                                # pylint: disable=cell-var-from-loop
                cost_and_scope = [(ppo_loss, scope, None)]

                # Creating optimizer op
                ppo_op = self.model.create_optimizer_op(cost_and_scope=cost_and_scope,
                                                        ignored_scope=global_ignored_scope,
                                                        max_gradient_norm=hps('max_gradient_norm'))

                # Making sure we are not using the AvgGradOptimizer, but directly the AdamOptimizer
                assert not isinstance(self.model.optimizer, AvgGradOptimizer), 'PPO does not use AvgGradOptimizer'

        # Storing outputs
        self._add_output('rl_policy_loss', surrogate_loss)
        self._add_output('rl_value_loss', baseline_mse_loss)
        self._add_output('rl_draw_loss', draw_gradient_loss)
        self._add_output('rl_entropy_loss', entropy_loss)
        self._add_output('rl_total_loss', ppo_loss)
        self._add_output('optimizer_op', ppo_op)

        # --------------------------------------
        #               Hooks
        # --------------------------------------
        def hook_baseline_pre_condition(dataset):
            """ Pre-Condition: First queue to run """
            if not hasattr(dataset, 'last_queue') or dataset.last_queue == '':
                return True
            return False

        def hook_baseline_post_queue(dataset):
            """ Post-Queue: Marks the baseline queue as processed """
            dataset.last_queue = 'ppo_policy_baseline'

        # --------------------------------------
        #               Queues
        # --------------------------------------
        self.queue_dataset.create_queue('ppo_policy_baseline',
                                        placeholders={self.model.placeholders['decoder_type']: [TRAINING_DECODER]},
                                        outputs=[self.model.outputs[output_name]
                                                 for output_name in ['optimizer_op'] + self.get_evaluation_tags()],
                                        pre_condition=hook_baseline_pre_condition,
                                        post_queue=hook_baseline_post_queue)
        self.queue_dataset.create_queue('ppo_increase_version',
                                        placeholders={self.model.placeholders['decoder_type']: [GREEDY_DECODER]},
                                        outputs=[tf.assign_add(self.version_step, 1)],
                                        with_status=True)

    @gen.coroutine
    def update(self, memory_buffer):
        """ Calculates the average gradients and applies them
            :param memory_buffer: An instance of memory buffer (for distributed training) or None for non-distributed.
            :return: A dictionary of results with evaluation_tags as key, and a list as value
            :type memory_buffer: diplomacy_research.models.self_play.memory_buffer.MemoryBuffer
        """
        assert memory_buffer is not None or self.cluster_config is None, 'Memory buffer required for dist. training'
        for epoch_ix in range(self.hparams['nb_mini_epochs']):
            yield self.sample(queue_name='ppo_policy_baseline', wait_per_mini_batch=True)

            # Not distributed - Continuing
            if not self.cluster_config:
                continue

            # Distributed - Non Chief
            # Setting status on barrier and waiting for all learners to complete epoch
            if not self.cluster_config.is_chief:
                nb_learners = self.cluster_config.count('learner')
                set_barrier_status(memory_buffer, 'train', value=epoch_ix + 1)
                wait_for_barrier(memory_buffer,
                                 barrier_name='train',
                                 job_name='learner',
                                 min_value=epoch_ix + 1,
                                 min_done=nb_learners)
                continue

            # Distributed - Chief
            # Waiting for all learners to have completed, then clear barrier
            nb_learners = self.cluster_config.count('learner')
            wait_for_barrier(memory_buffer,
                             barrier_name='train',
                             job_name='learner',
                             min_value=epoch_ix + 1,
                             min_done=nb_learners - 1)
            set_barrier_status(memory_buffer, 'train', value=epoch_ix + 1)

        # Increasing version if non-distributed or if chief
        if not self.cluster_config or self.cluster_config.is_chief:
            yield self.queue_dataset.get_results('ppo_increase_version', item={})
            return self.get_results()

        # Returning empty results if distributed and non-chief
        return {}

    @gen.coroutine
    def init(self):
        """ Initializes the algorithm and its optimizer. """
        # PPO does not use gradient accumulator and does not require initialization.
