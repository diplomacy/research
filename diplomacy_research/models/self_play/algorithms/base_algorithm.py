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
""" Base Algorithm
    - Represents the base reinforcement learning algorithm
"""
from abc import ABCMeta, abstractmethod
import logging
import math
import random
from tornado import gen
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.self_play.reward_functions import DEFAULT_PENALTY, DEFAULT_GAMMA
from diplomacy_research.models.self_play.transition import ReplaySample, ReplayPriority
from diplomacy_research.models.state_space import GO_ID, get_orderable_locs_for_powers
from diplomacy_research.utils.cluster import yield_with_display
from diplomacy_research.utils.proto import proto_to_dict

# Constants
LOGGER = logging.getLogger(__name__)
VERSION_STEP = 'VERSION_STEP'

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    # Default settings
    args = [
        ('str', 'model_type', 'order_based', 'Policy model family. "token_based", "order_based".'),
        ('int', 'value_model_id', -1, 'The model ID of the value function.'),
        ('str', 'power', '', 'The name of the power the agent should play (e.g. FRANCE) otherwise randomly assigned'),
        ('str', 'mode', '', 'The RL training mode ("supervised", "self-play", "staggered").'),
        ('str', 'eval_mode', 'supervised-1', 'The RL evaluation mode ("supervised-1", "supervised-0".)'),
        ('float', 'gamma', DEFAULT_GAMMA, 'The discount factor to use'),
        ('str', 'advantage_fn', '', 'The advantage function to use with the algorithm (different for each algo)'),
        ('str', 'start_strategy', 'beginning', 'Strategy to select initial state. "beginning", "uniform", "backplay"'),
        ('str', 'expert_dataset', '', 'A comma-separated list from no_press, press.'),
        ('float', 'experience_replay', 0., 'Float between 0 and 1 - % of transitions to sample from replay buffer.'),
        ('float', 'replay_alpha', 0.7, 'The alpha value to use to compute the priorities from the memory buffer'),
        ('int', 'min_buffer_items', 10000, 'The min. nb of items in the buffer required for experience replay'),
        ('int', 'max_buffer_items', 2000000, 'The maximum nb of items to store in the memory buffer'),
        ('float', 'penalty_per_phase', DEFAULT_PENALTY, 'The penalty to add per phase'),
        ('str', 'reward_fn', 'DefaultRewardFunction', 'The reward function to use'),
        ('int', 'max_nb_years', 35, 'The maximum number of years to play before ending a game'),
        ('int', 'nb_transitions_per_update', 2500, 'The number of transitions required to update the params.'),
        ('int', 'nb_evaluation_games', 100, 'The number of evaluation games per version update'),
        ('int', 'nb_thrashing_states', 3, 'If > 0, stops the game if this nb of identical states are detected.'),
        ('int', 'update_interval', 30, 'If > 0, actors send partial trajectories after this many seconds.'),
        ('bool', 'auto_draw', True, 'Games should be auto-drawn according to the supervised dataset prob.'),
        ('bool', 'sync_gradients', True, 'Indicates that gradients are synchronized among RL workers.'),
        ('bool', 'avg_gradients', True, 'Indicates that gradients are average across multiple mini-batches.'),
        ('float', 'clip_value_threshold', 5., 'The maximum square diff between the prev and curr value of a state'),
        ('float', 'dropout_rate', 0., 'The dropout rate to apply to players and opponents during training.'),
        ('bool', 'use_beam', True, 'Use the beam search decoder to select orders during training'),
        ('int', 'staggered_versions', 500, 'For mode=="staggered", how often to update the opponent version.'),
        ('str', 'training_mode', 'reinforcement', 'The current training mode ("supervised" or "reinforcement").'),
        ('int', 'eval_every', 125, 'Wait this nb of versions before running evaluation games.'),
        ('str', 'eval_reward_fns', 'DrawSizeReward,SumOfSquares,SurvivorWinReward', 'List of rew fns for evaluation'),
        ('str', 'ignored_scope', 'message', 'Scope(s) (comma separated) to ignore by the optimizer op.'),
        ('float', 'learning_rate', 1e-4, 'Initial learning rate.'),
        ('float', 'lr_decay_factor', 1., 'Learning rate decay factor.'),
        ('---', 'early_stopping_stop_after', 0, '[Deleting] For supervised learning only.')
    ]

    # Returning
    return args

class BaseAlgorithm(metaclass=ABCMeta):
    """ Base Algorithm - Abstract class that represents a RL algorithm """

    # Static Properties
    can_do_experience_replay = False

    def __init__(self, queue_dataset, model, hparams):
        """ Constructor
            :param queue_dataset: The dataset used to feed data into the model
            :param model: The underlying supervised model
            :param flags: The parsed flags (Tensorflow arguments)
            :param hparams: A dictionary of hyper parameters with their values
            :type queue_dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
            :type model: diplomacy_research.models.policy.base_policy_model.BasePolicyModel
        """
        assert isinstance(queue_dataset, QueueDataset), 'Expected a QueueDataset for the RL algorithm'
        self.hparams = hparams
        self.cluster_config = queue_dataset.cluster_config
        self.transition_buffer = []
        self.list_power_phases_per_game = {}            # {game_id: [(power_name, phase_ix)]}
        self.proto_fields = queue_dataset.proto_fields
        self.sample_futures = []                        # List of futures from .sample()

        # Building RL model
        self.queue_dataset = queue_dataset
        self.model = model
        self.model.validate()
        self.version_step = self.get_or_create_version_step()
        self.model.version_step = self.version_step
        self.build()

    @staticmethod
    def get_proto_fields():
        """ Returns the proto fields used by this algorithm """
        raise NotImplementedError()

    @staticmethod
    def get_evaluation_tags():
        """ Returns a list of fields that are computed during .update() to display in the stats"""
        return []

    @staticmethod
    def create_advantage_function(hparams, *, gamma, penalty_per_phase=0.):
        """ Returns the advantage function to use with this algorithm """
        raise NotImplementedError()

    def get_or_create_version_step(self):
        """ Gets or creates the version step """
        version_step_tensor = self._get_version_step()
        if version_step_tensor is None:
            version_step_tensor = self._create_version_step()
        return version_step_tensor

    @abstractmethod
    def build(self):
        """ Builds the RL model using the correct optimizer """
        raise NotImplementedError()

    @gen.coroutine
    def learn(self, saved_games_proto, all_power_phases_ix, advantage_function):
        """ Learns (Calculate gradient updates) from transitions in a saved game
            :param saved_games_proto: A list of `.proto.game_pb2.SavedGame` protocol buffer instances.
            :param all_power_phases_ix: A list of {power_name: [phases_ix]} for learning (1 list per game)
            :param advantage_function: An instance of `.models.self_play.advantages`.
            :type advantage_function: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
        """
        yield self._learn(saved_games_proto, all_power_phases_ix, advantage_function)

    @gen.coroutine
    def sample(self, queue_name, wait_per_mini_batch=False):
        """ Samples items from the transition buffer and puts them in the learning queue
            :param queue_name: The name of the queue where to put items
            :param wait_per_mini_batch: Boolean. If true, yield for each mini-batch, otherwise just put data in queues.
        """
        if len(self.transition_buffer) < 2:
            LOGGER.warning('Not enough items in the transition buffer to sample(). Expected at least 2.')
            return

        # Sampling transition buffer, in full increments of batch_size.
        nb_learners = 1 if not self.cluster_config else self.cluster_config.count('learner')
        max_items_to_sample = 1.5 * self.hparams['nb_transitions_per_update'] / nb_learners
        nb_samples = self.hparams['batch_size'] * \
                     (min(max_items_to_sample, len(self.transition_buffer)) // self.hparams['batch_size'])
        if nb_samples == 0:
            LOGGER.warning('Not enough data in the buffer to sample a full batch of data.')
            LOGGER.warning('Got: %d - Wanted batch of: %d.', len(self.transition_buffer), self.hparams['batch_size'])
            nb_samples = len(self.transition_buffer)
        nb_samples = int(nb_samples)
        LOGGER.info('Sampling %d items from the transition buffer.', nb_samples)

        # Sampling
        # Waiting for each mini-batch
        if wait_per_mini_batch:
            batch_size = self.hparams['batch_size']
            nb_batches = int(math.ceil(nb_samples / self.hparams['batch_size']))

            # Running all mini-batches
            for batch_ix in range(nb_batches):
                batch_items = [item for item in random.sample(self.transition_buffer,
                                                              min(len(self.transition_buffer), batch_size))]
                futures = [self.queue_dataset.get_results(queue_name, item) for item in batch_items]
                self.queue_dataset.last_queue = ''
                yield yield_with_display(futures, every=60, timeout=900)
                if (batch_ix) + 1 % 10 == 0 or (batch_ix + 1) == nb_batches:
                    LOGGER.info('Running mini-batch %d/%d.', batch_ix + 1, nb_batches)
                self.sample_futures += futures

        # Setting the entire buffer in memory, without waiting for results
        else:
            batch_items = [item for item in random.sample(self.transition_buffer,
                                                          min(len(self.transition_buffer), nb_samples))]
            for item in batch_items:
                self.sample_futures += [self.queue_dataset.get_results(queue_name, item)]

        LOGGER.info('Done sampling items from the transition buffer.')

    @abstractmethod
    @gen.coroutine
    def update(self, memory_buffer):
        """ Calculates the average gradients and applies them
            :param memory_buffer: An instance of memory buffer (for distributed training) or None for non-distributed.
            :return: A dictionary of results with evaluation_tags as key, and a list as value
            :type memory_buffer: diplomacy_research.models.self_play.memory_buffer.MemoryBuffer
        """
        raise NotImplementedError()

    @gen.coroutine
    def init(self):
        """ Initializes the algorithm and its optimizer. """
        raise NotImplementedError()

    def get_results(self):
        """ Retrieves the evaluation results from the last update / version
            :return: A dictionary with evaluation tag as key and a list of results as value
        """
        results = {eval_tag: [] for eval_tag in self.get_evaluation_tags()}

        # Processing each future
        for future in self.sample_futures:
            result = future.result()
            if not result:
                continue
            for eval_tag, eval_result in zip(self.get_evaluation_tags(), result[1:]):
                results[eval_tag] += [eval_result]

        # Returning
        return results

    @staticmethod
    def get_power_phases_ix(saved_game_proto, nb_rl_agents):
        """ Computes the list of (power_name, phase_ix) for the given game to learn
            :param saved_game_proto: An instance of `.proto.game_pb2.SavedGame`
            :param nb_rl_agents: The number of RL agents in the current mode.
            :return: A dictionary with power name as key and a list of phase_ix for that power as value
        """
        nb_phases = len(saved_game_proto.phases)
        rl_powers = saved_game_proto.assigned_powers[:nb_rl_agents]
        start_phase_ix = saved_game_proto.start_phase_ix

        # Returning
        return {power_name: [phase_ix for phase_ix in range(start_phase_ix, nb_phases - 1)]
                for power_name in rl_powers}

    @gen.coroutine
    def get_priorities(self, saved_games_proto, advantage_function):
        """ Computes the (original) replay priorities for the generated games
            :param saved_games_proto: An list of `.proto.game_pb2.SavedGame` protocol buffer instances.
            :param advantage_function: An instance of `.models.self_play.advantages`.
            :return: A list of `ReplayPriority` objects.
            :type advantage_function: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
            :rtype: [diplomacy_research.models.self_play.transition.ReplayPriority]
        """
        # Converting games to replay samples
        # Only keeping transitions where we know we issued orders
        replay_samples = []
        for saved_game_proto in saved_games_proto:

            # Converting [(power, phase_ix)] to {power: [phases_ix]}
            list_power_phases = self.list_power_phases_per_game.get(saved_game_proto.id, [])
            if list_power_phases:
                power_phases_ix = {}
                for power_name, phase_ix in list_power_phases:
                    if power_name not in power_phases_ix:
                        power_phases_ix[power_name] = []
                    power_phases_ix[power_name] += [phase_ix]
                replay_samples += [ReplaySample(saved_game_proto, power_phases_ix=power_phases_ix)]

        # Getting new priorities
        priorities = yield self.get_new_priorities(replay_samples, advantage_function)
        return priorities

    @staticmethod
    @gen.coroutine
    def get_new_priorities(replay_samples, advantage_function):
        """ Computes the new replay priorities for the replay samples retrieved
            :param replay_samples: A list of `ReplaySample` objects
            :param advantage_function: An instance of `.models.self_play.advantages`.
            :return: A list of `ReplayPriority` objects.
            :type replay_samples: [diplomacy_research.models.self_play.transition.ReplaySample]
            :type advantage_function: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
            :rtype: [diplomacy_research.models.self_play.transition.ReplayPriority]
        """
        # --- Currently disabled ---
        # --- This method will always return a new priority of 0, because it is too expensive to compute otherwise ---
        del advantage_function              # Unused args
        priorities = []
        for replay_sample in replay_samples:
            for power_name in replay_sample.power_phases_ix:
                for phase_ix in replay_sample.power_phases_ix[power_name]:
                    priorities += [ReplayPriority(game_id=replay_sample.saved_game_proto.id,
                                                  power_name=power_name,
                                                  phase_ix=phase_ix,
                                                  priority=0.)]
        return priorities

    @gen.coroutine
    def clear_buffers(self):
        """ Finalizes the version update, clears the buffers and updates the memory buffer if needed """
        self.transition_buffer = []
        self.list_power_phases_per_game = {}
        self.sample_futures = []

    # -------------------------------------
    # ---------  Private Methods ----------
    # -------------------------------------
    def _add_placeholder(self, placeholder_name, placeholder_value):
        """ Adds a placeholder to the model """
        from diplomacy_research.utils.tensorflow import tf
        if not self.model:
            LOGGER.warning('Cannot add the placeholder "%s". Model has not been built.', placeholder_name)
            return

        # Storing in collection
        cached_placeholder = tf.get_collection('placeholder_{}'.format(placeholder_name))
        if not cached_placeholder:
            tf.add_to_collection('placeholder_{}'.format(placeholder_name), placeholder_value)
        self.model.placeholders[placeholder_name] = placeholder_value

    def _add_output(self, output_name, output_value):
        """ Adds an output to the model """
        from diplomacy_research.utils.tensorflow import tf
        if not self.model:
            LOGGER.warning('Cannot add the output "%s". Model has not been built.', output_name)

        # Storing output in collection
        self.model.outputs[output_name] = output_value
        tf.add_to_collection(output_name, output_value)

    @staticmethod
    def _get_version_step():
        """ Gets the version step tensor """
        from diplomacy_research.utils.tensorflow import tf
        graph = tf.get_default_graph()
        version_step_tensors = graph.get_collection(VERSION_STEP)
        if not version_step_tensors:
            return None
        if len(version_step_tensors) == 1:
            return version_step_tensors[0]
        raise RuntimeError('Multiple version step tensors defined')

    @staticmethod
    def _create_version_step():
        """ Creates the version step tensor if it doesn't exist """
        from diplomacy_research.utils.tensorflow import tf
        if BaseAlgorithm._get_version_step() is not None:
            raise ValueError('"version_step" already exists.')
        with tf.get_default_graph().name_scope(None):
            return tf.get_variable(VERSION_STEP,
                                   shape=(),
                                   dtype=tf.int64,
                                   initializer=tf.zeros_initializer(),
                                   trainable=False,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, VERSION_STEP])

    @staticmethod
    def _post_process_transition_details(transition_details):
        """ Performs some post-processing on the transition details.
            Required by REINFORCE to compute average rewards
            :param transition_details: A list of TransitionDetails named tuple.
                    Each transition details contains
                        1) a `.models.self_play.transition.Transition` transition.
                        2) the corresponding reward target to be used for the policy gradient update,
                        3) the corresponding value target to be used for the critic update,
                        4) a list of log importance sampling ratio for each output token or None
                        5) the updated (current) log probs for each token in the model
            :return: The updated transition_details
        """
        return transition_details

    @gen.coroutine
    def _learn(self, saved_games_proto, all_power_phases_ix, advantage_function):
        """ Learns (Calculate gradient updates) from transitions in a saved game
            :param saved_games_proto: A list of `.proto.game_pb2.SavedGame` protocol buffer instances.
            :param all_power_phases_ix: A list of {power_name: [phases_ix]} for learning (1 list per game)
            :param advantage_function: An instance of `.models.self_play.advantages`.
            :type advantage_function: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
        """
        assert len(saved_games_proto) == len(all_power_phases_ix), 'Expecting one power_phases_ix per game'

        # Stop learning if we have more than 2x the number of target transitions
        nb_learners = 1 if not self.cluster_config else self.cluster_config.count('learner')
        max_nb_transitions = 2 * self.hparams['nb_transitions_per_update'] / nb_learners

        # Processing each saved game
        for saved_game_proto, power_phases_ix in zip(saved_games_proto, all_power_phases_ix):

            power_warned = []
            kwargs = {power_name: proto_to_dict(saved_game_proto.kwargs[power_name])
                      for power_name in saved_game_proto.kwargs}

            for power_name, phases_ix in power_phases_ix.items():
                transition_details = advantage_function.get_transition_details(saved_game_proto, power_name, **kwargs)
                transition_details = [transition_detail for transition_detail in transition_details
                                      if transition_detail.transition.phase_ix in phases_ix]
                transition_details = self._post_process_transition_details(transition_details)

                # Warning if we didn't get all the transition details we asked for
                if len(phases_ix) != len(transition_details):
                    LOGGER.warning('Asked for details of %d transitions. Got %s.',
                                   len(power_phases_ix), len(transition_details))

                # Looping over the locations on which we want to learn
                for transition_detail in transition_details:
                    transition = transition_detail.transition
                    power_name = transition_detail.power_name
                    phase_ix = transition.phase_ix
                    game_id = transition.state.game_id

                    # No orders
                    if transition.policy is not None \
                            and not transition.orders[power_name].value \
                            and not transition.policy[power_name].locs:
                        continue

                    # We are missing the details of the policy to learn, issuing a warning
                    if transition.policy is None \
                            or power_name not in transition.policy \
                            or not transition.policy[power_name].tokens \
                            or (transition.orders[power_name].value and not transition.policy[power_name].log_probs):
                        if power_name not in power_warned:
                            LOGGER.warning('The policy details for %s are missing for phase %d (Game %s). Skipping.',
                                           power_name,
                                           phase_ix,
                                           saved_game_proto.id)
                            power_warned += [power_name]
                        continue

                    # For adjustment phase, we need to use all the orderable locs
                    if transition.state.name[-1] == 'A':
                        locs, _ = get_orderable_locs_for_powers(transition.state, [power_name])
                    else:
                        locs = transition.policy[power_name].locs

                    # Getting feedable item
                    item = self.queue_dataset.\
                        get_feedable_item(locs=locs,
                                          state_proto=transition.state,
                                          power_name=power_name,
                                          phase_history_proto=transition.phase_history,
                                          possible_orders_proto=transition.possible_orders,
                                          **kwargs[power_name])

                    # No item - Skipping
                    if not item:
                        continue

                    current_tokens = list(transition.policy[power_name].tokens)
                    item['decoder_inputs'] = [GO_ID] + current_tokens
                    item['decoder_lengths'] = len(current_tokens)

                    # Algo-specific fields
                    if 'draw_action' in self.proto_fields:
                        item['draw_action'] = transition_detail.draw_action
                    if 'reward_target' in self.proto_fields:
                        item['reward_target'] = transition_detail.reward_target
                    if 'value_target' in self.proto_fields:
                        item['value_target'] = transition_detail.value_target
                    if 'old_log_probs' in self.proto_fields:
                        item['old_log_probs'] = transition_detail.log_probs
                    if 'log_importance_sampling_ratio' in self.proto_fields and transition_detail.log_p_t is not None:
                        item['log_importance_sampling_ratio'] = transition_detail.log_p_t

                    # Storing item
                    self.transition_buffer.append(item)
                    self.list_power_phases_per_game.setdefault(game_id, [])
                    self.list_power_phases_per_game[game_id] += [(power_name, phase_ix)]

                    # Max number of transitions reached
                    if len(self.transition_buffer) >= max_nb_transitions:
                        return

    @gen.coroutine
    def _update(self, update_queue_name):
        """ Calculates the average gradients and applies them
            :param update_queue_name: The name of the update_op queue
        """
        if self.cluster_config and not self.cluster_config.is_chief:
            LOGGER.error('Only the chief can run the update() op. Aborting.')
            return
        if len(self.transition_buffer) < 2:
            LOGGER.warning('Not enough items in the transition buffer to update(). Expected at least 2.')
            return

        # Setting last queue as '', so learning can start
        update_future = self.queue_dataset.get_results(update_queue_name, item={})
        self.queue_dataset.last_queue = ''
        yield yield_with_display(update_future, every=120, timeout=1800)

    @gen.coroutine
    def _run_init(self, init_queue_name):
        """ Runs the algorithm and its optimizer initialization.
            :param init_queue_name: The name of the init_op queue
        """
        # Continuing with a warning if this operating hangs
        try:
            update_future = self.queue_dataset.get_results(init_queue_name, item={})
            yield yield_with_display(update_future, every=30, timeout=120)
        except TimeoutError:
            LOGGER.warning('Unable to init() the accumulators. This might happen if they are already empty.')
