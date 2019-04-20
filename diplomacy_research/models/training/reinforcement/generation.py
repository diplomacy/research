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
""" Reinforcement - Generation
    - Class responsible for generating games
"""
import collections
import datetime
import logging
import multiprocessing
from queue import Empty as EmptyQueueException
import random
import sys
from threading import Thread
import time
import traceback
from tornado import gen, ioloop
from diplomacy_research.models.self_play.controller import generate_trajectory
from diplomacy_research.models.state_space import ALL_POWERS, NB_POWERS
from diplomacy_research.models.training.memory_buffer import MemoryBuffer, general_ops
from diplomacy_research.models.training.memory_buffer.expert_games import get_uniform_initial_states, \
    get_backplay_initial_states
from diplomacy_research.models.training.memory_buffer.online_games import save_games
from diplomacy_research.models.training.memory_buffer import priority_replay
from diplomacy_research.models.training.reinforcement.common import get_nb_rl_agents
from diplomacy_research.models.training.reinforcement.players import get_train_supervised_players, \
    get_train_self_play_players, get_train_staggered_players
from diplomacy_research.models.training.reinforcement.serving import get_tf_serving_port
from diplomacy_research.utils.proto import write_bytes_to_file, proto_to_bytes

# Constants
LOGGER = logging.getLogger(__name__)
SPAWN_CONTEXT = multiprocessing.get_context('spawn')
NEW_VERSION = 'new.version'

class AggregatorConfig(
        collections.namedtuple('AggregatorConfig', ('nb_left',              # The number of games left to generate
                                                    'nb_total',             # The total number of games to generate
                                                    'file_path',            # The file path where to save games
                                                    'save_to_buffer'))):    # Indicate to save the game to mem. buffer
    """ Configuration to pass to the aggregator on a new set of games """

@gen.coroutine
def generate_game_on_process(get_players_callable, get_players_kwargs, generate_trajectory_kwargs, queue):
    """ Generate a game on the current process
        :param get_players_callable: Callable function to get a list of players
        :param get_players_kwargs: A dictionary of kwargs to pass to the get_players_callable
        :param generate_trajectory_kwargs: A dictionary of kwargs to pass to the method generate_trajectory
        :param queue: A multiprocessing queue to save games to disk and display progress.
        :return: A saved game in bytes format.
        :type queue: multiprocessing.Queue
    """
    assert 'tensorflow' not in sys.modules, 'TensorFlow should not be loaded for generate games.'
    players = get_players_callable(**get_players_kwargs)
    saved_game_bytes = yield generate_trajectory(players, **generate_trajectory_kwargs, output_format='bytes')
    queue.put_nowait((True, 0, saved_game_bytes))           # is_full_game, nb_transitions, saved_game_bytes
    return saved_game_bytes

def start_game_process(kwargs):
    """ Starts an IO loop for game generation """
    try:
        return ioloop.IOLoop().run_sync(lambda: generate_game_on_process(**kwargs))                                     # pylint: disable=no-value-for-parameter
    except KeyboardInterrupt:
        pass
    except:                                                                                                             # pylint: disable=bare-except
        LOGGER.error('-' * 32)
        LOGGER.error('Exception occurred in process pool.')
        traceback.print_exc()
        LOGGER.error('-' * 32)
        return None

def start_game_generator(adapter_ctor, dataset_builder_ctor, reward_fn, advantage_fn, hparams, cluster_config,
                         process_pool, games_queue, transitions_queue):
    """ Start the game generator (to generate an infinite number of training games)
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param reward_fn: The reward function to use (Instance of.models.self_play.reward_functions`).
        :param advantage_fn: An instance of `.models.self_play.advantages`
        :param hparams: A dictionary of hyper-parameters
        :param cluster_config: The cluster configuration to use for distributed training
        :param process_pool: Optional. A ProcessPoolExecutor that was forked before TF and gRPC were loaded.
        :param games_queue: Queue to be used by processes to send games to the aggregator.
        :param transitions_queue: Inbound queue to receive the number of transitions and version updates.
        :return: Nothing
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type reward_fn: diplomacy_research.models.self_play.reward_functions.AbstractRewardFunction
        :type advantage_fn: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        :type process_pool: diplomacy_research.utils.executor.ProcessPoolExecutor
        :type games_queue: multiprocessing.Queue
        :type transitions_queue: multiprocessing.Queue
    """
    # pylint: disable=too-many-arguments
    memory_buffer = MemoryBuffer(cluster_config, hparams)
    nb_cores = multiprocessing.cpu_count()
    futures = []

    nb_pending_transitions = 0                      # For throttling if there are enough transitions for the learner
    nb_rl_agents = get_nb_rl_agents(hparams['mode'])

    # 1) Finding the right function to create player
    get_players_callable = {'supervised': get_train_supervised_players,
                            'self-play': get_train_self_play_players,
                            'staggered': get_train_staggered_players}[hparams['mode']]

    # Generating an infinite number of games
    while True:
        nb_new_transitions = 0

        # 1) Detecting the number of pending transactions to throttle if necessary
        while not transitions_queue.empty():
            item = transitions_queue.get()
            if item == NEW_VERSION:
                nb_pending_transitions = 0
                nb_cores = multiprocessing.cpu_count()
            else:
                nb_pending_transitions += nb_rl_agents * item / NB_POWERS
                nb_new_transitions += nb_rl_agents * item / NB_POWERS

        futures = [fut for fut in futures if not fut.done()]
        nb_games_being_generated = len(futures)

        # Finding the number of games to generate
        nb_new_games = nb_cores - nb_games_being_generated
        if nb_new_games <= 0:
            continue

        # 2) Generating the get_player_kwargs
        get_players_kwargs = [{'adapter_ctor': adapter_ctor,
                               'dataset_builder_ctor': dataset_builder_ctor,
                               'tf_serving_port': get_tf_serving_port(cluster_config, serving_id=0),
                               'cluster_config': cluster_config,
                               'hparams': hparams}] * nb_new_games

        # 3) Generating gen_trajectory_kwargs
        gen_trajectory_kwargs = []
        for _ in range(nb_new_games):
            gen_trajectory_kwargs += [{'hparams': hparams,
                                       'reward_fn': reward_fn,
                                       'advantage_fn': advantage_fn,
                                       'power_assignments': hparams.get('power', '') or random.choice(ALL_POWERS),
                                       'set_player_seed': bool(hparams['dropout_rate']),
                                       'update_interval': hparams['update_interval'],
                                       'update_queue': games_queue}]

        # 4) Adding initial states if required
        if hparams['start_strategy'] == 'uniform':
            initial_states_proto = get_uniform_initial_states(memory_buffer, nb_new_games)
            for game_ix, initial_state_proto in enumerate(initial_states_proto):
                gen_trajectory_kwargs[game_ix]['initial_state_bytes'] = proto_to_bytes(initial_state_proto)

        elif hparams['start_strategy'] == 'backplay':
            winning_power_names = []
            for kwargs in gen_trajectory_kwargs:
                if isinstance(kwargs['power_assignments'], list):
                    winning_power_names += [kwargs['power_assignments'][0]]
                else:
                    winning_power_names += [kwargs['power_assignments']]
            version_id = general_ops.get_version_id(memory_buffer)
            initial_states_proto = get_backplay_initial_states(memory_buffer, winning_power_names, version_id)
            for game_ix, initial_state_proto in enumerate(initial_states_proto):
                gen_trajectory_kwargs[game_ix]['initial_state_bytes'] = proto_to_bytes(initial_state_proto)

        # 6) Launching jobs using current pool
        tasks = []
        for player_kwargs, trajectory_kwargs in zip(get_players_kwargs, gen_trajectory_kwargs):
            tasks += [{'get_players_callable': get_players_callable,
                       'get_players_kwargs': player_kwargs,
                       'generate_trajectory_kwargs': trajectory_kwargs,
                       'queue': games_queue}]
        futures += [process_pool.submit(start_game_process, kwargs) for kwargs in tasks]

def start_aggregator(hparams, cluster_config, games_queue, transitions_queue, display_status=True):
    """ Games Aggregator - Displays status every minute and saves games to disk and to memory buffer
        :param hparams: A dictionary of hyper-parameters
        :param cluster_config: The cluster configuration to use for distributed training
        :param games_queue: Inbound queue to receive games from the process pool.
        :param transitions_queue: Outbound queue to send transitions and version updates to the generator for throttling
        :param display_status: Boolean that indicates to display the status on stdout.
        :return: Nothing
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        :type games_queue: multiprocessing.Queue
        :type transitions_queue: multiprocessing.Queue
    """
    # pylint: disable=too-many-nested-blocks
    memory_buffer = MemoryBuffer(cluster_config, hparams)
    buffer_file = None
    current_config = None               # type: AggregatorConfig
    new_config = None
    nb_left, nb_total, nb_completed = 0, 0, 0
    starting_time = time.time()
    last_status_time = time.time()
    queue_is_done = False
    last_version = -1

    # Looping forever - New configs will be sent on the queue
    while not queue_is_done:
        nb_full_games = 0

        # Dequeuing items from the queue
        saved_games_bytes = []
        while True:
            try:
                item = games_queue.get(False)
                if item is None:
                    queue_is_done = True
                    break
                elif isinstance(item, AggregatorConfig):
                    new_config = item
                    break
                else:
                    is_full_game, nb_transitions, saved_game_bytes = item
                    saved_games_bytes += [saved_game_bytes]

                    if is_full_game:
                        nb_full_games += 1

                    # On a new version, we send 'new.version' on the transition queue
                    # We also send the nb of transitions, so the generator can throttle when there is enough transitions
                    # for the current version
                    if transitions_queue and nb_transitions:
                        current_version = general_ops.get_version_id(memory_buffer)
                        if current_version != last_version:
                            transitions_queue.put_nowait(NEW_VERSION)
                            last_version = current_version
                        transitions_queue.put_nowait(nb_transitions)
            except EmptyQueueException:
                break

        # Processing current games
        if saved_games_bytes:
            if current_config.save_to_buffer:
                save_games(memory_buffer, saved_games_bytes=saved_games_bytes)
            for saved_game_bytes in saved_games_bytes:
                if buffer_file:
                    write_bytes_to_file(buffer_file, saved_game_bytes)
                if nb_left > 0:
                    nb_left -= 1
                nb_completed += 1

        # Printing status
        # Case 1 - Generate an infinite number of games
        if nb_total == -1 and time.time() - last_status_time >= 60:
            last_status_time = time.time()
            elapsed_time = last_status_time - starting_time
            games_per_day = 24 * 3600. * nb_full_games / max(1., elapsed_time)
            if display_status:
                LOGGER.info('Current status - Games/day: %.2f', games_per_day)

        # Case 2 - Generate a finite number of games
        elif nb_left > 0 and time.time() - last_status_time >= 60:
            last_status_time = time.time()
            elapsed_time = last_status_time - starting_time
            progress = 100. * (nb_total - nb_left) / max(1, nb_total)
            eta = datetime.timedelta(seconds=int(elapsed_time / max(1, nb_completed * nb_left)))
            if display_status:
                LOGGER.info('Current status - %.2f%% (%d/%d) - ETA: %s', progress, nb_total - nb_left, nb_total, eta)

        # Done current batch
        if nb_total and not nb_left:
            progress = 100. * (nb_total - nb_left) / max(1, nb_total)
            if display_status:
                LOGGER.info('Done generating games - Progress %.2f%% (%d/%d)', progress, nb_total - nb_left, nb_total)
            nb_left, nb_total, nb_completed = 0, 0, 0
            if buffer_file:
                buffer_file.close()
                buffer_file = None

        # Setting new config
        if new_config:
            starting_time = time.time()
            nb_left, nb_total, nb_completed = new_config.nb_left, new_config.nb_total, 0
            if buffer_file:
                buffer_file.close()
                buffer_file = None
            if new_config.file_path:
                buffer_file = open(new_config.file_path, 'ab')
            current_config = new_config
            if nb_total > 0:
                progress = 100. * (nb_total - nb_left) / max(1, nb_total)
                if display_status:
                    LOGGER.info('Generating games - Progress %.2f%% (%d/%d)', progress, nb_total - nb_left, nb_total)

        # Sleeping
        if not saved_games_bytes and new_config is None:
            time.sleep(0.25)
        new_config = None

def start_training_process(trainer, block=False):
    """ Starts a process that will generate an infinite number of training games
        :param trainer: A reinforcement learning trainer instance.
        :param block: Boolean that indicates that we want to block until the process is completed
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    # Starting aggregator and game generator
    if not trainer.aggregator['train']:

        # The to_aggregator queue is used to send games from the process pool to the aggregator
        # The to_generator queue is used to send transitions from the aggregator to the generator for throttling
        manager = multiprocessing.Manager()
        queue_to_aggregator = manager.Queue()
        queue_to_generator = manager.Queue()

        # Putting configuration for aggregator -- -1 = infinite games
        queue_to_aggregator.put_nowait(AggregatorConfig(nb_left=-1,
                                                        nb_total=-1,
                                                        file_path=None,
                                                        save_to_buffer=True))

        # Aggregator - Using separate process
        trainer.aggregator['train'] = SPAWN_CONTEXT.Process(target=start_aggregator,
                                                            kwargs={'hparams': trainer.hparams,
                                                                    'cluster_config': trainer.cluster_config,
                                                                    'games_queue': queue_to_aggregator,
                                                                    'transitions_queue': queue_to_generator,
                                                                    'display_status': bool(trainer.cluster_config)})
        trainer.aggregator['train'].start()

        # Generator - Using thread
        trainer.train_thread = Thread(target=start_game_generator,
                                      kwargs={'adapter_ctor': trainer.adapter_constructor,
                                              'dataset_builder_ctor': trainer.dataset_builder_constructor,
                                              'reward_fn': trainer.reward_fn,
                                              'advantage_fn': trainer.advantage_fn,
                                              'hparams': trainer.hparams,
                                              'cluster_config': trainer.cluster_config,
                                              'process_pool': trainer.process_pool,
                                              'games_queue': queue_to_aggregator,
                                              'transitions_queue': queue_to_generator})
        trainer.train_thread.start()

    # Blocking until thread completes
    if block:
        trainer.train_thread.join()

def get_replay_samples(trainer):
    """ Retrives a series of replay samples (to use for learning)
        :param trainer: A reinforcement learning trainer instance.
        :return: A list of `ReplaySample`.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    # Computing the number of samples - Rounding them to the nearest batch size
    perc_from_replay = 0.
    if trainer.algorithm_constructor.can_do_experience_replay:
        perc_from_replay = max(0., min(1., trainer.flags.experience_replay))
    nb_samples = perc_from_replay * trainer.flags.nb_transitions_per_update
    nb_samples = int(trainer.flags.batch_size * round(nb_samples / trainer.flags.batch_size))

    if not nb_samples:
        return []
    return priority_replay.get_replay_samples(trainer.memory_buffer, nb_samples)
