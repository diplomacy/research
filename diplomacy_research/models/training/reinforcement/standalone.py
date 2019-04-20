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
""" Reinforcement - Standalone training
    - Class responsible for training a model in a non-distributed setting
"""
import logging
import time
from tornado import gen
from diplomacy_research.models.training.memory_buffer.expert_games import load_expert_games
from diplomacy_research.models.training.memory_buffer.general_ops import set_version_id
from diplomacy_research.models.training.memory_buffer.online_games import get_online_games, mark_games_as_processed
from diplomacy_research.models.training.memory_buffer.memory_buffer import start_redis_server
from diplomacy_research.models.training.reinforcement.common import build_model_and_dataset, build_restore_saver, \
    build_algorithm, build_summaries, build_config_proto, build_hooks, create_adapter, create_advantage, \
    extract_games_and_power_phases, save_version_model, save_games_to_folder, complete_epoch, get_version_id, \
    get_version_directory
from diplomacy_research.models.training.reinforcement.generation import start_training_process, get_replay_samples
from diplomacy_research.models.training.reinforcement.memory_buffer import build_memory_buffer, update_priorities
from diplomacy_research.models.training.reinforcement.serving import start_tf_serving_server, check_opening_orders, \
    get_training_config
from diplomacy_research.models.training.reinforcement.statistics import reset_stats, update_stats, compile_stats, \
    save_stats, display_progress
from diplomacy_research.settings import SESSION_RUN_TIMEOUT

# Constants
LOGGER = logging.getLogger(__name__)

@gen.coroutine
def start_standalone_training(trainer):
    """ Starts training in standalone mode.
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, tf_debug

    # Builds the model and the dataset
    build_model_and_dataset(trainer)
    build_restore_saver(trainer)
    build_algorithm(trainer)
    build_summaries(trainer)
    build_config_proto(trainer)
    build_hooks(trainer)
    create_adapter(trainer)
    create_advantage(trainer)

    # Start the memory buffer and load expert games
    start_redis_server(trainer)
    build_memory_buffer(trainer)
    load_expert_games(trainer.memory_buffer)

    # Creating session and start training
    try:
        trainer.saver = tf.train.Saver(max_to_keep=5, restore_sequentially=True, pad_step_number=True)
        with tf.train.MonitoredTrainingSession(checkpoint_dir=trainer.flags.save_dir,
                                               scaffold=tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                                                          saver=trainer.saver),
                                               log_step_count_steps=0,
                                               hooks=trainer.hooks,
                                               chief_only_hooks=trainer.chief_only_hooks,
                                               config=tf.ConfigProto(**trainer.config_proto),
                                               save_checkpoint_secs=0,
                                               save_checkpoint_steps=0,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as sess:

            # Wrapping in a session debugger
            if trainer.flags.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Saving session
            trainer.session = sess
            trainer.model.sess = sess
            LOGGER.info('Session has been successfully created.')

            # Initializing adapter, algorithm, and stats
            trainer.run_func_without_hooks(sess, trainer.adapter.initialize)
            yield trainer.algorithm.init()
            reset_stats(trainer)

            # Starting training serving servers
            start_tf_serving_server(trainer,
                                    force_cpu=False,
                                    serving_id=0,
                                    config=get_training_config(trainer))

            # Querying the model to make sure to check if it has been trained
            check_opening_orders(trainer)

            # Launching the training process
            start_training_process(trainer, block=False)

            # Training forever
            while True:
                yield run_training_epoch(trainer)

    # CTRL-C closes the session to force a checkpoint
    # Tensorflow will throw a Runtime Error because the session is already closed.
    except RuntimeError as error:
        if trainer.model.sess is not None and trainer.model.sess._sess is not None:                                     # pylint: disable=protected-access
            raise error

@gen.coroutine
def run_training_epoch(trainer):
    """ Runs a training epoch
        :param trainer: A reinforcement trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    print('\n=========== TRAINING ===========')

    # Variables
    epoch_results = {}
    nb_missing_transitions = -1
    partial_games_only = bool(trainer.flags.update_interval)

    # Clearing transition buffers
    yield trainer.algorithm.clear_buffers()

    # Sampling new replay samples from the memory buffer
    if trainer.algorithm_constructor.can_do_experience_replay:
        trainer.replay_samples = get_replay_samples(trainer)

    # Getting games from the memory buffer
    learned_games_proto, game_ids = get_online_games(trainer.memory_buffer)
    saved_games_proto, power_phases_ix = extract_games_and_power_phases(trainer,
                                                                        learned_games_proto,
                                                                        trainer.replay_samples,
                                                                        partial_games_only=partial_games_only)
    yield trainer.algorithm.learn(saved_games_proto, power_phases_ix, trainer.advantage_fn)

    # Waiting for additional games - We don't have enough transitions to proceed
    while len(trainer.algorithm.transition_buffer) < trainer.flags.nb_transitions_per_update:
        if nb_missing_transitions != trainer.flags.nb_transitions_per_update - len(trainer.algorithm.transition_buffer):
            nb_missing_transitions = trainer.flags.nb_transitions_per_update - len(trainer.algorithm.transition_buffer)
            LOGGER.info('Waiting for additional games. Missing %d transitions.', nb_missing_transitions)
        yield gen.sleep(1.)

        # Getting additional games
        new_games_proto, new_game_ids = get_online_games(trainer.memory_buffer, excluding=game_ids)
        learned_games_proto += new_games_proto
        game_ids += new_game_ids

        # Learning from those games
        saved_games_proto, power_phases_ix = extract_games_and_power_phases(trainer,
                                                                            new_games_proto,
                                                                            [],
                                                                            partial_games_only=partial_games_only)
        yield trainer.algorithm.learn(saved_games_proto, power_phases_ix, trainer.advantage_fn)

    # Proceeding
    LOGGER.info('[Train] Retrieved %d games from Redis', len(learned_games_proto))
    try:
        epoch_results = yield trainer.algorithm.update(trainer.memory_buffer)
    except (TimeoutError, tf.errors.DeadlineExceededError):
        LOGGER.warning('learn/update took more than %d ms to run. Timeout error.', SESSION_RUN_TIMEOUT)

    # Updating priorities
    yield update_priorities(trainer, learned_games_proto, trainer.replay_samples)
    trainer.replay_samples = []

    # Saving to disk
    new_version_id = get_version_id(trainer)
    save_version_model(trainer, wait_for_grpc=True)
    set_version_id(trainer.memory_buffer, new_version_id)

    # Marking games as processed
    mark_games_as_processed(trainer.memory_buffer, game_ids)

    # Compiling stats (every 60s by default)
    if (time.time() - trainer.stats_every) >= trainer.last_stats_time:
        update_stats(trainer, 'train', learned_games_proto, epoch_results=epoch_results)
        compile_stats(trainer, 'train')
        save_stats(trainer)
        save_games_to_folder(trainer=trainer,
                             saved_games_proto=learned_games_proto,
                             target_dir=get_version_directory(trainer, 'player'),
                             filename='games_learned.pbz')
        trainer.last_stats_time = int(time.time())

    # Displaying progress and completing
    display_progress(trainer)
    complete_epoch(trainer)
