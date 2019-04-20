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
""" Reinforcement Learning - Common Ops
    - Class responsible for implementing build operations (common among standalone and distributed training)
"""
import glob
import logging
import os
import shutil
import time
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.self_play.algorithms.base_algorithm import VERSION_STEP
from diplomacy_research.models.state_space import ALL_POWERS
from diplomacy_research.models.training.memory_buffer import general_ops as buffer_gen_ops
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.checkpoint import build_saved_model
from diplomacy_research.utils.cluster import kill_processes_using_port, is_port_opened
from diplomacy_research.utils.proto import write_proto_to_file, read_next_proto

# Constants
LOGGER = logging.getLogger(__name__)


def build_summaries(trainer):
    """ Builds the fields required to log stats in TensorBoard
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, get_placeholder
    escape = lambda string: string.replace('[', '.').replace(']', '.')
    evaluation_tags = []
    for power_name in ALL_POWERS + ['ALL']:
        evaluation_tags += ['%s/%s' % (key, power_name) for key in
                            ['rew_avg', 'sc_avg', 'win_by_sc', 'most_sc', 'rank', 'nb_years']]
        evaluation_tags += ['%s/%s' % (key, power_name) for key in
                            ['year_at_%d_sc' % nb_sc for nb_sc in range(4, 20, 2)]]
        evaluation_tags += ['%s/%s' % (key, power_name) for key in
                            [reward_fn.name for reward_fn in trainer.eval_reward_fns]]
        evaluation_tags += ['%s/%s' % (key, power_name) for key in
                            ['done_engine', 'done_auto_draw', 'done_thrashing', 'done_phase_limit']]
    evaluation_tags += trainer.algorithm.get_evaluation_tags()
    evaluation_tags += ['version/player', 'version/opponent']
    evaluation_tags += ['nb_rl_agents', 'nb_steps', 'nb_games', 'versions_per_day', 'mem_usage_mb']

    dtypes = {'version/player': tf.int64,
              'version/opponent': tf.int64,
              'nb_steps': tf.int64,
              'nb_games': tf.int64}

    # Creating placeholders and summaries
    for tag_name in evaluation_tags:
        if tag_name not in trainer.placeholders:
            trainer.placeholders[tag_name] = get_placeholder(escape(tag_name),
                                                             dtype=dtypes.get(tag_name, tf.float32),
                                                             shape=(),
                                                             for_summary=True)
            trainer.summaries[tag_name] = tf.summary.scalar(escape(tag_name), trainer.placeholders[tag_name])

    # Creating merge op
    trainer.merge_op = tf.summary.merge(list(trainer.summaries.values()))

    # Creating writers
    os.makedirs(os.path.join(trainer.flags.save_dir, 'summary'), exist_ok=True)
    trainer.writers['train'] = tf.summary.FileWriter(os.path.join(trainer.flags.save_dir, 'summary', 'rl_train'))
    trainer.writers['eval'] = tf.summary.FileWriter(os.path.join(trainer.flags.save_dir, 'summary', 'rl_eval'))

def build_model_and_dataset(trainer):
    """ Builds the policy, and value model and the dataset
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    trainer.queue_dataset = QueueDataset(batch_size=trainer.flags.batch_size,
                                         dataset_builder=trainer.dataset_builder,
                                         cluster_config=trainer.cluster_config)

    # Policy Model
    trainer.model = trainer.policy_constructor(trainer.queue_dataset, trainer.hparams)

    # Value Model
    if trainer.value_constructor:
        trainer.model = trainer.value_constructor(trainer.model, trainer.queue_dataset, trainer.hparams)

    # Draw Model
    if trainer.draw_constructor:
        trainer.model = trainer.draw_constructor(trainer.model, trainer.queue_dataset, trainer.hparams)

    # Finalizing and validating
    trainer.model.finalize_build()
    trainer.model.validate()

    # Adding hooks
    session_hook = trainer.queue_dataset.make_session_run_hook()
    if session_hook is not None:
        trainer.hooks += [session_hook]

def build_train_server(trainer):
    """ Builds the Tensorflow tf.train.Server
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if not trainer.cluster_config:
        return

    from diplomacy_research.utils.tensorflow import tf
    task_address = trainer.cluster_config.cluster_spec[trainer.cluster_config.job_name][trainer.cluster_config.task_id]

    # Making port is not used by another process
    task_port = int(task_address.split(':')[1])
    LOGGER.info('Killing any processes that have port %d opened.', task_port)
    kill_processes_using_port(task_port)

    # Starting server
    LOGGER.info('Starting server with task id %d - Address: %s', trainer.cluster_config.task_id, task_address)
    LOGGER.info('Checking if port %d is already opened: %s', task_port, str(is_port_opened(task_port)))
    trainer.server = tf.train.Server(tf.train.ClusterSpec(trainer.cluster_config.cluster_spec),
                                     job_name=trainer.cluster_config.job_name,
                                     task_index=trainer.cluster_config.task_id,
                                     protocol=trainer.cluster_config.protocol)
    LOGGER.info('Server successfully started. Trying to contact other nodes...')

def build_algorithm(trainer):
    """ Builds the RL algorithm
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if not trainer.queue_dataset or not trainer.model:
        raise RuntimeError('You must build the model and the dataset before building the RL algorithm')
    trainer.algorithm = trainer.algorithm_constructor(trainer.queue_dataset, trainer.model, trainer.hparams)

def build_config_proto(trainer):
    """ Builds the session config proto
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    gpu_options = tf.GPUOptions()
    if trainer.flags.allow_gpu_growth:
        LOGGER.info('"allow_gpu_growth" flag set. GPU memory will not be pre-allocated and will grow as needed.')
        gpu_options = tf.GPUOptions(allow_growth=True)
    trainer.config_proto = {'allow_soft_placement': True, 'gpu_options': gpu_options}
    if trainer.cluster_config and trainer.cluster_config.job_name == 'learner':
        trainer.config_proto['device_filters'] = ['/job:ps', '/job:learner']
    if trainer.flags.use_xla:
        LOGGER.info('Using XLA to compile the graph.')
        graph_options = tf.GraphOptions()
        graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1                                     # pylint: disable=no-member
        trainer.config_proto['graph_options'] = graph_options

def build_restore_saver(trainer):
    """ Builds the restore saver to restore from a supervised model checkpoint
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return
    if os.path.exists(os.path.join(trainer.flags.save_dir, 'checkpoint')):
        checkpoint_file = 'checkpoint'
    elif os.path.exists(os.path.join(trainer.flags.save_dir, 'checkpoint.supervised')):
        checkpoint_file = 'checkpoint.supervised'
    else:
        LOGGER.warning('No checkpoints were detected on disk. Unable to load weights')
        return

    # Loading TensorFlow
    from diplomacy_research.utils.tensorflow import tf, list_variables

    # Detecting if we have a supervised or a RL checkpoint
    # Rewriting the checkpoint path to save_dir, so we can copy a supervised folder without modifying the paths.
    # If 'version_step' is saved in checkpoint, we have a RL checkpoint, so we can skip this method
    ckpt_path = tf.train.get_checkpoint_state(trainer.flags.save_dir, checkpoint_file).model_checkpoint_path            # pylint: disable=no-member
    ckpt_path = os.path.join(trainer.flags.save_dir, ckpt_path.split('/')[-1])
    if [1 for var, _ in list_variables(ckpt_path) if VERSION_STEP in var]:
        return

    # Overwriting checkpoint on disk
    with open(os.path.join(trainer.flags.save_dir, checkpoint_file), 'w') as file:
        file.write('model_checkpoint_path: "%s"\n' % ckpt_path)
        file.write('all_model_checkpoint_paths: "%s"\n' % ckpt_path)

    # Listing vars in model, and vars in checkpoint
    global_vars = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    model_vars = {var.name: tuple(var.shape.as_list()) for var in global_vars}
    ckpt_vars = {'%s:0' % var: tuple(shape) for var, shape in list_variables(ckpt_path)
                 if ('/gradients/' not in var
                     and '/Adam' not in var
                     and 'barrier' not in var
                     and 'beta1_power' not in var
                     and 'beta2_power' not in var)}

    # Building a set of tuples (var_name, shape)
    model_vars_set = {(var_name, var_shape) for var_name, var_shape in model_vars.items()}
    ckpt_vars_set = {(var_name, var_shape) for var_name, var_shape in ckpt_vars.items()}

    # Making sure all elements in graph are also in checkpoint, otherwise printing errors
    if model_vars_set - ckpt_vars_set:
        LOGGER.error('-' * 80)
        LOGGER.error('*** Unable to restore some variables from checkpoint. ***')
        LOGGER.error('')

        # Elements in graph, but not in checkpoint
        for (var_name, model_shape) in sorted(model_vars_set - ckpt_vars_set):
            if var_name not in ckpt_vars:
                LOGGER.error('Variable "%s" (Shape: %s) is in graph, but not in checkpoint.', var_name, model_shape)
            else:
                LOGGER.error('Variable "%s" has shape %s in graph, and shape %s in checkpoint.',
                             var_name, model_shape, ckpt_vars.get(var_name))

        # Elements in checkpoint, but not in graph
        for (var_name, ckpt_shape) in sorted(ckpt_vars_set - model_vars_set):
            if var_name not in model_vars:
                LOGGER.error('Variable "%s" (Shape: %s) is in checkpoint, but not in graph.', var_name, ckpt_shape)
        LOGGER.error('-' * 80)

    # Loading only variables common in both. (& = intersection)
    common_vars = [name_shape[0] for name_shape in model_vars_set & ckpt_vars_set]
    vars_to_load = [var for var in global_vars if var.name in common_vars]
    trainer.restore_saver = tf.train.Saver(var_list=vars_to_load, restore_sequentially=True)
    LOGGER.info('Found %d variables to load from the supervised checkpoint.', len(vars_to_load))

    # Renaming checkpoint to make sure the RL algo doesn't also try to load it.
    if checkpoint_file == 'checkpoint':
        shutil.move(os.path.join(trainer.flags.save_dir, 'checkpoint'),
                    os.path.join(trainer.flags.save_dir, 'checkpoint.supervised'))

def build_hooks(trainer):
    """ Adds the hooks required for training
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import ReinforcementStartTrainingHook, RestoreVariableHook
    trainer.chief_only_hooks += [RestoreVariableHook(trainer.restore_saver,
                                                     trainer.flags.save_dir,
                                                     'checkpoint.supervised'),
                                 ReinforcementStartTrainingHook(trainer)]

def create_adapter(trainer):
    """ Creates an adapter (for the learner)
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    trainer.adapter = trainer.adapter_constructor(trainer.queue_dataset, graph=tf.get_default_graph())

def create_advantage(trainer):
    """ Creates the advantage function
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    trainer.advantage_fn = \
        trainer.algorithm_constructor.create_advantage_function(trainer.hparams,
                                                                gamma=trainer.flags.gamma,
                                                                penalty_per_phase=trainer.flags.penalty_per_phase)

def save_version_model(trainer, version_id=None, wait_for_grpc=False):
    """ Builds a saved model for the given graph for inference
        :param trainer: A reinforcement learning trainer instance.
        :param version_id: Optional. Integer. The version id of the graph to save. Defaults to the current version.
        :param wait_for_grpc: If true, waits for the target to be loaded on the localhost TF Serving before returning.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    from diplomacy_research.models.training.reinforcement.serving import wait_for_version
    from diplomacy_research.utils.tensorflow import tf

    # Can only save model in standalone mode or while chief in distributed mode
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return

    # Creating output directory
    if version_id is None:
        version_id = get_version_id(trainer)
    output_dir = get_serving_directory(trainer, 'player')
    os.makedirs(output_dir, exist_ok=True)

    # Saving model
    proto_fields = trainer.dataset_builder.get_proto_fields()
    trainer.run_func_without_hooks(trainer.session,
                                   lambda _sess: build_saved_model(saved_model_dir=output_dir,
                                                                   version_id=version_id,
                                                                   signature=trainer.signature,
                                                                   proto_fields=proto_fields,
                                                                   graph=tf.get_default_graph(),
                                                                   session=_sess))

    # Saving opponent
    # For version = 0, or if we reached the numbers of version to switch version (in "staggered" mode)
    if version_id == 0 or (trainer.flags.mode == 'staggered'
                           and version_id % trainer.flags.staggered_versions == 0):
        create_opponent_from_player(trainer, version_id)

    # Saving checkpoint - Once every 10 mins by default
    if (time.time() - trainer.checkpoint_every) >= trainer.last_checkpoint_time:
        trainer.last_checkpoint_time = int(time.time())
        checkpoint_save_path = os.path.join(trainer.flags.save_dir, 'rl_model.ckpt')
        trainer.run_func_without_hooks(trainer.session,
                                       lambda _sess: trainer.saver.save(_sess,
                                                                        save_path=checkpoint_save_path,
                                                                        global_step=version_id))

    # Waiting for gRPC
    if wait_for_grpc:
        wait_for_version(trainer, model_name='player', version_id=version_id, timeout=10)

def create_opponent_from_player(trainer, version_id):
    """ Creates an opponent by copying a specific player version id
        :param trainer: A reinforcement learning trainer instance.
        :param version_id: The version id to copy from the player directory.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    output_dir = get_serving_directory(trainer, 'opponent')
    os.makedirs(output_dir, exist_ok=True)

    # Copying into a temp directory and then renaming to avoid serving trying to load an empty folder
    src_dir = get_version_directory(trainer, 'player', version_id)
    dest_dir = get_version_directory(trainer, 'opponent', version_id)
    if not os.path.exists(src_dir):
        LOGGER.warning('Unable to find the directory %s to copy the opponent model', src_dir)
    shutil.copytree(src_dir, dest_dir + '__')
    shutil.move(dest_dir + '__', dest_dir)

def extract_games_and_power_phases(trainer, learned_games_proto, replay_samples, partial_games_only=False):
    """ Extracts games_proto and (power, phase_ix) to use for learning
        :param trainer: A reinforcement learning trainer instance.
        :param learned_games_proto: A list of `SavedGame` instances from the RL agents.
        :param replay_samples: A list of `ReplaySamples` from the memory buffer.
        :param partial_games_only: Boolean that indicates to only extract info from partial games.
        :return: A tuple consisting of
                    1) A list of `SavedGame` instances to use for learning
                    2) A list dictionary (one per game) with power_name as key and list of phase_ix as value
                    e.g. [Game1, Game2, ...]
                         [{'AUSTRIA': [0, 1, 2], ...}, {'AUSTRIA': [1, 2, 3], ...}]
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    saved_games_proto, power_phases_ix = [], []
    for saved_game_proto in learned_games_proto:
        if not saved_game_proto.assigned_powers:
            continue
        if partial_games_only and not saved_game_proto.is_partial_game:
            continue
        saved_games_proto += [saved_game_proto]
        power_phases_ix += [trainer.algorithm.get_power_phases_ix(saved_game_proto,
                                                                  get_nb_rl_agents(trainer.flags.mode))]
    for replay_sample in replay_samples:
        saved_games_proto += [replay_sample.saved_game_proto]
        power_phases_ix += [replay_sample.power_phases_ix]
    return saved_games_proto, power_phases_ix

def complete_epoch(trainer):
    """ Cleanups at the end of an epoch
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return

    # Cleaning up
    delete_obsolete_versions(trainer)
    delete_temp_files(trainer, 'train')

def load_games_from_folder(trainer, target_dir, pattern, minimum=0, timeout=None, compressed=True):
    """ Loads games from disk
        :param trainer: A reinforcement learning trainer instance.
        :param target_dir: The directory where to load the games from.
        :param pattern: The file pattern to use for finding the protobuf file(s).
        :param minimum: The minimum number of files expected to match the pattern.
        :param timeout: Optional. The timeout (in seconds) to wait for the minimum nb of files to appear.
        :param compressed: Boolean. Indicates if the content is compressed
        :return: A list of saved game proto
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    saved_games_proto = []
    start_time = int(time.time())
    filenames = glob.glob(os.path.join(target_dir, pattern))

    # Waiting until timeout
    if len(filenames) < minimum and timeout:
        while int(time.time()) - start_time < timeout and len(filenames) < minimum:
            filenames = glob.glob(os.path.join(target_dir, pattern))
            time.sleep(1.)

    # Minimum number of files not found
    if len(filenames) < minimum:
        LOGGER.warning('Expected to find %d files on disk using pattern %s. Only found %d.',
                       minimum, pattern, len(filenames))

    # Parsing files
    for full_filename in filenames:
        filename = os.path.basename(full_filename)
        with trainer.memory_buffer.lock('lock.file.%s' % filename, timeout=120):
            saved_games_proto += load_games_from_file(os.path.join(target_dir, filename), compressed=compressed)
    return saved_games_proto

def load_games_from_file(file_path, compressed=False):
    """ Trying to load games stored on disk to resume training
        :param file_path: The path to the protocol buffer file where games can be stored
        :param compressed: Boolean. Indicates if the content is compressed
        :return: A list of `game.SavedGame` proto.
    """
    if not os.path.exists(file_path):
        return []

    # Loading file and getting games
    saved_games_proto = []
    with open(file_path, 'rb') as file:
        while True:
            saved_game_proto = read_next_proto(SavedGameProto, file, compressed)
            if saved_game_proto is None:
                break
            saved_games_proto += [saved_game_proto]
    return saved_games_proto

def save_games_to_folder(trainer, saved_games_proto, target_dir, filename):
    """ Saves games to disk
        :param trainer: A reinforcement learning trainer instance.
        :param saved_games_proto: A list of diplomacy_proto.game.SavedGame
        :param target_dir: The directory where to save the games.
        :param filename: The filename to use for the protobuf file.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Saving to temp path, then renaming
    temp_path = os.path.join(target_dir, '__' + filename)
    output_path = os.path.join(target_dir, filename)

    with trainer.memory_buffer.lock('lock.file.%s' % filename, timeout=120):
        with open(temp_path, 'ab') as game_file:
            for saved_game_proto in saved_games_proto:
                if saved_game_proto.is_partial_game:
                    continue
                write_proto_to_file(game_file, saved_game_proto, compressed=True)
            shutil.move(temp_path, output_path)

def get_versions_on_disk(trainer, model_name):
    """ Find the versions saved on disk for the given model name.
        :param trainer: A reinforcement learning trainer instance.
        :param model_name: The model name to use to find versions on disk (e.g. 'player', 'opponent')
        :return: A list of versions found for this model name.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    return list(sorted(next(os.walk(get_serving_directory(trainer, model_name)))[1]))

def delete_obsolete_versions(trainer, min_to_keep=5):
    """ Deleting obsolete player versions on disk to save space
        :param trainer: A reinforcement learning trainer instance.
        :param min_to_keep: The minimum number of versions to keep
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    player_dir = os.path.join(trainer.flags.save_dir, 'serving', 'player')
    player_versions = get_versions_on_disk(trainer, 'player')

    # Keeping the first 2 version and the last 'min_to_keep'
    versions_to_keep = {version for version in player_versions[:2]}
    versions_to_keep |= {version for version in player_versions[-min_to_keep:]}

    # Finding extra versions
    extra_versions = {version for version in player_versions
                      if int(version) % trainer.flags.eval_every != 1 and version not in versions_to_keep}

    # Deleting the versions
    for version in extra_versions:
        shutil.rmtree(os.path.join(player_dir, version), ignore_errors=True)

def delete_temp_files(trainer, current_mode):
    """ Deletes temp files on disk
        :param trainer: A reinforcement learning trainer instance.
        :param current_mode: The current mode (either 'train' or 'eval')
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if current_mode not in ('train', 'eval'):
        return
    temp_pattern = '.temp_%s*.pb' % current_mode
    temp_pattern = os.path.join(trainer.flags.save_dir, 'serving', temp_pattern)

    # Deleting the temp files if present
    for file_path in glob.glob(temp_pattern):
        if os.path.exists(file_path):
            os.unlink(file_path)

def get_nb_rl_agents(mode):
    """ Computes the number of RL agents in each game
        :param mode: The number of agents for the current mode (either flags.mode or flags.eval_mode)
        :param hparams: The model's hyper-parameters
    """
    return {'supervised': 1,                        # --- train modes ---
            'self-play': len(ALL_POWERS),
            'staggered': 1,
            'supervised-0': 1,                      # --- eval modes ---
            'supervised-1': 1}[mode]

def get_version_id(trainer):
    """ Returns the current iteration number
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if trainer.cluster_config and trainer.cluster_config.job_name != 'learner':
        return buffer_gen_ops.get_version_id(trainer.memory_buffer)
    return int(trainer.session.run(trainer.algorithm.version_step))

def get_opponent_version_id(trainer):
    """ Returns the version id used by the opponent (by looking on disk)
        :param trainer: A reinforcement learning trainer instance.
        :return: The version being used by the opponent (-1 when N/A (e.g. a rule-based model))
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if trainer.flags.mode == 'self-play':
        return get_version_id(trainer)
    if trainer.flags.mode == 'staggered' and trainer.flags.staggered_versions == 1:
        return get_version_id(trainer)
    if trainer.flags.mode == 'supervised':
        return 0

    # Returning -1 if no models on disk
    serving_dir = get_serving_directory(trainer, 'opponent')
    if not os.path.exists(serving_dir):
        return -1

    # Finding the max version
    for path in sorted(glob.glob(os.path.join(serving_dir, '*')), reverse=True):
        version = path.split('/')[-1]
        if version.isdigit():
            return int(version)
    return -1

def get_serving_directory(trainer, target_name):
    """ Returns the directory where all serving models should be saved
        :param trainer: A reinforcement learning trainer instance.
        :param target_name: The name of the model (e.g. 'player', 'opponent')
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    return os.path.join(trainer.flags.save_dir, 'serving', target_name)

def get_version_directory(trainer, target_name, version_id=None):
    """ Returns the directory where the TF Serving version should be saved.
        :param trainer: A reinforcement learning trainer instance.
        :param target_name: The name of the model (e.g. 'player', 'opponent')
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if version_id is None:
        version_id = get_version_id(trainer)
    return os.path.join(trainer.flags.save_dir, 'serving', target_name, '%09d' % version_id)
