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
""" Reinforcement Learning - Serving
    - Class responsible for for interacting with the TensorFlow Serving server
"""
import glob
import logging
import math
import multiprocessing
import os
import sys
from threading import Thread
import time
import uuid
from diplomacy import Game
from tornado import gen, ioloop
from diplomacy_research.models.datasets.grpc_dataset import ModelConfig, GRPCDataset
from diplomacy_research.models.training.reinforcement.common import save_version_model
from diplomacy_research.players import ModelBasedPlayer
from diplomacy_research.utils.cluster import kill_processes_using_port, is_port_opened
from diplomacy_research.utils.process import start_tf_serving, BatchingParameters, kill_subprocesses_on_exit
from diplomacy_research.settings import IN_PRODUCTION

# Constants
LOGGER = logging.getLogger(__name__)
SPAWN_CONTEXT = multiprocessing.get_context('spawn')
RESULT_DICT = {}

def ensure_model_on_disk(trainer):
    """ Makes sure there is at least 1 model on disk, otherwise saves a model for the serving server
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return
    if not has_tf_serving_model(trainer, 'player'):
        save_version_model(trainer, version_id=0)

def start_tf_serving_server(trainer, force_cpu, serving_id, config, endpoint_only=False):
    """ Starts the TF Serving server
        :param trainer: A reinforcement learning trainer instance.
        :param force_cpu: Boolean that indicates that we want the TF serving server to only use the CPU.
        :param serving_id: Integer that represents the serving server id (i.e. when multiple servers are launched)
        :param config: The configuration to set on the serving on launch (None to set no config on launch)
        :param endpoint_only: Boolean that indicates to only launch a sentinel to send orders to another server.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert 'grpc' not in sys.modules, 'gRPC should not be loaded on the main thread.'

    # Making sure we have a model on disk first
    ensure_model_on_disk(trainer)

    # Launching sentinel
    port = get_tf_serving_port(trainer.cluster_config, serving_id)
    trainer.thread_pipes[serving_id], trainer.sentinel_pipes[serving_id] = SPAWN_CONTEXT.Pipe()
    trainer.sentinels[serving_id] = \
        SPAWN_CONTEXT.Process(target=start_serving_sentinel,
                              kwargs={'pipe': trainer.sentinel_pipes[serving_id],
                                      'port': port,
                                      'save_dir': trainer.flags.save_dir,
                                      'force_cpu': force_cpu,
                                      'config': config,
                                      'adapter_ctor': trainer.adapter_constructor,
                                      'dataset_builder_ctor': trainer.dataset_builder_constructor,
                                      'cluster_config': trainer.cluster_config,
                                      'endpoint_only': endpoint_only})
    trainer.sentinels[serving_id].start()

    # Waiting for server
    for attempt_ix in range(300):
        time.sleep(1)
        if is_port_opened(port):
            break
        if (attempt_ix + 1) % 10 == 0:
            LOGGER.info('Waiting for TF Serving to load. Attempt %d / %d.', attempt_ix + 1, 300)
    else:
        LOGGER.error('The TF Serving server did not come online.')
        raise RuntimeError('Unable to contact the serving server.')
    LOGGER.info('Successfully connected to TF Serving.')

def get_tf_serving_port(cluster_config, serving_id):
    """ Returns the port used by the serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param serving_id: Integer that represents the serving server id (i.e. when multiple servers are launched)
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    port = 8500
    if cluster_config and cluster_config.job_name in ['serving', 'actor']:
        serving_address = cluster_config.cluster_spec['serving'][cluster_config.task_id]
        port = int(serving_address.split(':')[1])
    if serving_id:
        port += 2500 + serving_id
    return port

def get_training_config(trainer):
    """ Computes the configuration to set on the TF serving server for training
        :param trainer: A reinforcement learning trainer instance.
        :return: The default TF serving server configuration for training
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    base_path = os.path.join(trainer.flags.save_dir) if IN_PRODUCTION else '/work_dir'

    # All versions uses the latest version for the player
    # self-play uses the same version for both the player and the opponent
    config = [ModelConfig(name='player',
                          base_path=os.path.join(base_path, 'serving', 'player'),
                          version_policy={'latest': 1})]

    # Supervised - Player uses latest version, Opponent uses supervised version
    if trainer.flags.mode == 'supervised':
        config += [ModelConfig(name='opponent',
                               base_path=os.path.join(base_path, 'serving', 'opponent'),
                               version_policy={'specific': 0})]

    # Staggered - Uses the latest opponent version for the opponent
    # The opponent version will be updated sporadically
    elif trainer.flags.mode == 'staggered':
        config += [ModelConfig(name='opponent',
                               base_path=os.path.join(base_path, 'serving', 'opponent'),
                               version_policy={'latest': 1})]

    # Self-play - No need for an opponent
    elif trainer.flags.mode == 'self-play':
        pass

    # Otherwise - Unknown mode
    else:
        LOGGER.error('Valid RL modes are "supervised", "self-play", "staggered".')
        raise ValueError('Unknown RL mode %s. Aborting.' % trainer.flags.mode)

    # Returning configuration
    return config

def get_evaluation_config(trainer):
    """ Computes the default configuration to set on the TF serving server for evaluation
        :param trainer: A reinforcement learning trainer instance.
        :return: The default TF serving server configuration for evaluation
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    base_path = os.path.join(trainer.flags.save_dir) if IN_PRODUCTION else '/work_dir'

    # All versions uses the latest version for the player
    # Evaluation always use the supervised model has the opponent
    config = [ModelConfig(name='player',
                          base_path=os.path.join(base_path, 'serving', 'player'),
                          version_policy={'latest': 1})]

    if trainer.flags.eval_mode in ['supervised-0', 'supervised-1']:
        config += [ModelConfig(name='opponent',
                               base_path=os.path.join(base_path, 'serving', 'opponent'),
                               version_policy={'specific': 0})]

    # Otherwise - Unknown mode
    else:
        LOGGER.error('Valid RL eval modes are "supervised-0", "supervised-1".')
        raise ValueError('Unknown RL eval mode %s. Aborting.' % trainer.flags.eval_mode)

    # Returning configuration
    return config

def has_tf_serving_model(trainer, model_name):
    """ Checks if the specified model name already have a saved model available to load
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    version_folder = os.path.join(trainer.flags.save_dir, 'serving', model_name)
    if not os.path.exists(version_folder):
        return False

    # Looking for a saved_model.pb
    versions = [os.path.join(version_folder, model_version) for model_version in os.listdir(version_folder)
                if os.path.isdir(os.path.join(version_folder, model_version))]
    for version_path in versions:
        if os.path.exists(os.path.join(version_path, 'saved_model.pb')):
            return True

    # No saved models found
    return False

def update_serving(trainer, serving_id=0, config=None):
    """ Updates the configuration of the TF Serving server
        :param trainer: A reinforcement learning trainer instance.
        :param serving_id: Integer that represents the serving server id (i.e. when multiple servers are launched)
        :param config: The configuration to set on the serving. Uses the training configuration if not provided
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert serving_id in trainer.thread_pipes and trainer.thread_pipes[serving_id] is not None, 'Pipe is required.'
    if config is None:
        config = get_training_config(trainer)

    # Setting a request and sending it to server
    request_id = str(uuid.uuid4())
    RESULT_DICT[request_id] = 0
    trainer.thread_pipes[serving_id].send((request_id, 'update', config))

def wait_for_version(trainer, model_name, version_id=None, serving_id=0, timeout=10):
    """ Waits for a specific version to be loaded on the server
        :param trainer: A reinforcement learning trainer instance.
        :param model_name: The name of the model to check. (e.g. 'player' or 'opponent')
        :param version_id: Optional. The version id to wait for (defaults to the current version if not provided).
        :param serving_id: Integer that represents the serving server id (i.e. when multiple servers are launched)
        :param timeout: The maximum time to wait for the new version in seconds.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert serving_id in trainer.thread_pipes and trainer.thread_pipes[serving_id] is not None, 'Pipe is required.'
    request_id = str(uuid.uuid4())
    RESULT_DICT[request_id] = 0
    trainer.thread_pipes[serving_id].send((request_id, 'wait_for_version', {'model_name': model_name,
                                                                            'version_id': version_id,
                                                                            'timeout': timeout}))
    if not wait_for_results(trainer, request_id, timeout):
        LOGGER.info('Timeout of %d reached waiting for wait_for_version.', timeout)

def check_opening_orders(trainer, serving_id=0, timeout=30):
    """ Queries the serving server to get the opening orders and compares them to known openings
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert serving_id in trainer.thread_pipes and trainer.thread_pipes[serving_id] is not None, 'Pipe is required.'
    request_id = str(uuid.uuid4())
    RESULT_DICT[request_id] = 0
    trainer.thread_pipes[serving_id].send((request_id, 'check_opening_orders', None))
    if not wait_for_results(trainer, request_id, timeout):
        LOGGER.info('Timeout of %d reached waiting for check_openings.', timeout)

def wait_for_results(trainer, request_id, timeout):
    """ Waits for results from the sentinel
        :param trainer: A reinforcement learning trainer instance.
        :param request_id: The original request id.
        :param timeout: The maximum amount of time to wait for the request.
        :return: A boolean that indicates if the request was successfully received within the timeout period.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        for thread_pipe in trainer.thread_pipes.values():
            while thread_pipe.poll():
                item_request_id, item_value = thread_pipe.recv()
                RESULT_DICT[item_request_id] = item_value
        if RESULT_DICT.get(request_id, 0):
            return True
        for key, value in RESULT_DICT.items():
            if value and value - time.time() >= 120:
                del RESULT_DICT[key]
        time.sleep(0.1)
    return False

# --------------------------------------------
#             Sentinel and Tasks
# --------------------------------------------
def start_serving_sentinel(**kwargs):
    """ Starts a sentinel to start and monitor the TF serving server """
    return ioloop.IOLoop().run_sync(lambda: monitor_tf_serving(**kwargs))           # pylint: disable=no-value-for-parameter

@gen.coroutine
def monitor_tf_serving(pipe, port, save_dir, force_cpu, config, adapter_ctor, dataset_builder_ctor, cluster_config=None,
                       endpoint_only=False):
    """ Launches and monitors a TF serving server and restarts it if trouble is detected.
        :param pipe: The multiprocessing pipe to communicate with the main thread
        :param port: Integer. The port to use for the TF serving server.
        :param save_dir: The current flags.save_dir
        :param force_cpu: Boolean that indicates that we want the TF serving server to only use the CPU.
        :param config: The configuration to set on the serving on launch (None to set no config on launch)
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param cluster_config: The cluster configuration used for distributed training.
        :param endpoint_only: Boolean that indicates to only launch a sentinel to send orders to another server.

        :type pipe: multiprocessing.connection.Pipe
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    # pylint: disable=too-many-arguments

    # Waiting until we have at least 1 model in the save_dir before starting the server
    # Raising an exception after 5 mins
    for attempt_ix in range(300):
        if glob.glob('%s/*/saved_model.pb' % (os.path.join(save_dir, 'serving', 'player'))):
            break
        time.sleep(1)
        if (attempt_ix + 1) % 30 == 0:
            LOGGER.info('Waiting for TF model to be saved to disk. Attempt %d / 300.', attempt_ix + 1)
    else:
        LOGGER.error('The TF model was not detected on disk after 300 seconds. Aborting.')
        raise RuntimeError('TF serving not detected on disk')

    # Launching tf serving in a separate thread
    if not endpoint_only and not is_port_opened(port):
        task_launch_serving(port, save_dir, force_cpu, config, cluster_config)

    # Creating a game to monitor players
    game = Game()

    # Creating a model-based player for each configuration received
    player_models = []
    if not endpoint_only:
        player_models = task_get_player_models(port, config, adapter_ctor, dataset_builder_ctor, cluster_config)

    # Detects if we can monitor and restart the server
    monitoring_enabled = bool(player_models)
    if not monitoring_enabled and not endpoint_only:
        LOGGER.warning('A configuration was not provided. Serving monitoring has been disabled until it is received.')

    # Cannot do the monitoring if the config is not passed
    assert player_models or endpoint_only, 'A configuration is required when the serving is not only an endpoint.'

    # Processing tasks and monitoring server
    last_status_time = 0
    while True:

        # Monitor server (every 30 secs)
        if monitoring_enabled and (time.time() - last_status_time) >= 30:
            status_ok = yield task_monitor_serving(player_models, game, port, save_dir,
                                                   force_cpu=force_cpu,
                                                   config=config,
                                                   cluster_config=cluster_config)
            if not status_ok:
                continue
            last_status_time = time.time()

        # Performing requests
        while pipe.poll():
            request_id, request_name, request_args = pipe.recv()

            # Check Opening Orders
            if request_name == 'check_opening_orders':
                yield task_check_openings(player_models)

            # Update Config
            elif request_name == 'update':
                config = request_args
                yield task_set_config(port, config)

                # Regenerating players for serving monitoring
                if not endpoint_only:
                    player_models = task_get_player_models(port=port,
                                                           config=config,
                                                           adapter_ctor=adapter_ctor,
                                                           dataset_builder_ctor=dataset_builder_ctor,
                                                           cluster_config=cluster_config)
                    if not monitoring_enabled and player_models:
                        LOGGER.info('Serving monitoring is now enabled.')
                    elif monitoring_enabled and not player_models:
                        LOGGER.info('Serving monitoring is now disabled.')
                    monitoring_enabled = bool(player_models)

            # Wait for version
            elif request_name == 'wait_for_version':
                yield task_wait_for_version(port, **request_args)

            else:
                LOGGER.error('Unknown request: %s - Skipping.', request_name)

            # Marking request as processed
            pipe.send((request_id, int(time.time())))

        # Throttling
        yield gen.sleep(0.1)

def task_get_player_models(port, config, adapter_ctor, dataset_builder_ctor, cluster_config):
    """ Gets the players to use for monitor the serving server
        :param port: Integer. The port to use for the TF serving server.
        :param config: The configuration to set on the serving on launch (None to set no config on launch)
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param cluster_config: The cluster configuration used for distributed training.
        :return: A list of tuples (model_name, player)

        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    player_models = []
    if config is None:
        return player_models
    for model_config in config:
        player_dataset = GRPCDataset(hostname='localhost',
                                     port=port,
                                     model_name=model_config.name,
                                     signature=adapter_ctor.get_signature(),
                                     dataset_builder=dataset_builder_ctor(),
                                     cluster_config=cluster_config)
        player_adapter = adapter_ctor(player_dataset)
        player = ModelBasedPlayer(player_adapter)
        player_models += [(model_config.name, player)]
    return player_models

@gen.coroutine
def task_set_config(port, config, nb_retries=30):
    """ Sets a new configuration on the serving server
        :param port: Integer. The port to use for the TF serving server.
        :param config: A ModelConfig named tuple or a list of ModelConfig named tuple
        :param nb_retries: The number of times to retry setting the configuration.
    """
    if config is None:
        return
    for _ in range(nb_retries):
        if GRPCDataset.set_config('localhost', port, config):
            break
        yield gen.sleep(1.)
    else:
        LOGGER.error('Unable to set the configuration on the TF serving server. Tried %d times. Skipping.', nb_retries)

@gen.coroutine
def task_check_openings(player_models):
    """ Check orders for a certain number of player models
        :param player_models: A list of tuples with (model_name, player)
     """
    print('-' * 80)
    for model_name, player in player_models:
        print('%s:' % model_name)
        yield player.check_openings()
        print('-' * 80)

@gen.coroutine
def task_wait_for_version(port, model_name, version_id, timeout):
    """ Waits until the TF serving is done loading a version in memory
        :param port: The port used by the TensorFlow Serving server.
        :param model_name: The model name that needs to be loaded
        :param version_id: The version that should be loaded
        :param timeout: The maximum number of seconds to wait. Logs an error if reached.
    """
    GRPCDataset.wait_for_model_to_load('localhost', port, model_name, version_id, timeout=timeout + 30)

@gen.coroutine
def task_monitor_serving(player_models, game, port, save_dir, force_cpu, config, cluster_config, nb_retries=3):
    """ Sends a fake request order to the serving to make sure it is still online
        :param player_models: A list of tuples with (model_name, player)
        :param game: A game object to use to query the serving
        :param port: Integer. The port to use for the TF serving server.
        :param save_dir: The current flags.save_dir
        :param force_cpu: Boolean that indicates that we want the TF serving server to only use the CPU.
        :param config: The configuration to set on the serving on launch (None to set no config on launch)
        :param cluster_config: The cluster configuration used for distributed training.
        :param nb_retries: The nb of retries before restarting the server.
        :type game: diplomacy.Game
    """
    import grpc
    status_ok = True

    # No players
    if not player_models:
        return status_ok

    # Monitoring the server
    _, player = player_models[0]
    for _ in range(nb_retries):
        try:
            orders = yield player.get_orders(game, 'FRANCE', retry_on_failure=False)
            if [order for order in orders if order != '']:
                break
        except grpc.RpcError:
            pass
        yield gen.sleep(5)
    else:
        LOGGER.warning('Unable to retrieve orders from the server in the last %d attempts. Restarting it.', nb_retries)
        status_ok = False
        yield task_launch_serving(port, save_dir, force_cpu, config, cluster_config)

    # Returning the status
    return status_ok

@gen.coroutine
def task_launch_serving(port, save_dir, force_cpu, config, cluster_config=None):
    """ Launches (or restarts) a TF serving server
        :param port: Integer. The port to use for the TF serving server.
        :param save_dir: The current flags.save_dir
        :param force_cpu: Launches the tf serving on CPU, otherwise uses GPU when using distributed training.
        :param config: The configuration to set on the serving on launch (None to set no config on launch)
        :param cluster_config: The cluster configuration used for distributed training.
    """
    kill_processes_using_port(port)
    kill_subprocesses_on_exit()

    # Computing launch settings
    simult_players = multiprocessing.cpu_count() * 7                    # launching nb_cpus processes
    max_batch_size = 2 ** int(math.log(0.9 * simult_players, 2))        # rounding down to nearest exponent of 2.
    batch_timeout = 200000
    batching_parameters = BatchingParameters(max_batch_size=max_batch_size,
                                             batch_timeout_micros=batch_timeout,
                                             max_enqueued_batches=256,
                                             num_batch_threads=multiprocessing.cpu_count(),
                                             pad_variable_length_inputs=True)

    log_file_path = None
    if not cluster_config:
        log_file_path = os.path.join(save_dir, 'tf_serving_%d.log' % port)
    elif get_tf_serving_port(cluster_config, serving_id=0) != port:
        log_file_path = os.path.join(save_dir, 'tf_serving_%s_%d.log' % (cluster_config.job_name, port))
    force_cpu = force_cpu or bool(cluster_config is None)

    # Launching
    tf_serving_thread = Thread(target=start_tf_serving,
                               args=(port, save_dir, batching_parameters, cluster_config),
                               kwargs={'force_cpu': force_cpu,
                                       'log_file_path': log_file_path})
    tf_serving_thread.start()

    # Waiting for port
    for _ in range(120):
        if is_port_opened(port):
            break
        yield gen.sleep(1)
    else:
        LOGGER.error('Unable to connect to TF Serving after 2 mins.')

    # Setting configuration
    yield task_set_config(port, config)
