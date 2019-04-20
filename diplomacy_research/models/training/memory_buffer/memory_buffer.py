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
""" Memory Buffer
    - Contains the class for interacting with a Redis server that holds completed games and experience replay
"""
from collections import OrderedDict
import logging
import os
import pickle
from threading import Thread
import time
import redis
from redis.exceptions import BusyLoadingError, ResponseError
from diplomacy_research.models.training.memory_buffer.barrier import __BARRIER__
from diplomacy_research.models.state_space import ALL_POWERS
from diplomacy_research.utils.process import is_redis_running, start_redis, kill_subprocesses_on_exit
from diplomacy_research.settings import END_SCS_DATASET_PATH, REDIS_DATASET_PATH

# Constants
LOGGER = logging.getLogger(__name__)
__READY_KEY__ = 'cluster.ready'
__LOCK_READY__ = 'lock.ready'


def start_redis_server(trainer, import_db=True):
    """ Starts the redis server
        :param trainer: A reinforcement learning trainer instance.
        :param import_db: Optional. Boolean that indicates to import the pre-loaded game database
    """
    import_db_path = None if not import_db else REDIS_DATASET_PATH

    # In distributed mode, we launch in the current process
    if trainer.cluster_config:
        start_redis(trainer.flags.save_dir, import_db_path=import_db_path)
        kill_subprocesses_on_exit()
        return

    # Otherwise, we launch in a separate (non-blocking) thread
    redis_thread = Thread(target=start_redis,
                          kwargs={'save_dir': trainer.flags.save_dir,
                                  'log_file_path': os.path.join(trainer.flags.save_dir, 'redis.log'),
                                  'import_db_path': import_db_path})
    redis_thread.start()
    kill_subprocesses_on_exit()

class MemoryBuffer():
    """ A simple buffer just used for storing games and experience replay """

    def __init__(self, cluster_config=None, hparams=None):
        """  Constructor
            :param cluster_config: The cluster configuration to use to retrieve the server's address.
            :param hparams: The hyper-parameters used by the current model.
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        if cluster_config and not cluster_config.count('redis'):
            LOGGER.error('Unable to detect Redis server address. Make sure the Redis task is specified in the config.')
            raise RuntimeError('Unable to detect Redis address.')

        self.cluster_config = cluster_config
        self.hparams = hparams or {}
        self.partition_cache = OrderedDict()                        # (nb_samples, nb_transitions): partition
        self.redis = None
        self.won_game_ids = {power_name: [] for power_name in ALL_POWERS}
        self.initialized = False

        # Storing settings
        # min_items: Min nb of items required in the buffer before sampling for experience replay
        # max_items: Max nb of items in the buffer. Trimmed in FIFO manner if exceeded.
        # alpha: Alpha coefficient to use to compute buckets for sampling.
        self.min_items = self.hparams.get('min_buffer_items', 10000)
        self.max_items = self.hparams.get('max_buffer_items', 2000000)
        self.alpha = self.hparams.get('replay_alpha', 0.7)
        self.use_experience_replay = self.hparams.get('experience_replay', 0.) > 0

        # Connecting to server
        self.connect()

    def connect(self):
        """ Connects to the Redis server and waits for the database to be ready """
        redis_ip = 'localhost' if not self.cluster_config else self.cluster_config.cluster_spec['redis'][0]
        redis_ip = redis_ip.split(':')[0]

        # Making sure redis is running
        # Trying to connect every 1 secs for 5 mins
        for retry_ix in range(300):
            if is_redis_running(redis_ip):
                break
            if (retry_ix + 1) % 10 == 0:
                LOGGER.warning('Redis is not running on %s. Attempt %d / %d', redis_ip, retry_ix + 1, 300)
            time.sleep(1)
        else:
            raise RuntimeError('Redis server is not running. Aborting.')
        LOGGER.info('Successfully connected to Redis server.')

        # Creating connection
        self.redis = redis.Redis(host=redis_ip, socket_keepalive=True, decode_responses=False, retry_on_timeout=True)

        # Waiting until redis is ready (i.e. done loading database)
        for retry_ix in range(120):
            try:
                info = self.redis.info(section='persistence')
                if not info.get('loading', 1):
                    break
            except BusyLoadingError:
                continue
            if (retry_ix + 1) % 10 == 0:
                LOGGER.info('Waiting for redis to finish loading... %d / %d', retry_ix + 1, 120)
            time.sleep(1)
        else:
            LOGGER.error('Redis is still loading after timeout reached. Aborting.')
            raise BusyLoadingError('Redis is loading dataset in memory.')
        LOGGER.info('Redis has done loading the dataset from disk and is ready.')

    def initialize(self):
        """ Performs initialization tasks (i.e. get list of won game ids from disk) """
        if self.initialized:
            return
        self.initialized = True

        if not os.path.exists(END_SCS_DATASET_PATH):
            if self.hparams['start_strategy'] == 'backplay':
                raise RuntimeError('Unable to load the nb of supply centers. Required for "backplay". Aborting.')
            LOGGER.warning('Unable to load the nb of supply centers from %s. - File not found', END_SCS_DATASET_PATH)
            return

        # Loading nb of scs at end of game
        with open(END_SCS_DATASET_PATH, 'rb') as end_scs_dataset:
            end_scs_dataset = pickle.load(end_scs_dataset)

            # Finding games where each power won
            expert_dataset = self.hparams['expert_dataset'].split(',')
            for dataset_name in expert_dataset:
                for power_name in ALL_POWERS:
                    if dataset_name == 'no_press':
                        self.won_game_ids[power_name] += end_scs_dataset['no_press'][power_name][16]
                        self.won_game_ids[power_name] += end_scs_dataset['no_press'][power_name][17]
                        self.won_game_ids[power_name] += end_scs_dataset['no_press'][power_name][18]
                    if dataset_name == 'press':
                        self.won_game_ids[power_name] += end_scs_dataset['press'][power_name][16]
                        self.won_game_ids[power_name] += end_scs_dataset['press'][power_name][17]
                        self.won_game_ids[power_name] += end_scs_dataset['press'][power_name][18]

    @property
    def lock(self):
        """ Returns the Redis lock """
        return self.redis.lock

    def mark_as_ready(self):
        """ Called by the chief to indicate that the cluster is ready to start """
        if not self.cluster_config:
            return
        if not self.cluster_config.is_chief:
            LOGGER.warning('Only the chief can mark the cluster as ready.')
            return

        # Marking the cluster as ready
        timestamp = int(time.time())
        with self.redis.lock(__LOCK_READY__, timeout=120):
            barrier_keys = self.redis.keys(__BARRIER__ % '*')
            if barrier_keys:
                self.redis.delete(*barrier_keys)
            self.redis.set(__READY_KEY__, timestamp, ex=180)            # Opening a 3 min window where cluster can start
        LOGGER.info('[Chief] The cluster has been marked as ready.')

    def wait_until_ready(self, timeout=3600):
        """ Waits until the chief has indicated that the cluster is ready to start """
        # Chief never waits
        if not self.cluster_config or self.cluster_config.is_chief:
            return

        for retry_ix in range(timeout):
            time.sleep(1)
            timestamp = int(time.time())
            ready_key_value = self.redis.get(__READY_KEY__)

            # We have a ready key set within the 3 min window
            if ready_key_value is not None and abs(timestamp - int(ready_key_value)) <= 180:
                break

            # Otherwise, waiting for the ready key to be set.
            if retry_ix % 10 == 0:
                LOGGER.info('Waiting for chief to mark the cluster as ready... %d / %d', retry_ix, timeout)
        else:
            LOGGER.error('Waited %d seconds for cluster to be ready. Is the chief online?', timeout)
            raise RuntimeError('Cluster was never marked as ready.')
        LOGGER.info('Cluster was marked as ready by chief. Continuing.')

    def save(self, sync=False):
        """ Saves the buffer to disk """
        try:
            if sync:
                self.redis.save()
            else:
                self.redis.bgsave()
        except ResponseError:                # Save already in progress.
            pass

    def clear(self):
        """ Deletes the entire memory buffer """
        self.redis.flushall()

    def shutdown(self):
        """ Shutdown the redis server """
        self.redis.shutdown()
