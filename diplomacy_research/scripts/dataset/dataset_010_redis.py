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
""" Redis dataset
    - Populated the Redis server with the supervised games
    - Saves the redis database on disk for faster boot time.
"""
import logging
import os
import pickle
import shutil
from threading import Thread
from tqdm import tqdm
from diplomacy_research.models.training.memory_buffer import MemoryBuffer
from diplomacy_research.models.training.memory_buffer.expert_games import save_expert_games
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.process import start_redis
from diplomacy_research.utils.proto import bytes_to_zlib, bytes_to_proto, read_next_bytes
from diplomacy_research.settings import PROTO_DATASET_PATH, REDIS_DATASET_PATH, WORKING_DIR, \
    PHASES_COUNT_DATASET_PATH, IN_PRODUCTION

# Constants
LOGGER = logging.getLogger(__name__)

def run(**kwargs):
    """ Run the script - Determines if we need to build the dataset or not. """
    del kwargs          # Unused args
    if os.path.exists(REDIS_DATASET_PATH):
        LOGGER.info('... Dataset already exists. Skipping.')
    else:
        build()

def build():
    """ Building the Redis dataset """
    if not os.path.exists(PROTO_DATASET_PATH):
        raise RuntimeError('Unable to find the proto dataset at %s' % PROTO_DATASET_PATH)

    # Creating output directory if it doesn't exist
    os.makedirs(os.path.join(WORKING_DIR, 'containers', 'redis'), exist_ok=True)

    # Starting the Redis server and blocking on that thread
    redis_thread = Thread(target=start_redis, kwargs={'save_dir': os.path.join(WORKING_DIR, 'containers'),
                                                      'log_file_path': os.devnull,
                                                      'clear': True})
    redis_thread.start()

    # Creating a memory buffer object to save games in Redis
    memory_buffer = MemoryBuffer()
    memory_buffer.clear()

    # Loading the phases count dataset to get the number of games
    total = None
    if os.path.exists(PHASES_COUNT_DATASET_PATH):
        with open(PHASES_COUNT_DATASET_PATH, 'rb') as file:
            total = len(pickle.load(file))
    progress_bar = tqdm(total=total)

    # Loading dataset and converting
    LOGGER.info('... Creating redis dataset.')
    with open(PROTO_DATASET_PATH, 'rb') as file:
        while True:
            saved_game_bytes = read_next_bytes(file)
            if saved_game_bytes is None:
                break
            progress_bar.update(1)
            saved_game_proto = bytes_to_proto(saved_game_bytes, SavedGameProto)
            save_expert_games(memory_buffer, [bytes_to_zlib(saved_game_bytes)], [saved_game_proto.id])

    # Saving
    memory_buffer.save(sync=True)

    # Moving file
    redis_db_path = {True: '/work_dir/redis/saved_redis.rdb',
                     False: os.path.join(WORKING_DIR, 'containers', 'redis', 'saved_redis.rdb')}.get(IN_PRODUCTION)
    shutil.move(redis_db_path, REDIS_DATASET_PATH)
    LOGGER.info('... Done creating redis dataset.')

    # Stopping Redis and thread
    progress_bar.close()
    memory_buffer.shutdown()
    redis_thread.join(timeout=60)
