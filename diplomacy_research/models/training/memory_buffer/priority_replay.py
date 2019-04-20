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
""" Memory Buffer - Priority Replay
    - Class responsible for creating storing and retrieving replay samples with priorities in the memory buffer
"""
import logging
import time
import numpy as np
from numpy.random import randint
from diplomacy_research.models.self_play.transition import ReplaySample
from diplomacy_research.models.training.memory_buffer.online_games import __ONLINE_GAME__, __ALL_GAMES__
from diplomacy_research.models.state_space import ALL_POWERS
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import zlib_to_proto

# Constants
LOGGER = logging.getLogger(__name__)
__GAME_NB_ITEMS__ = '%s/%d'                     ## {game_id}/nb_items
__PRIORITY__ = 'saved.priority'                 ## [zset] saved.priority        -> {game_id}.{pow}.{phase_ix}: abs(td)
__PRIORITY_ITEM_ID__ = '%s.%s.%03d'             ## {game_id}.{power_name}.phase_ix


def get_replay_samples(buffer, nb_samples):
    """ Samples a series of replay samples from the memory buffer.
        The buffer must have at least min_items, otherwise no samples are returned

        :param buffer: An instance of the memory buffer.
        :param nb_samples: The number of items to sample from the buffer.
        :return: A list of ReplaySamples
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
        :rtype: [ReplaySample]
    """
    partitions = get_partitions(buffer, nb_samples)

    # The buffer is not full enough, skipping.
    if not partitions or not nb_samples:
        return []

    # Sampling and retrieving the priority ids
    ranks = [randint(start_rank, max(start_rank + 1, end_rank)) for (start_rank, end_rank) in partitions]
    pipeline = buffer.redis.pipeline()
    for rank in ranks:
        pipeline.zrange(__PRIORITY__, start=rank, end=rank)
    priority_ids = [zrange_result[0] for zrange_result in pipeline.execute()]

    # Extracting a dictionary of {power_name: [phases_ix]} for each game
    power_phases_ix_per_game = {}
    for priority_id in priority_ids:                                        # Format: {game_id.power_name.phase_ix}
        game_id, power_name, phase_ix = priority_id.split('.')
        phase_ix = int(phase_ix)
        if game_id not in power_phases_ix_per_game:
            power_phases_ix_per_game[game_id] = {}
        if power_name not in power_phases_ix_per_game[game_id]:
            power_phases_ix_per_game[game_id][power_name] = []
        power_phases_ix_per_game[game_id][power_name] += [phase_ix]

    # Retrieving all saved games zlib
    game_ids = list(power_phases_ix_per_game.keys())
    saved_games_zlib = buffer.redis.mget([__ONLINE_GAME__ % game_id for game_id in game_ids])
    saved_games_proto = [zlib_to_proto(saved_game_zlib, SavedGameProto) for saved_game_zlib in saved_games_zlib]

    # Building replay samples
    replay_samples, nb_replay_samples = [], 0
    for game_id, saved_game_proto in zip(game_ids, saved_games_proto):
        replay_samples += [ReplaySample(saved_game_proto=saved_game_proto,
                                        power_phases_ix=power_phases_ix_per_game[game_id])]
        for power_name in power_phases_ix_per_game[game_id]:
            nb_replay_samples += len(power_phases_ix_per_game[game_id][power_name])

    # Returning
    LOGGER.info('Sampled %d items from the memory buffer (Target: %d).', nb_replay_samples, nb_samples)
    return replay_samples

def update_priorities(buffer, new_priorities, first_update=False):
    """ Update the priorities of transitions
        :param buffer: An instance of the memory buffer.
        :param new_priorities: List of priorities
        :param first_update: Boolean that indicates that the priorities are being added for the first time.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
        :type new_priorities: [diplomacy_research.models.self_play.transition.ReplayPriority]
    """
    if not new_priorities:
        LOGGER.warning('Trying to update priorities, but no priorities provided. Skipping.')
        return

    keys_per_game = {}                                                  # {game_id: [keys]}
    priorities = []

    # Converting the new priorities
    for priority in new_priorities:
        priority_key = __PRIORITY_ITEM_ID__ % (priority.game_id, priority.power_name, priority.phase_ix)
        priorities += [(priority_key, priority.priority)]
        if first_update:
            if priority.game_id not in keys_per_game:
                keys_per_game[priority.game_id] = []
            keys_per_game[priority.game_id] += [priority_key]

    # Building pipeline
    pipeline = buffer.redis.pipeline()
    buffer.redis.zadd(__PRIORITY__, **dict(priorities))
    for game_id in keys_per_game:
        pipeline.zadd(__ALL_GAMES__,
                      __GAME_NB_ITEMS__ % (game_id, len(keys_per_game[game_id])),
                      time.time())
    LOGGER.info('%s the priorities of %d samples in the memory buffer.',
                'Added' if first_update else 'Updated', len(new_priorities))
    pipeline.execute()

def trim_buffer(buffer):
    """ Trim the buffer so that it does not exceed the maximum number of items allowed.
        :param buffer: An instance of the memory buffer.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    nb_items = buffer.redis.zcard(__PRIORITY__)
    if nb_items < buffer.max_items:
        return

    # Retrieving a list of games to possible remove
    # Deciding how many games to remove
    # Executing the pipeline
    while nb_items > buffer.max_items:
        list_game_ids = buffer.redis.zrange(__ALL_GAMES__, 0, 99)       # 100 games ids ({game_id}/{nb_items})

        # No game ids found - Error
        if not list_game_ids:
            LOGGER.error('Unable to trim the buffer - The list of games is empty.')
            break

        # Creating pipeline
        nb_excess_transitions = nb_items - buffer.max_items
        nb_deleted_transitions = 0
        pipeline = buffer.redis.pipeline()
        for game_id_nb_items in list_game_ids:
            game_id_nb_items = game_id_nb_items.decode('utf-8')
            game_id, nb_game_items = game_id_nb_items.split('/')
            nb_game_items = int(nb_game_items)

            # We pop the game from the buffer
            if nb_excess_transitions > 0:
                pipeline.delete(__ONLINE_GAME__ % game_id)
                pipeline.zrem(__PRIORITY__, *[__PRIORITY_ITEM_ID__ % (game_id, power_name, item_ix)
                                              for power_name in ALL_POWERS
                                              for item_ix in range(nb_game_items)])
                pipeline.zrem(__ALL_GAMES__, game_id_nb_items)
                nb_deleted_transitions += game_id_nb_items
                nb_excess_transitions -= game_id_nb_items

            # Otherwise, we can stop
            else:
                break

        # Executing pipeline
        pipeline.execute()
        nb_items -= nb_deleted_transitions

def get_partitions(buffer, nb_samples):
    """ Compute nb_samples partitions each of probability 1. / nb_samples from the transitions in the
        priority buffer.
        - The number of transitions in the buffer is rounded down to the nearest 10k to cache the partitions

        :param buffer: An instance of the memory buffer.
        :param nb_samples: The number of partitions to create.
        :return: A list of `nb_samples` tuples, each tuple being (start_rank, end_rank)
                 start_rank is inclusive, but end_rank is exclusive
                 e.g. [ (0, 3), (3, 5), ...
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    total_transitions = buffer.redis.zcard(__PRIORITY__)
    if total_transitions < buffer.min_items:
        return None

    # Trying to retrieve from cache
    if total_transitions > 10000:
        total_transitions = 10000 * (total_transitions // 10000)  # Rounding-down to the nearest 10k.
    if (nb_samples, total_transitions) in buffer.partition_cache:
        return buffer.partition_cache[(nb_samples, total_transitions)]

    # Otherwise, computing and storing in cache
    pdf = (np.arange(1, total_transitions + 1)) ** (-buffer.alpha)
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf)

    # Partition index can be obtained as the quotient of cdf / (1 / nb_parts)
    assigned_partition = cdf // (1. / nb_samples)

    # Building a list of (start_rank, end_rank)
    partitions = []
    last_end_rank = 0
    for partition_ix in range(nb_samples):
        after_partition = assigned_partition > partition_ix
        start_rank = last_end_rank
        end_rank = int(np.argmax(after_partition)) if np.sum(after_partition) else total_transitions
        partitions += [(start_rank, end_rank)]
        last_end_rank = end_rank

    # Storing in cache
    buffer.partition_cache[(nb_samples, total_transitions)] = partitions
    while len(buffer.partition_cache) > 5:
        del buffer.partition_cache[list(buffer.partition_cache.keys())[0]]

    # Returning
    return partitions
