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
""" Memory Buffer - Online Games
    - Class responsible for saving and handling completed (online) games and partial games played by the RL agents
"""
import logging
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import zlib_to_proto, proto_to_zlib, bytes_to_proto

# Constants
LOGGER = logging.getLogger(__name__)
__ONLINE_GAME__ = 'games.%s'                    ## [key] games.{game_id}        -> saved_game_zlib
__PARTIAL_GAME__ = 'games.partial.%s'           ## [key] games.{game_id}/{last} -> saved_game_zlib
__SET_ONLINE_GAMES__ = 'set.online.games'       ## [set] set.online.games       -> set of game_id for online games
__SET_PARTIAL_GAMES__ = 'set.partial_games'     ## [set] set.partial_games      -> set of {game_id}.{last_phase}
__ALL_GAMES__ = 'saved.games'                   ## [zset] saved.games           -> {game_id}/{nb_items}: ts
__PARTIAL_GAME_ID__ = '%s/%s'                   ## {game_id}/{last_phase}


def list_games(buffer, all_games=False, excluding=None, with_completed=True, with_partial=True):
    """ Retrieves the list of saved games id that are assigned to this shard
        :param buffer: An instance of the memory buffer.
        :param all_games: Boolean that indicates we want to return the games for all shards.
        :param excluding: Optional. A list of game ids to exclude (e.g. because they already have been retrieved)
        :param with_completed: Boolean that indicates we want to include completed games.
        :param with_partial: Boolean that indicates we want to include partial games.
        :return: A list of game ids matching the specified criteria.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    num_shards, shard_index = 1, 0
    if not all_games and buffer.cluster_config:
        assert buffer.cluster_config.job_name == 'learner', 'list_games() should only be used by a learner.'
        num_shards = buffer.cluster_config.num_shards
        shard_index = buffer.cluster_config.shard_index

    # Retrieving the games proto for the current shard
    all_game_ids = []
    if with_completed:
        all_game_ids += [game_id.decode('utf-8') for game_id in buffer.redis.smembers(__SET_ONLINE_GAMES__)]
    if with_partial:
        all_game_ids += [game_id.decode('utf-8') for game_id in buffer.redis.smembers(__SET_PARTIAL_GAMES__)]
    excluding = excluding or []
    shard_game_ids = [game_id for game_id in all_game_ids
                      if hash(game_id) % num_shards == shard_index and game_id not in excluding]
    return shard_game_ids

def save_games(buffer, saved_games_proto=None, saved_games_bytes=None):
    """ Stores a series of games in compressed saved game proto format
        :param buffer: An instance of the memory buffer.
        :param saved_games_bytes: List of saved game (bytes format) (either completed or partial)
        :param saved_games_proto: List of saved game proto (either completed or partial)
        :return: Nothing
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    assert bool(saved_games_bytes is None) != bool(saved_games_proto is None), 'Expected one of bytes or proto'
    saved_games_bytes = saved_games_bytes or []
    saved_games_proto = saved_games_proto or []

    if saved_games_bytes:
        saved_games_proto = [bytes_to_proto(game_bytes, SavedGameProto) for game_bytes in saved_games_bytes]

    # Splitting between completed game ids and partial game ids
    completed, partial = [], []
    for saved_game_proto in saved_games_proto:
        if not saved_game_proto.is_partial_game:
            completed += [saved_game_proto]
        else:
            partial += [saved_game_proto]

    # No games
    if not completed and not partial:
        LOGGER.warning('Trying to save saved_games_proto, but no games provided. Skipping.')
        return

    # Compressing games
    completed_games_zlib = [proto_to_zlib(saved_game_proto) for saved_game_proto in completed]
    completed_game_ids = [saved_game_proto.id for saved_game_proto in completed]

    partial_games_zlib = [proto_to_zlib(saved_game_proto) for saved_game_proto in partial]
    partial_game_ids = [__PARTIAL_GAME_ID__ % (saved_game_proto.id, saved_game_proto.phases[-1].name)
                        for saved_game_proto in partial]

    # Saving games
    pipeline = buffer.redis.pipeline()
    if completed_game_ids:
        for game_id, saved_game_zlib in zip(completed_game_ids, completed_games_zlib):
            pipeline.set(__ONLINE_GAME__ % game_id, saved_game_zlib)
        pipeline.sadd(__SET_ONLINE_GAMES__, *completed_game_ids)

    if partial_game_ids:
        for game_id, saved_game_zlib in zip(partial_game_ids, partial_games_zlib):
            pipeline.set(__PARTIAL_GAME__ % game_id, saved_game_zlib)
        pipeline.sadd(__SET_PARTIAL_GAMES__, *partial_game_ids)

    # Executing
    pipeline.execute()

def get_online_games(buffer, all_games=False, excluding=None, with_completed=True, with_partial=True):
    """ Retrieves the saved games proto that are assigned to this shard.
        :param buffer: An instance of the memory buffer.
        :param all_games: Boolean that indicates we want to return the games for all shards.
        :param excluding: Optional. A list of game ids to exclude (e.g. because they already have been retrieved)
        :param with_completed: Boolean that indicates we want to include completed games.
        :param with_partial: Boolean that indicates we want to include partial games.
        :return: Tuple of:
                    1) List of saved games proto
                    2) List of game ids
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    shard_game_ids = list_games(buffer,
                                all_games=all_games,
                                excluding=excluding,
                                with_completed=with_completed,
                                with_partial=with_partial)
    if not shard_game_ids:
        return [], []

    # Splitting between partial and completed games
    completed_game_ids = [game_id for game_id in shard_game_ids if '/' not in game_id]
    partial_game_ids = [game_id for game_id in shard_game_ids if '/' in game_id]

    # Retrieving the games
    saved_games_proto = []
    if completed_game_ids:
        completed_game_zlibs = buffer.redis.mget([__ONLINE_GAME__ % game_id for game_id in completed_game_ids])
        saved_games_proto += [zlib_to_proto(saved_game_zlib, SavedGameProto)
                              for saved_game_zlib in completed_game_zlibs if saved_game_zlib is not None]
    if partial_game_ids:
        partial_game_zlibs = buffer.redis.mget([__PARTIAL_GAME__ % game_id for game_id in partial_game_ids])
        saved_games_proto += [zlib_to_proto(saved_game_zlib, SavedGameProto)
                              for saved_game_zlib in partial_game_zlibs if saved_game_zlib is not None]

    # Returning
    return saved_games_proto, completed_game_ids + partial_game_ids

def mark_games_as_processed(buffer, game_ids):
    """ Mark the games as processed, so they don't appear in the list of online games anymore
        :param buffer: An instance of the memory buffer.
        :param game_ids: List of game ids to mark as processed.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    if not game_ids:
        LOGGER.warning('Trying to mark 0 games as processed. Skipping.')
        return

    # Splitting between completed and partial game ids
    completed_game_ids = [game_id for game_id in game_ids if '/' not in game_id]
    partial_game_ids = [game_id for game_id in game_ids if '/' in game_id]

    # Deleting games
    pipeline = buffer.redis.pipeline()
    if completed_game_ids:
        pipeline.srem(__SET_ONLINE_GAMES__, *completed_game_ids)

    if partial_game_ids:
        pipeline.srem(__SET_PARTIAL_GAMES__, *partial_game_ids)
        pipeline.delete(*[__PARTIAL_GAME__ % game_id for game_id in partial_game_ids])

    # No experience replay so we can delete the games
    # Otherwise, we just delete them from the set
    if not buffer.use_experience_replay and completed_game_ids:
        pipeline.delete(*[__ONLINE_GAME__ % game_id for game_id in completed_game_ids])

    # Executing
    pipeline.execute()
    LOGGER.info('%d completed and %d partial games have been marked as processed',
                len(completed_game_ids), len(partial_game_ids))
