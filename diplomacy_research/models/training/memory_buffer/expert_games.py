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
""" Memory Buffer - Expert Games
    - Class responsible for creating interacting with expert games on the memory buffer
"""
import logging
import os
import pickle
import h5py
from numpy.random import choice, randint
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import zlib_to_proto
from diplomacy_research.settings import DATASET_PATH, DATASET_INDEX_PATH

# Constants
LOGGER = logging.getLogger(__name__)
__EXPERT_GAME__ = 'expert.%s'                   ## [key] expert.{game_id}       -> saved_game_zlib
__SET_EXPERT_GAMES__ = 'set.expert.games'       ## [set] set.expert.games       -> set of game_id for expert games


def list_expert_games(buffer):
    """ Returns a set with the game ids of all expert games in the memory buffer
        :param buffer: An instance of the memory buffer.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    return {item.decode('utf-8') for item in buffer.redis.smembers(__SET_EXPERT_GAMES__)}

def load_expert_games(buffer):
    """ Load expert games in the memory buffer
        :param buffer: An instance of the memory buffer.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    if buffer.cluster_config and not buffer.cluster_config.is_chief:
        return

    # Making sure the dataset index exists on disk
    if not os.path.exists(DATASET_INDEX_PATH):
        if buffer.hparams['start_strategy'] != 'beginning':
            raise RuntimeError('The dataset index path is required for %s' % buffer.hparams['start_strategy'])
        LOGGER.warning('Unable to find the dataset index on disk at %s. Skipping', DATASET_INDEX_PATH)
        return

    # Build the list of games in the expert dataset
    expert_game_ids = set()
    with open(DATASET_INDEX_PATH, 'rb') as dataset_index:
        dataset_index = pickle.load(dataset_index)
    expert_dataset = buffer.hparams['expert_dataset'].split(',')

    # Listing all games in dataset
    for dataset_name in expert_dataset:
        if dataset_name == 'no_press':
            expert_game_ids |= dataset_index.get('standard_no_press', set())
        elif dataset_name == 'press':
            expert_game_ids |= dataset_index.get('standard_press_with_msgs', set())
            expert_game_ids |= dataset_index.get('standard_press_without_msgs', set())
            expert_game_ids |= dataset_index.get('standard_public_press', set())
        else:
            LOGGER.warning('Unknown expert dataset "%s". Expected "press", or "no_press".', dataset_name)

    # Listing all games in the memory buffer, and computing difference
    memory_game_ids = list_expert_games(buffer)
    missing_game_ids = list(expert_game_ids - memory_game_ids)
    extra_game_ids = list(memory_game_ids - expert_game_ids)

    # Storing missing games
    if missing_game_ids:
        LOGGER.info('Loading %d expert games in the memory buffer.', len(missing_game_ids))
        nb_games_remaining = len(missing_game_ids)
        game_ids, saved_games_zlib = [], []
        with h5py.File(DATASET_PATH) as dataset:
            for game_id in missing_game_ids:
                game_ids += [game_id]
                saved_games_zlib += [dataset[game_id].value.tostring()]

                # Saving mini-batches of 100 games
                if len(game_ids) >= 100:
                    save_expert_games(buffer, saved_games_zlib, game_ids)
                    nb_games_remaining -= len(game_ids)
                    game_ids, saved_games_zlib = [], []
                    LOGGER.info('%d games still needing to be saved in the memory buffer.', nb_games_remaining)

        # Done transferring all games
        save_expert_games(buffer, saved_games_zlib, game_ids)

    # Removing extra games
    if extra_game_ids:
        LOGGER.info('Removing %d extra expert games from the memory buffer', len(extra_game_ids))
        delete_expert_games(buffer, expert_game_ids)

    # Done
    if missing_game_ids or expert_game_ids:
        buffer.save()
    LOGGER.info('Done saving all %d expert games in the memory buffer.', len(expert_game_ids))

def save_expert_games(buffer, saved_games_zlib, game_ids):
    """ Stores a series of expert games in compressed saved game proto format
        :param buffer: An instance of the memory buffer.
        :param saved_games_zlib: List of compressed saved game proto
        :param game_ids: List of game ids (same length as list_saved_game_zlib)
        :return: Nothing
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    if not game_ids:
        LOGGER.warning('Trying to save expert games, but no expert games provided. Skipping.')
        return
    pipeline = buffer.redis.pipeline()
    for game_id, saved_game_zlib in zip(game_ids, saved_games_zlib):
        pipeline.set(__EXPERT_GAME__ % game_id, saved_game_zlib, nx=True)           # Saving game
    pipeline.sadd(__SET_EXPERT_GAMES__, *game_ids)                                  # Adding expert games to set
    pipeline.execute()

def delete_expert_games(buffer, game_ids):
    """ Deletes a series of expert games from the memory buffer
        :param buffer: An instance of the memory buffer.
        :param game_ids: List of game ids to delete
        :return: Nothing
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    if not game_ids:
        LOGGER.warning('Trying to delete expert games, but no expert games provided. Skipping.')
        return
    pipeline = buffer.redis.pipeline()
    pipeline.delete(*[__EXPERT_GAME__ % game_id for game_id in game_ids])
    pipeline.srem(__SET_EXPERT_GAMES__, *game_ids)
    pipeline.execute()

def get_uniform_initial_states(buffer, nb_states):
    """ Returns a list of random states from the expert games
        :param buffer: An instance of the memory buffer.
        :param nb_states: The number of random states we want.
        :return: A list of state_proto to use as the initial state of a game
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    game_ids = [game_id.decode('utf-8')
                for game_id in buffer.redis.srandmember(__SET_EXPERT_GAMES__, -1 * nb_states)]    # With replacement
    saved_games_zlib = buffer.redis.mget([__EXPERT_GAME__ % game_id for game_id in game_ids])
    saved_games_proto = [zlib_to_proto(saved_game_zlib, SavedGameProto) for saved_game_zlib in saved_games_zlib
                         if saved_game_zlib is not None]

    # No games found
    if not saved_games_proto:
        raise RuntimeError('Unable to retrieve random states from the memory buffer.')

    # Sampling random states
    random_states = []
    while len(random_states) < nb_states:
        saved_game_proto = saved_games_proto[choice(len(saved_games_proto))]
        random_states += [saved_game_proto.phases[choice(len(saved_game_proto.phases))].state]
    return random_states

def get_backplay_initial_states(buffer, winning_power_names, version_id):
    """ Return a list of backplay states from the expert games
        :param buffer: An instance of the memory buffer.
        :param winning_power_names: A list of the power_name winning the game (e.g. ['AUSTRIA', 'FRANCE', ...])
        :param version_id: Integer. The current version id
        :return: A list of state_proto (1 per power) to use as the initial state of a game
    """
    buffer.initialize()                                          # To make sure we have a list of won game ids.
    version_id = max(0, version_id)
    selected_states = {}

    # Sampling a certain nb of games for each power
    for power_name in set(winning_power_names):
        nb_games = len([1 for pow_name in winning_power_names if pow_name == power_name])
        nb_won_games = len(buffer.won_game_ids[power_name])
        game_ids = [buffer.won_game_ids[power_name][choice(nb_won_games)] for _ in range(nb_games)]
        saved_games_zlib = buffer.redis.mget([__EXPERT_GAME__ % game_id for game_id in game_ids])
        saved_games_proto = [zlib_to_proto(saved_game_zlib, SavedGameProto) for saved_game_zlib in saved_games_zlib
                             if saved_game_zlib is not None]

        # No games found
        if not saved_games_proto:
            raise RuntimeError('Unable to retrieve expert states from the memory buffer for %s.' % power_name)

        # Selecting states
        selected_states[power_name] = []
        while len(selected_states[power_name]) < nb_games:
            saved_game_proto = saved_games_proto[choice(len(saved_games_proto))]
            min_phase_ix = max(0, len(saved_game_proto.phases) - 5 - version_id // 3)
            max_phase_ix = len(saved_game_proto.phases)
            selected_states[power_name] += [saved_game_proto.phases[randint(min_phase_ix, max_phase_ix)].state]

    # Re-ordering
    backplay_states = []
    for power_name in winning_power_names:
        backplay_states += [selected_states[power_name].pop(0)]
    return backplay_states
