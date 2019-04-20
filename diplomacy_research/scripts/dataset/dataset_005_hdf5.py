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
""" hdf5_dataset
    - Builds a hdf5 of compressed protos for each json in the zip dataset
    - Also adds the possible orders to each phase
"""
from collections import OrderedDict
import glob
import logging
import multiprocessing
import os
import pickle
import shutil
import zipfile
from diplomacy import Game, Map
import h5py
import numpy as np
from tqdm import tqdm
import ujson as json
from diplomacy_research.models.self_play.reward_functions import DefaultRewardFunction
from diplomacy_research.models.state_space import get_map_powers, dict_to_flatten_board_state, \
    dict_to_flatten_prev_orders_state, ALL_POWERS
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import proto_to_zlib, dict_to_proto, write_proto_to_file, zlib_to_proto
from diplomacy_research.settings import ZIP_DATASET_PATH, DATASET_PATH, PROTO_DATASET_PATH, DATASET_INDEX_PATH, \
    END_SCS_DATASET_PATH, HASH_DATASET_PATH, MOVES_COUNT_DATASET_PATH, PHASES_COUNT_DATASET_PATH

# Constants
LOGGER = logging.getLogger(__name__)

def run(**kwargs):
    """ Run the script - Determines if we need to build the dataset or not. """
    del kwargs          # Unused args
    if os.path.exists(DATASET_PATH):
        LOGGER.info('... Dataset already exists. Skipping.')
    else:
        build()

def add_possible_orders_to_saved_game(saved_game):
    """ Adds possible_orders for each phase of the saved game """
    if saved_game['map'].startswith('standard'):
        for phase in saved_game['phases']:
            game = Game(map_name=saved_game['map'], rules=saved_game['rules'])
            game.set_state(phase['state'])
            phase['possible_orders'] = game.get_all_possible_orders()
    return saved_game

def add_cached_states_to_saved_game(saved_game):
    """ Adds a cached representation of board_state and prev_orders_state to the saved game """
    if saved_game['map'].startswith('standard'):
        map_object = Map(saved_game['map'])
        for phase in saved_game['phases']:
            phase['state']['board_state'] = dict_to_flatten_board_state(phase['state'], map_object)
            if phase['name'][-1] == 'M':
                phase['prev_orders_state'] = dict_to_flatten_prev_orders_state(phase, map_object)
    return saved_game

def add_rewards_to_saved_game_proto(saved_game_proto, reward_fn):
    """ Adds a cached list of rewards for each power in the game according the the reward fn """
    if reward_fn is not None and saved_game_proto.map.startswith('standard'):
        saved_game_proto.reward_fn = reward_fn.name
        for power_name in ALL_POWERS:
            power_rewards = reward_fn.get_episode_rewards(saved_game_proto, power_name)
            saved_game_proto.rewards[power_name].value.extend(power_rewards)
    return saved_game_proto

def process_game(line):
    """ Process a line in the .jsonl file
        :return: A tuple (game_id, saved_game_zlib)
    """
    if not line:
        return None, None
    saved_game = json.loads(line)
    saved_game = add_cached_states_to_saved_game(saved_game)
    saved_game = add_possible_orders_to_saved_game(saved_game)
    saved_game_proto = dict_to_proto(saved_game, SavedGameProto)
    saved_game_proto = add_rewards_to_saved_game_proto(saved_game_proto, DefaultRewardFunction())
    saved_game_zlib = proto_to_zlib(saved_game_proto)
    return saved_game['id'], saved_game_zlib

def build():
    """ Building the hdf5 dataset """
    if not os.path.exists(ZIP_DATASET_PATH):
        raise RuntimeError('Unable to find the zip dataset at %s' % ZIP_DATASET_PATH)

    # Extracting
    extract_dir = os.path.join(os.path.dirname(ZIP_DATASET_PATH), 'zip_dataset')
    if not os.path.exists(extract_dir):
        LOGGER.info('... Extracting files from zip dataset.')
        with zipfile.ZipFile(ZIP_DATASET_PATH, 'r') as zip_dataset:
            zip_dataset.extractall(extract_dir)

    # Additional information we also want to store
    map_object = Map()
    all_powers = get_map_powers(map_object)
    sc_to_win = len(map_object.scs) // 2 + 1

    hash_table = {}                                         # zobrist_hash: [{game_id}/{phase_name}]
    moves = {}                                              # Moves frequency: {move: [nb_no_press, nb_press]}
    nb_phases = OrderedDict()                               # Nb of phases per game
    end_scs = {'press': {power_name: {nb_sc: [] for nb_sc in range(0, sc_to_win + 1)} for power_name in all_powers},
               'no_press': {power_name: {nb_sc: [] for nb_sc in range(0, sc_to_win + 1)} for power_name in all_powers}}

    # Building
    dataset_index = {}
    LOGGER.info('... Building HDF5 dataset.')
    with multiprocessing.Pool() as pool:
        with h5py.File(DATASET_PATH, 'w') as hdf5_dataset, open(PROTO_DATASET_PATH, 'wb') as proto_dataset:

            for json_file_path in glob.glob(extract_dir + '/*.jsonl'):
                LOGGER.info('... Processing: %s', json_file_path)
                category = json_file_path.split('/')[-1].split('.')[0]
                dataset_index[category] = set()

                # Processing file using pool
                with open(json_file_path, 'r') as json_file:
                    lines = json_file.read().splitlines()
                    for game_id, saved_game_zlib in tqdm(pool.imap_unordered(process_game, lines), total=len(lines)):
                        if game_id is None:
                            continue
                        saved_game_proto = zlib_to_proto(saved_game_zlib, SavedGameProto)

                        # Saving to disk
                        hdf5_dataset[game_id] = np.void(saved_game_zlib)
                        write_proto_to_file(proto_dataset, saved_game_proto, compressed=False)
                        dataset_index[category].add(game_id)

                        # Recording additional info
                        get_end_scs_info(saved_game_proto, game_id, all_powers, sc_to_win, end_scs)
                        get_moves_info(saved_game_proto, moves)
                        nb_phases[game_id] = len(saved_game_proto.phases)

                        # Recording hash of each phase
                        for phase in saved_game_proto.phases:
                            hash_table.setdefault(phase.state.zobrist_hash, [])
                            hash_table[phase.state.zobrist_hash] += ['%s/%s' % (game_id, phase.name)]

    # Storing info to disk
    with open(DATASET_INDEX_PATH, 'wb') as file:
        pickle.dump(dataset_index, file, pickle.HIGHEST_PROTOCOL)
    with open(END_SCS_DATASET_PATH, 'wb') as file:
        pickle.dump(end_scs, file, pickle.HIGHEST_PROTOCOL)
    with open(HASH_DATASET_PATH, 'wb') as file:
        pickle.dump(hash_table, file, pickle.HIGHEST_PROTOCOL)
    with open(MOVES_COUNT_DATASET_PATH, 'wb') as file:
        pickle.dump(moves, file, pickle.HIGHEST_PROTOCOL)
    with open(PHASES_COUNT_DATASET_PATH, 'wb') as file:
        pickle.dump(nb_phases, file, pickle.HIGHEST_PROTOCOL)

    # Deleting extract_dir
    LOGGER.info('... Deleting extracted files.')
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    LOGGER.info('... Done building HDF5 dataset.')


def get_end_scs_info(saved_game_proto, game_id, all_powers, sc_to_win, end_scs):
    """ Records the ending supply center information """
    # Only keeping games with the standard map (or its variations) that don't use BUILD_ANY
    if not saved_game_proto.map.startswith('standard'):
        return
    if 'BUILD_ANY' in saved_game_proto.rules:
        return

    # Detecting if no press
    is_no_press = 'NO_PRESS' in saved_game_proto.rules

    # Counting the nb of ending scs for each power
    for power_name in all_powers:
        nb_sc = min(sc_to_win, len(saved_game_proto.phases[-1].state.centers[power_name].value))
        if is_no_press:
            end_scs['no_press'][power_name][nb_sc] += [game_id]
        else:
            end_scs['press'][power_name][nb_sc] += [game_id]

def get_moves_info(saved_game_proto, moves):
    """ Recording the frequency of each order """
    # Only keeping games with the standard map (or its variations)
    if not saved_game_proto.map.startswith('standard'):
        return

    # Detecting if no press
    is_no_press = 'NO_PRESS' in saved_game_proto.rules

    # Counting all orders
    for phase in saved_game_proto.phases:
        for power_name in phase.orders:
            for order in phase.orders[power_name].value:
                moves.setdefault(order, [0, 0])
                moves[order][is_no_press] += 1
