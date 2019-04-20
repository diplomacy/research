#!/usr/bin/env python3
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
""" Opening moves
    - Loops over the dataset and displays the opening database on the standard map (press and no press combined)
"""
import logging
import os
import pickle
from tqdm import tqdm
from diplomacy import Map
from diplomacy_research.models.state_space import get_map_powers
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import read_next_proto
from diplomacy_research.settings import PROTO_DATASET_PATH, PHASES_COUNT_DATASET_PATH

# Constants
LOGGER = logging.getLogger('diplomacy_research.scripts.opening_moves')

# -----------------------------------------------
# ---------       OPENING MOVES        ----------
# -----------------------------------------------
def show_opening_moves_data(proto_dataset_path):
    """ Displays a list of opening moves for each power on the standard map
        :param proto_dataset_path: The path to the proto dataset
        :return: Nothing
    """
    if not os.path.exists(proto_dataset_path):
        raise RuntimeError('Unable to find Diplomacy dataset at {}'.format(proto_dataset_path))

    # Openings dict
    map_object = Map('standard')
    openings = {power_name: {} for power_name in get_map_powers(map_object)}

    # Loading the phases count dataset to get the number of games
    total = None
    if os.path.exists(PHASES_COUNT_DATASET_PATH):
        with open(PHASES_COUNT_DATASET_PATH, 'rb') as file:
            total = len(pickle.load(file))
    progress_bar = tqdm(total=total)

    # Loading dataset and building database
    LOGGER.info('... Building an opening move database.')
    with open(PROTO_DATASET_PATH, 'rb') as proto_dataset:

        # Reading games
        while True:
            saved_game_proto = read_next_proto(SavedGameProto, proto_dataset, compressed=False)
            if saved_game_proto is None:
                break
            progress_bar.update(1)

            # Only keeping games with the standard map (or its variations)
            if not saved_game_proto.map.startswith('standard'):
                continue

            initial_phase = saved_game_proto.phases[0]
            for power_name in initial_phase.orders:
                orders = initial_phase.orders[power_name].value
                orders = tuple(sorted(orders, key=lambda order: order.split()[1]))      # Sorted by location
                if orders not in openings[power_name]:
                    openings[power_name][orders] = 0
                openings[power_name][orders] += 1

    # Printing results
    for power_name in get_map_powers(map_object):
        print('=' * 80)
        print(power_name)
        print()
        for opening, count in sorted(openings[power_name].items(), key=lambda item: item[1], reverse=True):
            print(opening, 'Count:', count)

    # Closing
    progress_bar.close()

if __name__ == '__main__':
    show_opening_moves_data(PROTO_DATASET_PATH)
