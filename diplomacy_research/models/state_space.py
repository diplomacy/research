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
""" State Space
    - Functions to convert a game, state, or phase to state_space
    - Functions to build the adjacency matrix
    - List of words in standard vocabulary with a function to convert to/from index.
"""
# pylint: disable=too-many-lines
import base64
from collections import OrderedDict
import hashlib
import logging
from operator import itemgetter
import os
import pickle
from random import shuffle
import zlib
import numpy as np
from diplomacy import Game
from diplomacy import Map
from diplomacy_research.proto.diplomacy_proto.common_pb2 import MapStringList
from diplomacy_research.proto.diplomacy_proto.game_pb2 import State as StateProto, PhaseHistory as PhaseHistoryProto
from diplomacy_research.utils.proto import dict_to_proto
from diplomacy_research.settings import MOVES_COUNT_DATASET_PATH

# Constants
LOGGER = logging.getLogger(__name__)
ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

# ====================================
# STATE SPACE   (Supervised Learning)
# ====================================
#
# 75 provinces + 6 coasts (BUL/EC, BUL/SC, SPA/NC, SPA/NC, STP/NC, STP/NC) = 81 nodes
# Note: There is actually 76 provinces, but we excluded SWI (Switzerland) because it is impassable in the game
# Information recorded per node
# -------------------------------
# Normal Unit Type (3 values): {A, F, None}
# Normal Unit Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
# Buildable / Removable (2 values): {Buildable,Removable}  (0 or 1)
# Dislodged Unit Type (3 values): {A, F, None}
# Dislodged Unit Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
# Area Type (3 values): {LAND, WATER, COAST}
# SC Owning Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
#                       Note: Locations that are not supply centers will be all 0.
#
# State Space Size: 81 x 35
#
# ====================================
# PREV ORDERS SPACE   (Supervised Learning)
# ====================================
#
# Note: Only used for Movement phases
# 81 locs (nodes) - Same as state space
#
# Hold:         A PAR H                 -- Issuing: PAR     Src: None       Dest; None
# Support:      A PAR S A MAR           -- Issuing: PAR     Src: MAR        Dest: None
# Support:      A PAR S A MAR - BUR     -- Issuing: PAR     Src: MAR        Dest: BUR
# Move          A PAR - MAR             -- Issuing: PAR     Src: None       Dest: MAR
# Convoy        F MAO C A PAR - LON     -- Issuing: MAO     Src: PAR        Dest: LON
# -------------------------------
#
# Unit Type (3 values): {A, F, None}
# Issuing Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
# Order Type (5 values): {HOLD, MOVE, SUPPORT, CONVOY, None}
# Src Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
# Dest Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
# SC Owning Power (8 values): {AUS, ENG, FRA, GER, ITA, RUS, TUR, None}
#                       Note: Locations that are not supply centers will be all 0.
#
# State Space Size: 81 x 40
#

# Cache for adjacency matrix and sorted locs
ADJACENCY_MATRIX = {}
SORTED_LOCS = {}

# Constants
NB_LOCS = 81
NB_SUPPLY_CENTERS = 34
NB_FEATURES = 35
NB_ORDERS_FEATURES = 40
NB_POWERS = 7
NB_SEASONS = 3
NB_TYPES = 2
NB_NODES = NB_LOCS
TOKENS_PER_ORDER = 5
MAX_LENGTH_ORDER_PREV_PHASES = 350
MAX_CANDIDATES = 240
NB_PREV_ORDERS = 1                  # We only feed the last movement phase
NB_PREV_ORDERS_HISTORY = 3          # We need to have an history of at least 3, to get at least 1 movement phase

def get_sorted_locs(map_object):
    """ Returns the list of locations for the given map in sorted order, using topological order
        :param map_object: The instantiated map
        :return: A sorted list of locations
        :type map_object: diplomacy.Map
    """
    if map_object.name not in SORTED_LOCS:
        key = None if not map_object.name.startswith('standard') else STANDARD_TOPO_LOCS.index
        locs = [l.upper() for l in map_object.locs if map_object.area_type(l) != 'SHUT']
        SORTED_LOCS[map_object.name] = sorted(locs, key=key)
    return SORTED_LOCS[map_object.name]

def get_player_seed(game_id, power_name):
    """ Returns a unique player seed given a game id and a power name """
    crc_adler32 = zlib.adler32(game_id.encode('utf-8') + power_name.encode('utf-8'))
    sha256_hash = hashlib.sha256()
    sha256_hash.update(crc_adler32.to_bytes((crc_adler32.bit_length() + 7) // 8, 'big'))
    sha256_hash.update(game_id.encode('utf-8'))
    return int.from_bytes(sha256_hash.digest()[8:12], 'big') % (2 ** 31 - 1)

def get_game_id(salt='', previous_id=''):
    """ Standardizes the game id format to 72 bits (12 base64 characters)
        :param salt: Optional salt. To hash a known string into a deterministic game id.
        :param previous_id: Optional. To hash a known string into a deterministic game id.
    """
    sha256_hash = hashlib.sha256()
    if salt and previous_id:
        sha256_hash.update(salt.encode('utf-8'))
        sha256_hash.update(previous_id.encode('utf-8'))
    else:
        sha256_hash.update(os.urandom(32))
    return base64.b64encode(sha256_hash.digest()[-12:], b'-_').decode('utf-8')

def build_game_from_state_proto(state_proto):
    """ Builds a game object from a state_proto """
    game = Game(map_name=state_proto.map, rules=state_proto.rules)
    game.set_current_phase(state_proto.name)

    # Setting units
    game.clear_units()
    for power_name in state_proto.units:
        game.set_units(power_name, list(state_proto.units[power_name].value))

    # Setting centers
    game.clear_centers()
    for power_name in state_proto.centers:
        game.set_centers(power_name, list(state_proto.centers[power_name].value))

    # Returning
    return game

def extract_state_proto(game):
    """ Extracts the state_proto from a diplomacy.Game object
        :type game: diplomacy.Game
    """
    state = game.get_state()
    state['game_id'] = game.game_id
    state['map'] = game.map.name
    state['rules'] = list(game.rules)
    return dict_to_proto(state, StateProto)

def extract_phase_history_proto(game, nb_previous_phases=NB_PREV_ORDERS_HISTORY):
    """ Extracts the phase_history_proto from a diplomacy.Game object
        :param game: The diplomacy.Game object
        :param nb_previous_phases: Integer. If set, only the last x phases will be returned.
                                            If None, the full history since the beginning of the game is returned.
        :return: A list of `.proto.game.PhaseHistory` proto.
        :type game: diplomacy.Game
    """
    from_phase = None if nb_previous_phases is None else -1 * nb_previous_phases
    phase_history = Game.get_phase_history(game, from_phase=from_phase)
    return [dict_to_proto(hist.to_dict(), PhaseHistoryProto) for hist in phase_history]

def extract_possible_orders_proto(game):
    """ Extracts the possible_orders_proto from a diplomacy.Game object
        :type game: diplomacy.Game
    """
    possible_orders = game.get_all_possible_orders()
    return dict_to_proto(possible_orders, MapStringList)

def dict_to_flatten_board_state(state, map_object):
    """ Converts a game state to its flatten (list) board state representation.
        :param state: A game state.
        :param map_object: The instantiated Map object
        :return: A flatten (list) representation of the phase (81*35 = 2835)
    """
    state_proto = dict_to_proto(state, StateProto)
    return proto_to_board_state(state_proto, map_object).flatten().tolist()

def proto_to_board_state(state_proto, map_object):
    """ Converts a `.proto.game.State` proto to its matrix board state representation
        :param state_proto: A `.proto.game.State` proto of the state of the game.
        :param map_object: The instantiated Map object
        :return: The board state (matrix representation) of the phase (81 x 35)
        :type map_object: diplomacy.Map
    """
    # Retrieving cached version directly from proto
    if state_proto.board_state:
        if len(state_proto.board_state) == NB_NODES * NB_FEATURES:
            return np.array(state_proto.board_state, dtype=np.uint8).reshape(NB_NODES, NB_FEATURES)
        LOGGER.warning('Got a cached board state of dimension %d - Expected %d',
                       len(state_proto.board_state), NB_NODES * NB_FEATURES)

    # Otherwise, computing it
    locs = get_sorted_locs(map_object)
    scs = sorted([sc.upper() for sc in map_object.scs])
    powers = get_map_powers(map_object)
    remaining_scs = scs[:]

    # Sizes
    nb_locs = len(locs)
    nb_powers = len(powers)

    # Creating matrix components for locations
    loc_norm_type_matrix = np.zeros((nb_locs, NB_TYPES + 1), dtype=np.uint8)
    loc_norm_power_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)
    loc_build_removable_matrix = np.zeros((nb_locs, 2), dtype=np.uint8)
    loc_dis_type_matrix = np.zeros((nb_locs, NB_TYPES + 1), dtype=np.uint8)
    loc_dis_power_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)
    loc_area_type_matrix = np.zeros((nb_locs, 3), dtype=np.uint8)
    loc_owner = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)

    # Settings units
    for power_name in state_proto.units:
        build_count = state_proto.builds[power_name].count

        # Marking regular and removable units
        for unit in state_proto.units[power_name].value:
            # Checking in what phase we are in
            is_dislodged = bool(unit[0] == '*')

            # Removing leading * if dislodged
            unit = unit[1:] if is_dislodged else unit
            loc = unit[2:]
            unit_type = unit[0]
            loc_ix = locs.index(loc)

            # Calculating unit owner ix and unit type ix
            power_ix = powers.index(power_name)
            type_ix = 0 if unit_type == 'A' else 1
            if not is_dislodged:
                loc_norm_power_matrix[loc_ix, power_ix] = 1
                loc_norm_type_matrix[loc_ix, type_ix] = 1
            else:
                loc_dis_power_matrix[loc_ix, power_ix] = 1
                loc_dis_type_matrix[loc_ix, type_ix] = 1

            # Setting number of removable units
            if build_count < 0:
                loc_build_removable_matrix[loc_ix, 1] = 1

            # Also setting the parent location if it's a coast
            if '/' in loc:
                loc_without_coast = loc[:3]
                loc_without_coast_ix = locs.index(loc_without_coast)
                if not is_dislodged:
                    loc_norm_power_matrix[loc_without_coast_ix, power_ix] = 1
                    loc_norm_type_matrix[loc_without_coast_ix, type_ix] = 1
                else:
                    loc_dis_power_matrix[loc_without_coast_ix, power_ix] = 1
                    loc_dis_type_matrix[loc_without_coast_ix, type_ix] = 1
                if build_count < 0:
                    loc_build_removable_matrix[loc_without_coast_ix, 1] = 1

        # Handling build locations
        if build_count > 0:
            buildable_locs = [loc for loc in locs if loc[:3] in state_proto.builds[power_name].homes]

            # Marking location as buildable (with no units on it)
            for loc in buildable_locs:
                loc_ix = locs.index(loc)

                # There are no units on it, so "Normal unit" is None
                loc_norm_type_matrix[loc_ix, -1] = 1
                loc_norm_power_matrix[loc_ix, -1] = 1
                loc_build_removable_matrix[loc_ix, 0] = 1

    # Setting rows with no values to None
    loc_norm_type_matrix[(np.sum(loc_norm_type_matrix, axis=1) == 0, -1)] = 1
    loc_norm_power_matrix[(np.sum(loc_norm_power_matrix, axis=1) == 0, -1)] = 1
    loc_dis_type_matrix[(np.sum(loc_dis_type_matrix, axis=1) == 0, -1)] = 1
    loc_dis_power_matrix[(np.sum(loc_dis_power_matrix, axis=1) == 0, -1)] = 1

    # Setting area type
    for loc in locs:
        loc_ix = locs.index(loc)
        area_type = map_object.area_type(loc)
        if area_type in ['PORT', 'COAST']:
            area_type_ix = 2
        elif area_type == 'WATER':
            area_type_ix = 1
        elif area_type == 'LAND':
            area_type_ix = 0
        else:
            raise RuntimeError('Unknown area type {}'.format(area_type))
        loc_area_type_matrix[loc_ix, area_type_ix] = 1

    # Supply center ownership
    for power_name in state_proto.centers:
        if power_name == 'UNOWNED':
            continue
        for center in state_proto.centers[power_name].value:
            for loc in [map_loc for map_loc in locs if map_loc[:3] == center[:3]]:
                if loc[:3] in remaining_scs:
                    remaining_scs.remove(loc[:3])
                loc_ix = locs.index(loc)
                power_ix = powers.index(power_name)
                loc_owner[loc_ix, power_ix] = 1

    # Unowned supply centers
    for center in remaining_scs:
        for loc in [map_loc for map_loc in locs if map_loc[:3] == center[:3]]:
            loc_ix = locs.index(loc)
            power_ix = nb_powers
            loc_owner[loc_ix, power_ix] = 1

    # Merging and returning
    return np.concatenate([loc_norm_type_matrix,
                           loc_norm_power_matrix,
                           loc_build_removable_matrix,
                           loc_dis_type_matrix,
                           loc_dis_power_matrix,
                           loc_area_type_matrix,
                           loc_owner],
                          axis=1)

def dict_to_flatten_prev_orders_state(phase, map_object):
    """ Converts a phase to its flatten (list) prev orders state representation.
        :param phase: A phase from a saved game.
        :param map_object: The instantiated Map object
        :return: A flatten (list) representation of the prev orders (81*40 = 3240)
    """
    phase_proto = dict_to_proto(phase, PhaseHistoryProto)
    return proto_to_prev_orders_state(phase_proto, map_object).flatten().tolist()

def proto_to_prev_orders_state(phase_proto, map_object):
    """ Converts a `.proto.game.PhaseHistory` proto to its matrix prev orders state representation
        :param phase_proto: A `.proto.game.PhaseHistory` proto of the phase history of the game.
        :param map_object: The instantiated Map object
        :return: The prev orders state (matrix representation) of the prev orders (81 x 40)
        :type map_object: diplomacy.Map
    """
    # Retrieving cached version directly from proto
    if phase_proto.prev_orders_state:
        if len(phase_proto.prev_orders_state) == NB_NODES * NB_ORDERS_FEATURES:
            return np.array(phase_proto.prev_orders_state, dtype=np.uint8).reshape(NB_NODES, NB_ORDERS_FEATURES)
        LOGGER.warning('Got a cached prev orders state of dimension %d - Expected %d',
                       len(phase_proto.prev_orders_state), NB_NODES * NB_ORDERS_FEATURES)

    # Otherwise, computing it
    locs = get_sorted_locs(map_object)
    powers = get_map_powers(map_object)
    scs = sorted([sc.upper() for sc in map_object.scs])

    # Sizes
    nb_locs = len(locs)
    nb_powers = len(powers)

    # Not a movement phase
    if phase_proto.name[-1] != 'M':
        LOGGER.warning('Trying to compute the prev_orders_state of a non-movement phase.')
        return np.zeros((NB_NODES, NB_FEATURES), dtype=np.uint8)

    # Creating matrix components for locations
    loc_unit_type_matrix = np.zeros((nb_locs, NB_TYPES + 1), dtype=np.uint8)
    loc_issuing_power_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)
    loc_order_type_matrix = np.zeros((nb_locs, 5), dtype=np.uint8)
    loc_src_power_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)
    loc_dest_power_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)
    loc_sc_owner_matrix = np.zeros((nb_locs, nb_powers + 1), dtype=np.uint8)

    # Storing the owners of each location
    # The owner of a location is the unit owner if there is a unit, otherwise the SC owner, otherwise None
    owner = {}
    for power_name in phase_proto.state.units:
        for unit in phase_proto.state.units[power_name].value:
            loc = unit.split()[-1]
            owner[loc[:3]] = power_name

    # Storing the owners of each center
    remaining_scs = scs[:]
    for power_name in phase_proto.state.centers:
        if power_name == 'UNOWNED':
            continue
        for center in phase_proto.state.centers[power_name].value:
            for loc in [map_loc for map_loc in locs if map_loc[:3] == center[:3]]:
                if loc[:3] not in owner:
                    owner[loc[:3]] = power_name
                loc_sc_owner_matrix[locs.index(loc), powers.index(power_name)] = 1
            remaining_scs.remove(center)
    for center in remaining_scs:
        for loc in [map_loc for map_loc in locs if map_loc[:3] == center[:3]]:
            loc_sc_owner_matrix[locs.index(loc), -1] = 1

    # Parsing each order
    for issuing_power_name in phase_proto.orders:
        issuing_power_ix = powers.index(issuing_power_name)

        for order in phase_proto.orders[issuing_power_name].value:
            word = order.split()

            # Movement phase - Expecting Hold, Move, Support or Convoy
            if len(word) <= 2 or word[2] not in 'H-SC':
                LOGGER.warning('Unsupported order %s', order)
                continue

            # Detecting unit type, loc and order type
            unit_type, unit_loc, order_type = word[:3]
            unit_type_ix = 0 if unit_type == 'A' else 1
            order_type_ix = 'H-SC'.index(order_type)

            # Adding both with and without coasts
            unit_locs = [unit_loc]
            if '/' in unit_loc:
                unit_locs += [unit_loc[:3]]

            for unit_loc in unit_locs:
                unit_loc_ix = locs.index(unit_loc)

                # Setting unit type, loc, order type
                loc_unit_type_matrix[unit_loc_ix, unit_type_ix] = 1
                loc_issuing_power_matrix[unit_loc_ix, issuing_power_ix] = 1
                loc_order_type_matrix[unit_loc_ix, order_type_ix] = 1

                # Hold order
                if order_type == 'H':
                    loc_src_power_matrix[unit_loc_ix, -1] = 1
                    loc_dest_power_matrix[unit_loc_ix, -1] = 1

                # Move order
                elif order_type == '-':
                    dest = word[-1]
                    dest_power_ix = -1 if dest[:3] not in owner else powers.index(owner[dest[:3]])
                    loc_src_power_matrix[unit_loc_ix, -1] = 1
                    loc_dest_power_matrix[unit_loc_ix, dest_power_ix] = 1

                # Support hold
                elif order_type == 'S' and '-' not in word:
                    src = word[-1]
                    src_power_ix = -1 if src[:3] not in owner else powers.index(owner[src[:3]])
                    loc_src_power_matrix[unit_loc_ix, src_power_ix] = 1
                    loc_dest_power_matrix[unit_loc_ix, -1] = 1

                # Support move / Convoy
                elif order_type in ('S', 'C') and '-' in word:
                    src = word[word.index('-') - 1]
                    dest = word[-1]
                    src_power_ix = -1 if src[:3] not in owner else powers.index(owner[src[:3]])
                    dest_power_ix = -1 if dest[:3] not in owner else powers.index(owner[dest[:3]])
                    loc_src_power_matrix[unit_loc_ix, src_power_ix] = 1
                    loc_dest_power_matrix[unit_loc_ix, dest_power_ix] = 1

                # Unknown other
                else:
                    LOGGER.error('Unsupported order - %s', order)

    # Setting rows with no values to None
    loc_unit_type_matrix[(np.sum(loc_unit_type_matrix, axis=1) == 0, -1)] = 1
    loc_issuing_power_matrix[(np.sum(loc_issuing_power_matrix, axis=1) == 0, -1)] = 1
    loc_order_type_matrix[(np.sum(loc_order_type_matrix, axis=1) == 0, -1)] = 1
    loc_src_power_matrix[(np.sum(loc_src_power_matrix, axis=1) == 0, -1)] = 1
    loc_dest_power_matrix[(np.sum(loc_dest_power_matrix, axis=1) == 0, -1)] = 1

    # Adding prev order state at the beginning of the list (to keep the phases in the correct order)
    return np.concatenate([loc_unit_type_matrix,
                           loc_issuing_power_matrix,
                           loc_order_type_matrix,
                           loc_src_power_matrix,
                           loc_dest_power_matrix,
                           loc_sc_owner_matrix,], axis=1)

def get_top_victors(saved_game_proto, map_object):
    """ Returns a list of the top victors (i.e. owning more than 25% -1 of the centers on the map)
        We will only used the orders from these victors for the supervised learning
        :param saved_game_proto: A `.proto.game.SavedGame` object from the dataset.
        :param map_object: The instantiated Map object
        :return: A list of victors (powers)
        :type map_object: diplomacy.Map
    """
    powers = get_map_powers(map_object)
    nb_scs = len(map_object.scs)
    min_nb_scs = nb_scs // 4 - 1

    # Retrieving the number of centers for each power at the end of the game
    # Only keeping powers with at least 7 centers
    scs_last_phase = saved_game_proto.phases[-1].state.centers
    ending_scs = [(power_name, len(scs_last_phase[power_name].value)) for power_name in powers
                  if len(scs_last_phase[power_name].value) >= min_nb_scs]
    ending_scs = sorted(ending_scs, key=itemgetter(1), reverse=True)

    # Not victors found, returning all powers because they all are of similar strength
    if not ending_scs:
        return powers
    return [power_name for power_name, _ in ending_scs]

def get_orderable_locs_for_powers(state_proto, powers, shuffled=False):
    """ Returns a list of all orderable locations and a list of orderable location for each power
        :param state_proto: A `.proto.game.State` object.
        :param powers: A list of powers for which to retrieve the orderable locations
        :param shuffled: Boolean. If true, the orderable locations for each power will be shuffled.
        :return: A tuple consisting of:
            - A list of all orderable locations for all powers
            - A dictionary with the power name as key, and a set of its orderable locations as value
    """
    # Detecting if we are in retreats phase
    in_retreats_phase = state_proto.name[-1] == 'R'

    # Detecting orderable locations for each top victor
    # Not storing coasts for orderable locations
    all_orderable_locs = set()
    orderable_locs = {power_name: set() for power_name in powers}
    for power_name in powers:

        # Adding build locations
        if state_proto.name[-1] == 'A' and state_proto.builds[power_name].count >= 0:
            for home in state_proto.builds[power_name].homes:
                all_orderable_locs.add(home[:3])
                orderable_locs[power_name].add(home[:3])

        # Otherwise, adding units (regular and dislodged)
        else:
            for unit in state_proto.units[power_name].value:
                unit_type, unit_loc = unit.split()
                if unit_type[0] == '*' and in_retreats_phase:
                    all_orderable_locs.add(unit_loc[:3])
                    orderable_locs[power_name].add(unit_loc[:3])
                elif unit_type[0] != '*' and not in_retreats_phase:
                    all_orderable_locs.add(unit_loc[:3])
                    orderable_locs[power_name].add(unit_loc[:3])

    # Sorting orderable locations
    key = None if not state_proto.map.startswith('standard') else STANDARD_TOPO_LOCS.index
    orderable_locs = {power_name: sorted(orderable_locs[power_name], key=key) for power_name in powers}

    # Shuffling list if requested
    if shuffled:
        for power_name in powers:
            shuffle(orderable_locs[power_name])

    # Returning
    return list(sorted(all_orderable_locs, key=key)), orderable_locs

def get_orders_by_loc(phase_proto, orderable_locations, powers):
    """ Returns a dictionary with loc as key and its corresponding order as value
        Note: only the locs in orderable_locations are included in the dictionary

        :param phase_proto: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param orderable_locations: A list of locs from which we want the orders
        :param powers: A list of powers for which to retrieve orders
        :return: A dictionary with locs as key, and their corresponding order as value
               (e.g. {'PAR': 'A PAR - MAR'})
    """
    orders_by_loc = {}
    for power_name in phase_proto.orders:
        if power_name not in powers:
            continue
        for order in phase_proto.orders[power_name].value:
            order_loc = order.split()[1]

            # Skipping if not one of the orderable locations
            if order_loc not in orderable_locations and order_loc[:3] not in orderable_locations:
                continue

            # Adding order to dictionary
            # Removing coast from key
            orders_by_loc[order_loc[:3]] = order

    # Returning order by location
    return orders_by_loc

def get_issued_orders_for_powers(phase_proto, powers, shuffled=False):
    """ Extracts a list of orderable locations and corresponding orders for a list of powers
        :param phase_proto: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param powers: A list of powers for which we want issued orders
        :param shuffled: Boolean. If true, orderable locations are shuffled, otherwise they are sorted alphabetically
        :return: A dictionary with the power name as key, and for value a dictionary of:
                    - orderable location for that power as key (e.g. 'PAR')
                    - the corresponding order at that location as value (e.g. 'A PAR H')
    """
    # Retrieving orderable locations
    all_orderable_locs, orderable_locs = get_orderable_locs_for_powers(phase_proto.state,
                                                                       powers,
                                                                       shuffled=shuffled)

    # Retrieving orders by loc for orderable locations
    orders_by_loc = get_orders_by_loc(phase_proto, all_orderable_locs, powers)

    # Computing list of issued orders for each top victor
    issued_orders = OrderedDict()
    for power_name in powers:
        issued_orders[power_name] = OrderedDict()
        for loc in orderable_locs[power_name]:
            for issued_order_loc in [order_loc for order_loc in orders_by_loc
                                     if order_loc == loc or (order_loc[:3] == loc[:3] and '/' in order_loc)]:
                issued_orders[power_name][issued_order_loc] = orders_by_loc[issued_order_loc]

    # Returning
    return issued_orders

def get_possible_orders_for_powers(phase_proto, powers):
    """ Extracts a list of possible orders for all locations where a power could issue an order
        :param phase_proto: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param powers: A list of powers for which we want the possible orders
        :return: A dictionary for each location, the list of possible orders
    """
    possible_orders_proto = phase_proto.possible_orders

    # Making sure we have a list of possible orders attached to the phase_proto
    # Otherwise, creating a game object to retrieve the possible orders
    if not possible_orders_proto:
        LOGGER.warning('The list of possible orders was not attached to the phase_proto. Generating it.')
        game = build_game_from_state_proto(phase_proto.state)
        possible_orders_proto = extract_possible_orders_proto(game)
        for loc in possible_orders_proto:
            phase_proto.possible_orders[loc].value.extend(possible_orders_proto[loc].value)

    # Getting orderable locations
    all_orderable_locations, _ = get_orderable_locs_for_powers(phase_proto.state, powers)

    # Getting possible orders
    possible_orders = {}
    for loc in all_orderable_locations:
        possible_orders_at_loc = list(possible_orders_proto[loc].value)
        if possible_orders_at_loc:
            possible_orders[loc] = possible_orders_at_loc

    # Returning
    return possible_orders

def get_map_powers(map_object):
    """ Returns the list of powers on the map """
    if map_object.name.startswith('standard'):
        return ALL_POWERS
    return sorted([power_name for power_name in map_object.powers])

def get_current_season(state_proto):
    """ Returns the index of the current season (0 = S, 1 = F, 2 = W)
        :param state_proto: A `.proto.game.State` object.
        :return: The integer representation of the current season
    """
    season = state_proto.name
    if season == 'COMPLETED':
        return 0
    if season[0] not in 'SFW':
        LOGGER.warning('Unrecognized season %s. Using "Spring" as the current season.', season)
        return 0
    return 'SFW'.index(season[0])

def get_adjacency_matrix(map_name='standard'):
    """ Computes the adjacency matrix for map
        :param map_name: The name of the map
        :return: A (nb_nodes) x (nb_nodes) matrix
    """
    if map_name in ADJACENCY_MATRIX:
        return ADJACENCY_MATRIX[map_name]

    # Finding list of all locations
    current_map = Map(map_name)
    locs = get_sorted_locs(current_map)
    adjacencies = np.zeros((len(locs), len(locs)), dtype=np.bool)

    # Building adjacencies between locs
    # Coasts are adjacent to their parent location (without coasts)
    for i, loc_1 in enumerate(locs):
        for j, loc_2 in enumerate(locs):
            if current_map.abuts('A', loc_1, '-', loc_2) or current_map.abuts('F', loc_1, '-', loc_2):
                adjacencies[i, j] = 1
            if loc_1 != loc_2 and (loc_1[:3] == loc_2 or loc_1 == loc_2[:3]):
                adjacencies[i, j] = 1

    # Storing in cache and returning
    ADJACENCY_MATRIX[map_name] = adjacencies
    return adjacencies

def get_board_alignments(locs, in_adjustment_phase, tokens_per_loc, decoder_length):
    """ Returns a n list of (NB_NODES vector) representing the alignments (probs) for the locs on the board state
        :param locs: The list of locs being outputted by the model
        :param in_adjustment_phase: Indicates if we are in A phase (all locs possible at every position) or not.
        :param tokens_per_loc: The number of tokens per loc (TOKENS_PER_ORDER for token_based, 1 for order_based).
        :param decoder_length: The length of the decoder.
        :return: A list of [NB_NODES] vector of probabilities (alignments for each location)
    """
    alignments = []

    # Regular phase
    if not in_adjustment_phase:
        for loc in locs:
            alignment = np.zeros([NB_NODES], dtype=np.uint8)
            alignment_index = ALIGNMENTS_INDEX.get(loc[:3], [])
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning('Location %s is not in the alignments index.', loc)
            if alignment_index:
                for index in alignment_index:
                    alignment[index] = 1
            alignments += [alignment] * tokens_per_loc
        if decoder_length != len(locs) * tokens_per_loc:
            LOGGER.warning('Got %d tokens, but decoder length is %d', len(locs) * tokens_per_loc, decoder_length)
        if decoder_length > len(alignments):
            LOGGER.warning('Got %d locs, but the decoder length is %d', len(locs), decoder_length)
            alignments += [np.zeros([NB_NODES], dtype=np.uint8)] * (decoder_length - len(alignments))

    # Adjustment phase (All locs at all positions)
    else:
        alignment = np.zeros([NB_NODES], dtype=np.uint8)
        alignment_index = set()
        for loc in locs:
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning('Location %s is not in the alignments index.', loc)
            for index in ALIGNMENTS_INDEX.get(loc[:3], []):
                alignment_index.add(index)
        if alignment_index:
            for index in alignment_index:
                alignment[index] = 1
        alignments = [alignment] * decoder_length

    # Validating size
    if decoder_length != len(alignments):
        LOGGER.warning('Got %d alignments, but decoder length is %d', len(alignments), decoder_length)

    # Returning
    return alignments

def get_alignments_index(map_name='standard'):
    """ Computes a list of nodes index for each possible location
        e.g. if the sorted list of locs is ['BRE', 'MAR', 'PAR'] would return {'BRE': [0], 'MAR': [1], 'PAR': [2]}
    """
    current_map = Map(map_name)
    sorted_locs = get_sorted_locs(current_map)
    alignments_index = {}

    # Computing the index of each loc
    for loc in sorted_locs:
        if loc[:3] in alignments_index:
            continue
        alignments_index[loc[:3]] = [index for index, sorted_loc in enumerate(sorted_locs) if loc[:3] == sorted_loc[:3]]
    return alignments_index

def get_token_based_mask(list_possible_orders, offset=0, coords=None):
    """ Computes the possible order mask to apply to the decoder of a token-based policy model
        :param list_possible_orders: The list of possible orders (e.g. ['A PAR H', 'A PAR - BUR', ...])
        :param offset: An integer offset to add to the position before storing the coords
        :param coords: A set of coordinates to which we want add additional masking data
        :return: A set of coordinates (x, y, z) for each non-zero value in the matrix.

        ** Note: The dense mask matrix would be of shape: (TOK/ORD, VOCAB_SIZE, VOCAB_SIZE) **
    """
    coords = coords or set()           # (position, prev_token, token)

    # Masking possible order mask based on previous token
    for order in list_possible_orders:
        try:
            prev_token = 0
            order_tokens = get_order_tokens(order) + [EOS_TOKEN]
            order_tokens += [PAD_TOKEN] * (TOKENS_PER_ORDER - len(order_tokens))
            for position, order_token in enumerate(order_tokens):
                token = token_to_ix(order_token)
                if position == 0:
                    coords.add((position + offset, PAD_ID, token))
                    coords.add((position + offset, GO_ID, token))
                    coords.add((position + offset, EOS_ID, token))
                else:
                    coords.add((position + offset, prev_token, token))
                prev_token = token
        except KeyError:
            LOGGER.warning('[get_token_based_mask] Order "%s" is invalid. Skipping.', order)

    # Returning
    return coords

def get_order_based_mask(list_possible_orders, max_length=MAX_CANDIDATES):
    """ Returns a list of candidates ids padded to the max length
        :param list_possible_orders: The list of possible orders (e.g. ['A PAR H', 'A PAR - BUR', ...])
        :return: A list of candidates padded. (e.g. [1, 50, 252, 0, 0, 0, ...])
    """
    candidates = [order_to_ix(order) for order in list_possible_orders]
    candidates = [token for token in candidates if token is not None]
    candidates += [PAD_ID] * (max_length - len(candidates))
    if len(candidates) > max_length:
        LOGGER.warning('Found %d candidates, but only allowing a maximum of %d', len(candidates), max_length)
        candidates = candidates[:max_length]
    return candidates

def get_order_tokens(order):
    """ Retrieves the order tokens used in an order
        e.g. 'A PAR - MAR' would return ['A PAR', '-', 'MAR']
    """
    # We need to keep 'A', 'F', and '-' in a temporary buffer to concatenate them with the next word
    # We replace 'R' orders with '-'
    # Tokenization would be: 'A PAR S A MAR - BUR' --> 'A PAR', 'S', 'A MAR', '- BUR'
    #                        'A PAR R MAR'         --> 'A PAR', '- MAR'
    buffer, order_tokens = [], []
    for word in order.replace(' R ', ' - ').split():
        buffer += [word]
        if word not in ['A', 'F', '-']:
            order_tokens += [' '.join(buffer)]
            buffer = []
    return order_tokens

def get_vocabulary():
    """ Returns the list of words in the dictionary
        :return: The list of words in the dictionary
    """
    map_object = Map()
    locs = sorted([loc.upper() for loc in map_object.locs])

    vocab = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, DRAW_TOKEN]                                # Utility tokens
    vocab += ['<%s>' % power_name for power_name in get_map_powers(map_object)]         # Power names
    vocab += ['B', 'C', 'D', 'H', 'S', 'VIA', 'WAIVE']                                  # Order Tokens (excl '-', 'R')
    vocab += ['- %s' % loc for loc in locs]                                             # Locations with '-'
    vocab += ['A %s' % loc for loc in locs if map_object.is_valid_unit('A %s' % loc)]   # Army Units
    vocab += ['F %s' % loc for loc in locs if map_object.is_valid_unit('F %s' % loc)]   # Fleet Units
    return vocab

def token_to_ix(order_token):
    """ Computes the index of an order token in the vocabulary (i.e. order_token ==> token)
        :param order_token: The order token to get the index from (e.g. 'A PAR')
        :return: The index of the order token, a.k.a. the corresponding token (e.g. 10)
    """
    return VOCABULARY_KEY_TO_IX[order_token]

def ix_to_token(token):
    """ Computes the order token at a given index in the vocabulary (i.e. token ==> order_token)
        :param token: The token to convert to an order token (e.g. 10)
        :return: The corresponding order_token (e.g. 'A PAR')
    """
    return VOCABULARY_IX_TO_KEY[max(0, token)]

def get_order_frequency_table():
    """ Generates the order frequency table, with orders as key and the number of times the order was
        seen in the dataset as value
    """
    order_freq = {}
    if os.path.exists(MOVES_COUNT_DATASET_PATH):
        with open(MOVES_COUNT_DATASET_PATH, 'rb') as order_freq:
            order_freq = pickle.load(order_freq)                                    # {move: (nb_no_press, no_press)}
    return order_freq

def get_order_frequency(order, no_press_only):
    """ Computes the number of time an order has been seen in the dataset
        :param order: The order to check (e.g. 'A PAR H')
        :param no_press_only: Boolean flag to indicate we only want the count for no press games.
        :return: The number of time the order has been seen in the dataset
        :type no_press_only: bool
    """
    if order not in ORDER_FREQUENCY:
        return 0
    press_count, no_press_count = ORDER_FREQUENCY[order]
    return no_press_count if no_press_only else press_count + no_press_count

def get_power_vocabulary():
    """ Computes a sorted list of powers in the standard map
        :return: A list of the powers
    """
    standard_map = Map()
    return sorted([power_name for power_name in standard_map.powers])

def get_order_vocabulary():
    """ Computes the list of all valid orders on the standard map
        :return: A sorted list of all valid orders on the standard map
    """
    # pylint: disable=too-many-nested-blocks,too-many-branches
    categories = ['H', 'D', 'B', '-', 'R', 'SH', 'S-',
                  '-1', 'S1', 'C1',                 # Move, Support, Convoy (using 1 fleet)
                  '-2', 'S2', 'C2',                 # Move, Support, Convoy (using 2 fleets)
                  '-3', 'S3', 'C3',                 # Move, Support, Convoy (using 3 fleets)
                  '-4', 'S4', 'C4']                 # Move, Support, Convoy (using 4 fleets)
    orders = {category: set() for category in categories}
    map_object = Map()
    locs = sorted([loc.upper() for loc in map_object.locs])

    # All holds, builds, and disbands orders
    for loc in locs:
        for unit_type in ['A', 'F']:
            if map_object.is_valid_unit('%s %s' % (unit_type, loc)):
                orders['H'].add('%s %s H' % (unit_type, loc))
                orders['D'].add('%s %s D' % (unit_type, loc))

                # Allowing builds in all SCs (even though only homes will likely be used)
                if loc[:3] in map_object.scs:
                    orders['B'].add('%s %s B' % (unit_type, loc))

    # Moves, Retreats, Support Holds
    for unit_loc in locs:
        for dest in [loc.upper() for loc in map_object.abut_list(unit_loc, incl_no_coast=True)]:
            for unit_type in ['A', 'F']:
                if not map_object.is_valid_unit('%s %s' % (unit_type, unit_loc)):
                    continue

                if map_object.abuts(unit_type, unit_loc, '-', dest):
                    orders['-'].add('%s %s - %s' % (unit_type, unit_loc, dest))
                    orders['R'].add('%s %s R %s' % (unit_type, unit_loc, dest))

                # Making sure we can support destination
                if not (map_object.abuts(unit_type, unit_loc, 'S', dest)
                        or map_object.abuts(unit_type, unit_loc, 'S', dest[:3])):
                    continue

                # Support Hold
                for dest_unit_type in ['A', 'F']:
                    for coast in ['', '/NC', '/SC', '/EC', '/WC']:
                        if map_object.is_valid_unit('%s %s%s' % (dest_unit_type, dest, coast)):
                            orders['SH'].add('%s %s S %s %s%s' % (unit_type, unit_loc, dest_unit_type, dest, coast))

    # Convoys, Move Via
    for nb_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if nb_fleets > 4:
            continue

        for start, fleets, dests in map_object.convoy_paths[nb_fleets]:
            for end in dests:
                orders['-%d' % nb_fleets].add('A %s - %s VIA' % (start, end))
                orders['-%d' % nb_fleets].add('A %s - %s VIA' % (end, start))
                for fleet_loc in fleets:
                    orders['C%d' % nb_fleets].add('F %s C A %s - %s' % (fleet_loc, start, end))
                    orders['C%d' % nb_fleets].add('F %s C A %s - %s' % (fleet_loc, end, start))

    # Support Move (Non-Convoyed)
    for start_loc in locs:
        for dest_loc in [loc.upper() for loc in map_object.abut_list(start_loc, incl_no_coast=True)]:
            for support_loc in (map_object.abut_list(dest_loc, incl_no_coast=True)
                                + map_object.abut_list(dest_loc[:3], incl_no_coast=True)):
                support_loc = support_loc.upper()

                # A unit cannot support itself
                if support_loc[:3] == start_loc[:3]:
                    continue

                # Making sure the src unit can move to dest
                # and the support unit can also support to dest
                for src_unit_type in ['A', 'F']:
                    for support_unit_type in ['A', 'F']:
                        if (map_object.abuts(src_unit_type, start_loc, '-', dest_loc)
                                and map_object.abuts(support_unit_type, support_loc, 'S', dest_loc[:3])
                                and map_object.is_valid_unit('%s %s' % (src_unit_type, start_loc))
                                and map_object.is_valid_unit('%s %s' % (support_unit_type, support_loc))):
                            orders['S-'].add('%s %s S %s %s - %s' %
                                             (support_unit_type, support_loc, src_unit_type, start_loc, dest_loc[:3]))

    # Support Move (Convoyed)
    for nb_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if nb_fleets > 4:
            continue

        for start_loc, fleets, ends in map_object.convoy_paths[nb_fleets]:
            for dest_loc in ends:
                for support_loc in map_object.abut_list(dest_loc, incl_no_coast=True):
                    support_loc = support_loc.upper()

                    # A unit cannot support itself
                    if support_loc[:3] == start_loc[:3]:
                        continue

                    # A fleet cannot support if it convoys
                    if support_loc in fleets:
                        continue

                    # Making sure the support unit can also support to dest
                    # And that the support unit is not convoying
                    for support_unit_type in ['A', 'F']:
                        if (map_object.abuts(support_unit_type, support_loc, 'S', dest_loc)
                                and map_object.is_valid_unit('%s %s' % (support_unit_type, support_loc))):
                            orders['S%d' % nb_fleets].add(
                                '%s %s S A %s - %s' % (support_unit_type, support_loc, start_loc, dest_loc[:3]))

    # Building the list of final orders
    final_orders = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, DRAW_TOKEN]
    final_orders += ['<%s>' % power_name for power_name in get_map_powers(map_object)]
    final_orders += ['WAIVE']

    # Sorting each category
    for category in categories:
        category_orders = [order for order in orders[category] if order not in final_orders]
        final_orders += list(sorted(category_orders, key=lambda value: (value.split()[1],        # Sorting by loc
                                                                        value)))                 # Then alphabetically
    return final_orders

def order_to_ix(order):
    """ Computes the index of an order in the order vocabulary
        :param order: The order to get the index from
        :return: The index of the order  (None if not found)
    """
    if order in ORDER_VOCABULARY_KEY_TO_IX:
        return ORDER_VOCABULARY_KEY_TO_IX[order]

    # Adjustment for Supporting a move to a coast (stripping the coast)
    words = order.split()
    if len(words) == 7 and words[2] == 'S' and '/' in words[-1]:
        words[-1] = words[-1][:3]
        order = ' '.join([word for word in words])
    return ORDER_VOCABULARY_KEY_TO_IX[order] if order in ORDER_VOCABULARY_KEY_TO_IX else None

def ix_to_order(order_ix):
    """ Computes the order at a given index in the order vocabulary
        :param order_ix: The index of the order to return
        :return: The order at index
    """
    return ORDER_VOCABULARY_IX_TO_KEY[max(0, order_ix)]


# Vocabulary and constants
PAD_TOKEN = '<PAD>'
GO_TOKEN = '<GO>'
EOS_TOKEN = '<EOS>'
DRAW_TOKEN = '<DRAW>'

__VOCABULARY__ = get_vocabulary()
VOCABULARY_IX_TO_KEY = {token_ix: token for token_ix, token in enumerate(__VOCABULARY__)}
VOCABULARY_KEY_TO_IX = {token: token_ix for token_ix, token in enumerate(__VOCABULARY__)}
VOCABULARY_SIZE = len(__VOCABULARY__)
del __VOCABULARY__

__ORDER_VOCABULARY__ = get_order_vocabulary()
ORDER_VOCABULARY_IX_TO_KEY = {order_ix: order for order_ix, order in enumerate(__ORDER_VOCABULARY__)}
ORDER_VOCABULARY_KEY_TO_IX = {order: order_ix for order_ix, order in enumerate(__ORDER_VOCABULARY__)}
ORDER_VOCABULARY_SIZE = len(__ORDER_VOCABULARY__)
del __ORDER_VOCABULARY__

__POWER_VOCABULARY__ = get_power_vocabulary()
POWER_VOCABULARY_LIST = __POWER_VOCABULARY__
POWER_VOCABULARY_IX_TO_KEY = {power_ix: power for power_ix, power in enumerate(__POWER_VOCABULARY__)}
POWER_VOCABULARY_KEY_TO_IX = {power: power_ix for power_ix, power in enumerate(__POWER_VOCABULARY__)}
POWER_VOCABULARY_SIZE = len(__POWER_VOCABULARY__)
del __POWER_VOCABULARY__

ORDER_FREQUENCY = get_order_frequency_table()

PAD_ID = token_to_ix(PAD_TOKEN)
GO_ID = token_to_ix(GO_TOKEN)
EOS_ID = token_to_ix(EOS_TOKEN)
DRAW_ID = token_to_ix(DRAW_TOKEN)

# Predefined location order
STANDARD_TOPO_LOCS = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
                      'NWG', 'ENG', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
                      'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC',
                      'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
                      'STP/NC', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', 'SPA/NC',
                      'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'BOT', 'LVN',
                      'PRU', 'STP/SC', 'MOS', 'TUN', 'LYO', 'TYS', 'PIE',
                      'BOH', 'SIL', 'TYR', 'WAR', 'SEV', 'UKR', 'ION',
                      'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
                      'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU',
                      'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR',
                      'BUL', 'BUL/EC', 'CON', 'BUL/SC']

# Caching alignments
ALIGNMENTS_INDEX = get_alignments_index()

# Validating lengths to avoid accidental changes
assert VOCABULARY_SIZE == 220
assert ORDER_VOCABULARY_SIZE == 13042
assert POWER_VOCABULARY_SIZE == 7
