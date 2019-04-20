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
""" Distance
    - Responsible for computing the distance between 2 states
"""
import numpy as np
from diplomacy import Map
from diplomacy_research.models.state_space import get_sorted_locs, get_map_powers

# Constants
CACHE = {}

# ----------------------------------------------------
# ----------           Distances       ---------------
# ----------------------------------------------------
def l1_distance(matrix_1, matrix_2):
    """ Computes the L1 distance between 2 matrices """
    return np.sum(np.abs(matrix_1 - matrix_2))

def l2_distance(matrix_1, matrix_2):
    """ Computes the L2 distance (Frobenius norm) between 2 matrices """
    return np.linalg.norm(matrix_1 - matrix_2)


# ----------------------------------------------------
# ----------      State to Matrix      ---------------
# ----------------------------------------------------
def _build_unit_heat_map(map_object, unit_power, unit):
    """ Builds the heat map for a single unit
        :param map_object: The underlying `diplomacy.Map` object.
        :param unit_power: The power owning the unit (e.g. 'FRANCE')
        :param unit: The unit (e.g. 'A PAR')
        :return: A NB_LOCS x NB_POWERS heat map
        :type map_object: diplomacy.Map
    """
    locs = get_sorted_locs(map_object)
    powers = get_map_powers(map_object)
    unit_heat_map = np.zeros(shape=(len(locs), len(powers)), dtype=np.float32)
    unit = unit.replace('*', '')
    unit_type, unit_loc = unit.split(' ')[:2]

    # Finding neighbour locations
    neighbour_locs = [unit_loc]
    if '/' in unit_loc:
        neighbour_locs += [unit_loc[:3]]
    for loc in locs:
        if loc[:3] != unit_loc[:3] and map_object.abuts(unit_type, unit_loc, '-', loc):
            neighbour_locs += [loc]

    # Adding +0.5 for other powers, and +1 for current power for all neighbouring locations (incl unit_loc)
    for other_power in powers:
        for loc in neighbour_locs:
            if other_power == unit_power:
                unit_heat_map[locs.index(loc), powers.index(other_power)] += 1.
            else:
                unit_heat_map[locs.index(loc), powers.index(other_power)] += 0.5

    # Returning heat map
    return unit_heat_map

def proto_to_heat_map(state_proto):
    """ Computes a heat map representation of the state of the game

        Size: 81 x 7    (NB_LOCS x NB_POWERS)
        > +1 is added for each locs where power has a unit and for each location that unit can reach
        > +0.5 is added for each loc where another power has a unit and for each loc that unit can reach

        :param state_proto: A `.proto.game.State` proto of the state of the game.
        :return: A heat map representation (NB_LOCS, NB_POWERS) of the state of the game
    """
    map_object = None

    # Building heat map
    heat_maps = []
    for power_name in state_proto.units:
        for unit in state_proto.units[power_name].value:
            cache_key = 'heat_map/{}/{}'.format(power_name, unit)
            cache_value = CACHE.get(cache_key, None)

            # Computing heat map for unit and storing in cache
            if cache_value is None:
                map_object = map_object or Map(state_proto.map)
                cache_value = _build_unit_heat_map(map_object, power_name, unit)
                CACHE[cache_key] = cache_value

            # Adding the unit heap map
            heat_maps += [cache_value]

    # Summing the heat maps and returning
    return np.sum(heat_maps, axis=0)
