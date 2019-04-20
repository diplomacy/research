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
""" A Python version of David Norman's DumbBot. """
import collections
import logging
import random
from diplomacy_research.models.state_space import build_game_from_state_proto

# --- Constants ---
LOGGER = logging.getLogger(__name__)

# Nb of proximity maps
PROXIMITY_DEPTHS = 10

# Shape the power size by ax^2 + bx + c
SIZE_SQUARE_COEFFICIENT = 1.
SIZE_COEFFICIENT = 4.
SIZE_CONSTANT = 16

# Importance of attack SC we don't own in spring/fall
SPRING_PROXIMITY_ATTACK_WEIGHT = 700
FALL_PROXIMITY_ATTACK_WEIGHT = 600

# Importance of defending our own center in spring/fall
SPRING_PROXIMITY_DEFENSE_WEIGHT = 300
FALL_PROXIMITY_DEFENSE_WEIGHT = 400

# Importance of proximity_map[n] in Spring/Fall/Building/Disbanding
SPRING_PROXIMITY_WEIGHTS = [100, 1000, 30, 10, 6, 5, 4, 3, 2, 1]
FALL_PROXIMITY_WEIGHTS = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]
BUILD_PROXIMITY_WEIGHTS = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]
REMOVE_PROXIMITY_WEIGHTS = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]

# Importance of attack strength in Spring/Fall
SPRING_STRENGTH_WEIGHT = 1000
FALL_STRENGTH_WEIGHT = 1000

# Importance of lack of competition in Spring/Fall
SPRING_COMPETITION_WEIGHT = 1000
FALL_COMPETITION_WEIGHT = 1000

# Importance of building in province we need to defend
BUILD_DEFENSE_WEIGHT = 1000

# Importance of removing unit we don't need to defend
REMOVE_DEFENSE_WEIGHT = 1000

# If not automatic, chance of playing best move if inferior move is nearly as good
ALTERNATIVE_DIFF_MODIFIER = 5

# percentage chance of automatically playing the next move
PLAY_ALTERNATIVE = 0.5

# --- Named tuples ---
class Factors(
        collections.namedtuple('Factors', ('proximity_maps',        # A list of maps, with unit as key
                                           'competition_map',       # A dict with province as key
                                           'strength_map',          # A dict with province as key
                                           'defense_map'))):        # A dict with province as key
    """ A class to hold all factors computed for the encoding """

class FactorWeights(
        collections.namedtuple('FactorWeights', ('proximity_weights',
                                                 'competition_weight',
                                                 'strength_weight',
                                                 'defense_weight'))):
    """ A class to hold all factor weights used for the encoding """


# ---------------------------------
#           MAIN FUNCTION
# ---------------------------------
def run_ruleset(state_proto, power_name):
    """ Gets the move for the given power according to the ruleset.
        :param state_proto: A `.proto.game.State` representation of the state of the game.
        :param power_name: The name of the power we are playing
        :return: A list of orders for that power.
    """
    # Power has been eliminated
    if not state_proto.units[power_name].value and not state_proto.centers[power_name].value:
        return []

    # Building the game object
    game = build_game_from_state_proto(state_proto)

    # Game is forming / completed
    if game.get_current_phase()[0] not in 'SFW':
        return []

    # Encoding the board to factors
    dest_unit_value, factors = get_board_factors(game, power_name)

    # Decode orders
    return decode_orders(game, power_name, dest_unit_value, factors)


# ---------------------------------
#           ENCODING
# ---------------------------------
def get_board_factors(game, power_name):
    """ Compute destination value by computing various factors
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :return: A tuple consisting of
                    1) the dest_unit_value,
                    2) the factors
        :type game: diplomacy.Game
    """
    season = game.get_current_phase()[0]
    power = game.get_power(power_name)

    # Compute factors
    if season in 'SW':
        factors = calculate_factors(game=game,
                                    power_name=power_name,
                                    proximity_attack_weight=SPRING_PROXIMITY_ATTACK_WEIGHT,
                                    proximity_defense_weight=SPRING_PROXIMITY_DEFENSE_WEIGHT)
    else:
        factors = calculate_factors(game=game,
                                    power_name=power_name,
                                    proximity_attack_weight=FALL_PROXIMITY_ATTACK_WEIGHT,
                                    proximity_defense_weight=FALL_PROXIMITY_DEFENSE_WEIGHT)

    # Computing factor weights
    if season == 'S':
        factor_weights = FactorWeights(proximity_weights=SPRING_PROXIMITY_WEIGHTS,
                                       competition_weight=SPRING_COMPETITION_WEIGHT,
                                       strength_weight=SPRING_STRENGTH_WEIGHT,
                                       defense_weight=0)
    elif season == 'F':
        factor_weights = FactorWeights(proximity_weights=FALL_PROXIMITY_WEIGHTS,
                                       competition_weight=FALL_COMPETITION_WEIGHT,
                                       strength_weight=FALL_STRENGTH_WEIGHT,
                                       defense_weight=0)
    else:
        nb_builds = len(power.centers) - len(power.units)

        # Build
        if nb_builds >= 0:
            factor_weights = FactorWeights(proximity_weights=BUILD_PROXIMITY_WEIGHTS,
                                           competition_weight=0,
                                           strength_weight=0,
                                           defense_weight=BUILD_DEFENSE_WEIGHT)
        # Disband
        else:
            factor_weights = FactorWeights(proximity_weights=REMOVE_PROXIMITY_WEIGHTS,
                                           competition_weight=0,
                                           strength_weight=0,
                                           defense_weight=REMOVE_DEFENSE_WEIGHT)

    # Computing destination value
    dest_unit_value = calculate_dest_unit_value(factors=factors,
                                                factor_weights=factor_weights,
                                                is_winter=(season == 'W'))

    # Returning board factors
    return dest_unit_value, factors

def calculate_factors(game, power_name, proximity_attack_weight, proximity_defense_weight):
    """ Compute the proximity_maps, competition_map, and strength_map, as defined in the original C++ code.
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param proximity_attack_weight: The weight used to compute the importance of attacking.
        :param proximity_defense_weight: The weight used to compute the importance of defending.
        :return: The factors (proximity_maps, competition_map, strength_map, defense_map)
        :type game: diplomacy.Game
        :rtype: Factors
    """
    # Get attack, defense values
    attack_map, defense_map = calculate_attack_defense(game, power_name)

    # List of all possible units
    all_units = ['{} {}'.format(unit_type, loc.upper())
                 for unit_type in 'AF'
                 for loc in game.map.locs
                 if game.map.is_valid_unit('{} {}'.format(unit_type, loc.upper()))]

    # Compute initial proximity value, for all non-dislodged units on the board
    init_proximity_map = {}
    for unit in all_units:
        init_proximity_map[unit] = (attack_map[unit[2:5]] * proximity_attack_weight
                                    + defense_map[unit[2:5]] * proximity_defense_weight)
    proximity_maps = [init_proximity_map]

    # Building deeper proximity maps
    # For deeper maps, the value of a location is equal to the (sum of adjacent units + self) / 5
    for proximity_depth in range(1, PROXIMITY_DEPTHS):
        prev_proximity_map = proximity_maps[proximity_depth - 1]
        curr_proximity_map = {unit: 0 for unit in all_units}

        # Updating all units
        for unit in all_units:

            # Finding adjacent locations
            adj_locs = set()
            for dest_coast in game.map.find_coasts(unit[2:5]):
                adj_locs |= {loc.upper()[:3] for loc in game.map.abut_list(dest_coast, incl_no_coast=True)}

            # Finding potentially adjacent units
            adj_units = [adj_unit for adj_unit in all_units if adj_unit[2:5] in adj_locs]

            # Finding units that could in the current provice
            self_units = [self_unit for self_unit in all_units if self_unit[2:5] == unit[2:5]]

            # Computing self contributions
            self_contrib = 0
            for self_unit in self_units:
                self_contrib = max(self_contrib, prev_proximity_map[self_unit])

            # Computing other contributions
            other_contrib = 0.
            for adj_unit in adj_units:
                if game.map.abuts(adj_unit[0], adj_unit[2:], '-', unit[2:]) \
                        or game.map.abuts(adj_unit[0], adj_unit[2:], '-', unit[2:5]):
                    other_contrib += prev_proximity_map[adj_unit]

            # Update score
            # Dividing by 5, since each location has on average 4 locations (+ itself)
            curr_proximity_map[unit] = (self_contrib + other_contrib) / 5.

        # Append proximity map to list
        proximity_maps += [curr_proximity_map]

    # Compute adjacent unit counts
    adjacent_unit_counts = calculate_adjacent_unit_counts(game)

    # Compute strength and competition map
    # Strength map: Number of adjacent units from same power
    # Competition map: Largest number of enemy adjacent units
    provinces = [loc.upper() for loc in game.map.locs if '/' not in loc]
    strength_map = {loc: 0 for loc in provinces}
    competition_map = {loc: 0 for loc in provinces}

    for loc in provinces:
        for adjacent_power, nb_adjacent_units in adjacent_unit_counts[loc].items():
            if adjacent_power == power_name:
                strength_map[loc] = nb_adjacent_units
            else:
                competition_map[loc] = max(competition_map[loc], nb_adjacent_units)

    # Returning factors
    return Factors(proximity_maps=proximity_maps,
                   competition_map=competition_map,
                   strength_map=strength_map,
                   defense_map=defense_map)

def calculate_attack_defense(game, power_name):
    """ Compute the attack and defense maps for the current power.
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :return: A tuple consisting of:
                    1) attack_map: Dictionary with province as key and attack weight as value
                    2) defense_map: Dictionary with province as key and defense weight as value
        :type game: diplomacy.Game
    """
    # compute power size
    power_sizes = get_power_sizes(game)

    # Compute attack and defense value for each province
    provinces = [loc.upper() for loc in game.map.locs if '/' not in loc]
    attack_map = {loc: 0 for loc in provinces}
    defense_map = {loc: 0 for loc in provinces}

    for power in game.powers.values():
        for loc in power.centers:

            # Not ours, updating attack value by the size of the owning power
            if power.name != power_name:
                attack_map[loc] = power_sizes[power.name]

            # It's ours, update defense value by the size of the largest enemy which has a unit that can move in
            else:
                defense_map[loc] = get_defense_value(game=game,
                                                     power_name=power_name,
                                                     loc=loc,
                                                     power_sizes=power_sizes)

    # Returning the attack, defense map
    return attack_map, defense_map

def get_power_sizes(game):
    """ Return a dict that with power_name as key and value of its supply center as value.
        :param game: An instance of `diplomacy.Game`
        :return: A dict for power name as key and A * (nb_sc) ^2 + B * nb_sc + C as value
        :type game: diplomacy.Game
    """
    A, B, C = SIZE_SQUARE_COEFFICIENT, SIZE_COEFFICIENT, SIZE_CONSTANT                                                  # pylint: disable=invalid-name
    return {power.name: A * len(power.centers) ** 2 + B * len(power.centers) + C
            for power in game.powers.values()}

def get_defense_value(game, power_name, loc, power_sizes):
    """ Compute the defense value of a location (i.e. the power size of the largest power of an adjacent unit)
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param loc: The location for which we want to compute the defense value.
        :param power_sizes: Dictionary with the name of a power as key and (A*nb_sc^2 + B*nb_sc + C) as value
        :return: The location defense value.
                 i.e. the power_size of the largest power with a unit that can move in.
        :type game: diplomacy.Game
    """
    largest_power_size = 0
    loc_with_coasts = game.map.find_coasts(loc)

    # Finding the largest enemy unit that can move to loc
    for power in game.powers.values():
        if power.name == power_name:
            continue
        for unit in power.units:
            for dest in loc_with_coasts:
                if game.map.abuts(unit[0], unit[2:], '-', dest):
                    largest_power_size = max(power_sizes[power.name], largest_power_size)
                    break
    return largest_power_size

def calculate_adjacent_unit_counts(game):
    """ Compute the number of units from a power that are adjacent to each location
        :param game: An instance of `diplomacy.Game`
        :param units_on_board: A set containing all the units on the board {'A PAR', 'F BRE', ...}
        :return: A dict with
                    - loc as key and a dictionary of power_name as key and nb adj units from power as value
                e.g. {'PAR': {'FRANCE': 2, 'ENGLAND': 0, ...}}
                if 2 units for FRANCE can move in Paris, but none from England
        :type game: diplomacy.Game
    """
    provinces = [loc.upper() for loc in game.map.locs if '/' not in loc]
    adjacent_unit_counts = {loc: {power_name: set() for power_name in game.powers} for loc in provinces}

    for dest in provinces:

        # Building a list of src locs that could move to dest
        src_locs = set()
        for dest_coast in game.map.find_coasts(dest):
            src_locs |= {loc.upper() for loc in game.map.abut_list(dest_coast, incl_no_coast=True)}

        for src in src_locs:

            # Trying to check if we have an occupant
            occupant = game._occupant(src)                                  # STP -> A STP              - pylint: disable=protected-access
            if occupant is None:
                continue

            # Finding if the occupant can move
            for dest_coast in game.map.find_coasts(dest):
                if game.map.abuts(occupant[0], occupant[2:], '-', dest_coast):
                    break
            else:
                continue

            # Increasing the count of the owner
            occupant_owner = game._unit_owner(occupant)                                                 # pylint: disable=protected-access
            adjacent_unit_counts[dest][occupant_owner.name].add(occupant)

    # Returning the adjacent_unit_counts
    return {loc: {power_name: len(adjacent_unit_counts[loc][power_name]) for power_name in adjacent_unit_counts[loc]}
            for loc in adjacent_unit_counts}

def calculate_dest_unit_value(factors, factor_weights, is_winter=False):
    """ Compute the destination value for each loc
        :param factors: An instance of `Factors`
        :param factor_weights: An instance of `FactorWeights`
        :param is_winter: Whether or not it's in adjustment phase
        :return: dest_unit_value. A dict with unit as key, and the unit value as value
        :type factors: Factors
        :type factor_weights: FactorWeights
    """
    assert len(factors.proximity_maps) == len(factor_weights.proximity_weights), 'Different proximity lengths.'
    assert len(factors.proximity_maps) == PROXIMITY_DEPTHS, 'Expected %d proximity maps.' % PROXIMITY_DEPTHS

    # Destination value is computed by two parts:
    # 1. weighted sum of proximity values
    # 2. balance between competition and strength if not winter
    # 3. add defense value if winter.
    dest_unit_value = {loc: 0 for loc in factors.proximity_maps[0]}
    for unit in dest_unit_value:
        for prox_ix in range(PROXIMITY_DEPTHS):
            dest_unit_value[unit] += factor_weights.proximity_weights[prox_ix] * factors.proximity_maps[prox_ix][unit]
        if is_winter:
            dest_unit_value[unit] += factor_weights.defense_weight * factors.defense_map[unit[2:5]]
        else:
            dest_unit_value[unit] += factor_weights.strength_weight * factors.strength_map[unit[2:5]]
            dest_unit_value[unit] -= factor_weights.competition_weight * factors.competition_map[unit[2:5]]
    return dest_unit_value


# ---------------------------------
#           DECODING
# ---------------------------------
def decode_orders(game, power_name, dest_unit_value, factors):
    """ Decode orders from computed factors
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: A list of orders
        :type factors: Factors
        :type game: diplomacy.Game
    """
    phase_type = game.get_current_phase()[-1]

    # Movement phase
    if phase_type == 'M':
        return generate_movement_orders(game, power_name, dest_unit_value, factors)

    # Retreat Phaes
    if phase_type == 'R':
        return generate_retreat_orders(game, power_name, dest_unit_value)

    # Adjustment
    if phase_type == 'A':
        power = game.get_power(power_name)
        nb_builds = len(power.centers) - len(power.units)

        # Building
        if nb_builds >= 0:
            return generate_build_orders(game, power_name, dest_unit_value)

        # Disbanding
        return generate_disband_orders(game, power_name, dest_unit_value)

    # Otherwise, invalid phase_type
    LOGGER.error('Invalid phase type. Got %s. Expected M, R, A', phase_type)
    return []

def generate_movement_orders(game, power_name, dest_unit_value, factors):
    """ Generate movement orders
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: A list of orders
        :type factors: Factors
        :type game: diplomacy.Game
    """
    # Shuffling units
    power = game.get_power(power_name)
    unordered_units = power.units[:]
    units = power.units[:]
    all_units = [unit for unit in dest_unit_value]
    random.shuffle(unordered_units)

    # Moving units:  {unit: dest}   e.g. 'F STP/NC -> BAR' would have {'F STP/NC': 'BAR'}
    moving_units = {}

    # Dependencies: List of locations that depend of the key
    # e.g. {'PAR': ['MAR', 'BRE']} indicates that MAR and BRE are waiting for (depends on) the PAR order
    dependencies = {unit[2:5]: [] for unit in units}

    # List of final orders - {province: order}
    orders = {}

    # Generating orders
    while unordered_units:
        curr_unit = unordered_units.pop(0)

        # Finding adjacent locs
        adj_locs = set()
        for coast in game.map.find_coasts(curr_unit[2:5]):
            adj_locs |= {loc.upper() for loc in game.map.abut_list(coast, incl_no_coast=True)}

        # Building a list of destinations in reverse order (i.e. destination with highest value first)
        # Including itself, but excluding dependencies
        dest_units = sorted([unit for unit in all_units if unit[2:] in adj_locs
                             and game.map.abuts(curr_unit[0], curr_unit[2:], '-', unit[2:])     # Valid dest only
                             and unit[2:5] not in dependencies[curr_unit[2:5]]]                 # except if waiting
                            + [curr_unit],                                                      # including itself
                            key=lambda unit: dest_unit_value[unit],
                            reverse=True)

        # Picking destination
        selection_is_okay = False
        unit_ordered_to_move = True
        while not selection_is_okay:

            # Getting next destination
            selected_dest_unit = get_next_item(dest_units, dest_unit_value)
            selection_is_okay = True

            # Case 0 - Unit is holding (moving to same location)
            if selected_dest_unit[2:5] == curr_unit[2:5]:
                orders[curr_unit[2:5]] = '{} H'.format(curr_unit)
                break

            # Case 1 - Deal with occupying situation
            unit_occupying = [unit for unit in units if unit[2:5] == selected_dest_unit[2:5]]
            unit_occupying = None if not unit_occupying else unit_occupying[0]

            # If occupying unit is not ordered, insert current unit back after occupying unit
            # since we can't decide yet.
            if unit_occupying and unit_occupying[2:5] not in orders:
                unordered_units.insert(unordered_units.index(unit_occupying) + 1, curr_unit)
                unit_ordered_to_move = False
                dependencies[unit_occupying[2:5]] += [curr_unit[2:5]]

            # If occupying unit is not moving
            # Check if it needs support, otherwise the destination is not acceptable.
            elif unit_occupying and unit_occupying not in moving_units:
                if factors.competition_map[unit_occupying[2:5]] > 1:
                    orders[curr_unit[2:5]] = '{} S {}'.format(curr_unit, unit_occupying)
                    unit_ordered_to_move = False
                else:
                    selection_is_okay = False
                    dest_units.remove(selected_dest_unit)

            # Case 2 - Deal with units moving to the same location
            if selection_is_okay:
                unit_moving = [unit for unit, dest in moving_units.items() if dest[:3] == selected_dest_unit[2:5]]
                unit_moving = None if not unit_moving else unit_moving[0]

                # Support is someone already move in and there is competition on that location
                # Otherwise, the destination is not acceptable
                if unit_moving:
                    if factors.competition_map[selected_dest_unit[2:5]] > 0:
                        orders[curr_unit[2:5]] = '{} S {} - {}'.format(curr_unit,
                                                                       unit_moving,
                                                                       moving_units[unit_moving][:3])
                        unit_ordered_to_move = False
                    else:
                        selection_is_okay = False
                        dest_units.remove(selected_dest_unit)

            # Ready to issue move order
            if selection_is_okay and unit_ordered_to_move:
                orders[curr_unit[2:5]] = '{} - {}'.format(curr_unit, selected_dest_unit[2:])
                moving_units[curr_unit] = selected_dest_unit[2:]

    # Check for wasted holds
    orders = check_wasted_holds(game, orders, moving_units, dest_unit_value, factors)

    # Extract orders from order details
    return list(orders.values())

def generate_retreat_orders(game, power_name, dest_unit_value):
    """ Generate retreat orders
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: A list of orders
        :type factors: Factors
        :type game: diplomacy.Game
    """
    # Shuffling units
    power = game.get_power(power_name)
    unordered_units = [unit for unit in power.retreats]
    all_units = [unit for unit in dest_unit_value]
    random.shuffle(unordered_units)

    # Moving units:  {unit: dest}   e.g. 'F STP/NC -> BAR' would have {'F STP/NC': 'BAR'}
    moving_units = {}

    # List of final orders - {province: order}
    orders = {}

    # Generating orders
    while unordered_units:
        curr_unit = unordered_units.pop(0)

        # Finding adjacent locs
        adj_locs = set()
        for coast in game.map.find_coasts(curr_unit[2:5]):
            adj_locs |= {loc.upper() for loc in game.map.abut_list(coast, incl_no_coast=True)}

        # Building a list of destinations in reverse order (i.e. destination with highest value first)
        dest_units = sorted([unit for unit in all_units if unit[2:] in adj_locs
                             and game.map.abuts(curr_unit[0], curr_unit[2:], '-', unit[2:])],    # Valid dest only
                            key=lambda unit: dest_unit_value[unit],
                            reverse=True)

        # Picking destination
        selection_is_okay = False
        while not selection_is_okay:

            # No destination - Disbanding
            if not dest_units:
                orders[curr_unit[2:5]] = '{} D'.format(curr_unit)
                break

            # Getting next destination
            selected_dest_unit = get_next_item(dest_units, dest_unit_value)
            selection_is_okay = True

            # Selecting next destination if there is already a moving unit
            unit_moving = [unit for unit, dest in moving_units.items() if dest[:3] == selected_dest_unit[2:5]]
            unit_moving = None if not unit_moving else unit_moving[0]
            if unit_moving:
                selection_is_okay = False
                dest_units.remove(selected_dest_unit)

            # Check if that destination is already occupied
            occupant = game._occupant(selected_dest_unit[2:5], any_coast=1)                                       # pylint: disable=protected-access
            if occupant:
                selection_is_okay = False
                dest_units.remove(selected_dest_unit)

            # Otherwise, it's okay to retreat there
            if selection_is_okay:
                orders[curr_unit[2:5]] = '{} R {}'.format(curr_unit, selected_dest_unit[2:])
                moving_units[curr_unit] = selected_dest_unit[2:]

    # Returning orders
    return list(orders.values())

def generate_build_orders(game, power_name, dest_unit_value):
    """ Generate build orders
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: A list of orders
        :type factors: Factors
        :type game: diplomacy.Game
    """
    open_homes = game.get_orderable_locations(power_name)
    power = game.get_power(power_name)
    nb_builds = min(len(open_homes), len(power.centers) - len(power.units))

    # Getting the list of possible units that can be built
    # Sorted by decreasing value
    sorted_units = sorted(['{} {}'.format(unit_type, coast)
                           for unit_type in 'AF'
                           for loc in open_homes
                           for coast in game.map.find_coasts(loc)
                           if game.map.is_valid_unit('{} {}'.format(unit_type, coast))],
                          key=lambda unit: dest_unit_value[unit],
                          reverse=True)

    # Generating orders
    orders = {}                                                 # {province: order}
    while len(orders) < nb_builds and sorted_units:
        selected_unit = get_next_item(sorted_units, dest_unit_value)
        orders[selected_unit[2:5]] = '{} B'.format(selected_unit)
        sorted_units = [unit for unit in sorted_units if unit[2:5] != selected_unit[2:5]]

    # Returning
    return list(orders.values())

def generate_disband_orders(game, power_name, dest_unit_value):
    """ Generate disband orders
        :param game: An instance of `diplomacy.Game`
        :param power_name: The name of the power we are playing
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: A list of orders
        :type factors: Factors
        :type game: diplomacy.Game
    """
    power = game.get_power(power_name)
    nb_disbands = abs(len(power.centers) - len(power.units))

    # Getting the list of units that can be disbanded
    # Sorted by increasing value
    sorted_units = sorted([unit for unit in power.units],
                          key=lambda unit: dest_unit_value[unit])

    # Generating orders
    orders = {}                                                 # {province: order}
    for _ in range(nb_disbands):
        selected_unit = get_next_item(sorted_units, dest_unit_value)
        orders[selected_unit[2:5]] = '{} D'.format(selected_unit)
        sorted_units = [unit for unit in sorted_units if unit != selected_unit]

    # Returning
    return list(orders.values())

def check_wasted_holds(game, orders, moving_units, dest_unit_value, factors):
    """ Replace unnecessary holds with a support if possible
        :param game: An instance of `diplomacy.Game`
        :param orders: A dictionary with the province as key and the order for the unit at that province as value
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :param factors: An instance of `Factors`
        :return: An updated orders dictionary
        :type factors: Factors
        :type game: diplomacy.Game
    """
    holding_units = [' '.join(order.split()[:2]) for order in orders.values() if order.split()[-1] == 'H']
    for unit in holding_units:

        # Track the best destination we could support
        max_dest_value = 0.
        other_unit = None
        other_unit_dest = None

        # Destinations that the unit can move to
        for other_loc in [loc.upper() for loc in game.map.abut_list(unit[2:])
                          if game.map.abuts(unit[0], unit[2:], '-', loc.upper())]:

            # There is a moving unit there and it needs support
            # Recording unit if it has the best value
            unit_moving = [unit for unit, dest in moving_units.items() if dest[:3] == other_loc[:3]]
            unit_moving = None if not unit_moving else unit_moving[0]
            if unit_moving and factors.competition_map[other_loc[:3]] > 0:
                if dest_unit_value[unit_moving] > max_dest_value:
                    max_dest_value = dest_unit_value[unit_moving]
                    other_unit_dest = other_loc
                    other_unit = unit_moving

            # Checking if there is a unit occupying the location, not moving and needing support
            unit_occupying = [' '.join(order.split()[:2]) for loc, order in orders.items() if loc == other_loc]
            unit_occupying = None if not unit_occupying else unit_occupying[0]
            if unit_occupying and unit_occupying not in moving_units and factors.competition_map[other_loc[:3]] > 1:
                if dest_unit_value[unit_occupying] > max_dest_value:
                    max_dest_value = dest_unit_value[unit_occupying]
                    other_unit_dest = other_loc
                    other_unit = unit_occupying

        # If there is something worth supporting, changing the H to a S
        if max_dest_value > 0:
            if other_unit[2:5] == other_unit_dest[:3]:
                orders[unit[2:5]] = '{} S {}'.format(unit, other_unit)
            else:
                orders[unit[2:5]] = '{} S {} - {}'.format(unit, other_unit, other_unit_dest)

    # Returning orders
    return orders

def get_next_item(sorted_units, dest_unit_value):
    """ Selects the next destination
        :param sorted_units: A sorted list of units (increasing or decreasing)
        :param dest_unit_value: A dict with unit as key, and unit value as value
        :return: The next item
    """
    item_ix = 0
    while True:

        # Last item
        if item_ix + 1 == len(sorted_units):
            break

        # Determining whether or not to pick the item
        curr_item_value = dest_unit_value[sorted_units[item_ix + 0]]
        next_item_value = dest_unit_value[sorted_units[item_ix + 1]]
        if curr_item_value == 0:
            next_chance = 0
        else:
            next_chance = abs(curr_item_value - next_item_value) * ALTERNATIVE_DIFF_MODIFIER / curr_item_value

        # Selecting next move
        if PLAY_ALTERNATIVE > random.random() >= next_chance:
            item_ix += 1
            continue

        # Otherwise, selecting the current move
        break

    # Returning the chosen item
    return sorted_units[item_ix]
