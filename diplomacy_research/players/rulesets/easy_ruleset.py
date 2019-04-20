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
""" Easy Ruleset

    Movement phase:
        1) - Hold if unit is on a foreign SC and SC is not captured
        2) - Attack unoccupied enemy SC
        3) - Move to unoccupied enemy territory
        4) - Attack occupied enemy SC
        5) - Attack occupied enemy unit
        6) - Move in direction of closest foreign SC
        7) - Otherwise hold

    Retreat phase:
        - Move to state having most friendly surrounding units
        - Disband if no retreat locations possible

    Adjustement phase:
        - If build, maintain a 60% land, 40% fleet ratio, build in location closest to closest enemy SC first
        - If disband, disband units that are further from enemy territory
"""
from operator import itemgetter
from diplomacy_research.models.state_space import build_game_from_state_proto
from diplomacy_research.players.rulesets.utils import get_distance

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

    # Finding orderable locs
    orderable_locations = game.get_orderable_locations(power_name)
    current_phase_type = state_proto.name[-1]

    # No orders to submit
    if not orderable_locations:
        return []

    # Phase selection
    if current_phase_type == 'M':
        return _run_movement_phase(game, power_name)
    if current_phase_type == 'R':
        return _run_retreats_phase(game, power_name)
    if current_phase_type == 'A':
        return _run_adjustment_phase(game, power_name)

    # Otherwise, returning no orders (i.e. game is completed)
    return []

def _run_movement_phase(game, power_name):
    """ Gets the move for the given power according to the ruleset. (for the movement phase)
        :param game: The game object
        :param power_name: The name of the power we are playing
        :return: A list of orders for that power.
        :type game: diplomacy.Game
    """
    orders = []

    # Movements phase
    # Calculating current center allocations, and friendly and enemy territories
    power = game.get_power(power_name)
    own_centers = power.centers
    other_centers = [center for center in game.map.scs if center not in own_centers]
    own_influence = [loc for loc in power.influence if (game.map.area_type(loc) in ['LAND', 'COAST']
                                                        and loc.upper()[:3] not in power.centers)]
    other_influence = [loc.upper() for loc in game.map.locs if (game.map.area_type(loc) in ['LAND', 'COAST']
                                                                and loc.upper()[:3] not in game.map.scs
                                                                and loc.upper()[:3] not in own_influence)]

    # Calculating locations of enemy units
    enemy_units = []
    for other_power in game.powers.values():
        if power != other_power:
            enemy_units += [unit for unit in other_power.units]

    # Unoccupied enemy centers and territories
    unoccupied_sc = [center for center in other_centers if not [1 for unit in enemy_units if unit[2:5] == center]]
    unoccupied_terr = [loc for loc in other_influence if not [1 for unit in enemy_units if unit[2:5] == loc]]

    # Computing list of valid dests
    unordered_units = power.units[:]
    valid_dests = [loc.upper() for loc in game.map.locs if loc.upper() not in own_centers + own_influence]

    # First ordering units on uncaptured supply center to hold
    for unit in unordered_units[:]:
        if unit[2:5] in other_centers:
            orders += ['{} H'.format(unit)]
            unordered_units.remove(unit)
            valid_dests = [loc for loc in valid_dests if loc[:3] != unit[2:5]]

    # Second, assigning a priority to every adjacent location
    # Priority 1 - Unoccupied enemy SC
    # Priority 2 - Unoccupied enemy territory
    # Priority 3 - Occupied SC
    # Priority 4 - Enemy units
    priority = []
    for unit in unordered_units:
        for adj_loc in game.map.abut_list(unit[2:], incl_no_coast=True):
            adj_loc = adj_loc.upper()

            # Priority 1 - Enemy SC (unoccupied)
            if adj_loc[:3] in unoccupied_sc:
                priority += [(unit, adj_loc, 1)]

            # Priority 2 - Enemy Territory (unoccupied)
            elif adj_loc in unoccupied_terr:
                priority += [(unit, adj_loc, 2)]

            # Priority 3 - Occupied SC
            elif adj_loc[:3] in other_centers:
                priority += [(unit, adj_loc, 3)]

            # Priority 4 - Enemy Units
            elif [1 for unit in enemy_units if adj_loc[:3] == unit[2:5]]:
                priority += [(unit, adj_loc, 4)]

    # Sorting by priority
    priority.sort(key=itemgetter(2))

    # Assigning orders based on priority
    # Only assigning priority 1 and 2 here
    for unit, dest, current_priority in priority:
        if current_priority > 2:
            continue
        if unit in unordered_units and dest in valid_dests:

            # Direct move
            # Skipping if we can't move there, or if this would create a two-way bounce
            if not game.map.abuts(unit[0], unit[2:], '-', dest) or \
                    [1 for order in orders if '{} - {}'.format(dest, unit[2:]) in order]:
                continue
            unordered_units.remove(unit)
            valid_dests = [loc for loc in valid_dests if loc[:3] != dest[:3]]
            valid_dests += game.map.find_coasts(unit[2:5])
            orders += ['{} - {}'.format(unit, dest)]
            continue

    # Moving in direction of closest unoccupied SC or territory
    for unit in unordered_units[:]:
        dests = [loc.upper() for loc in game.map.abut_list(unit[2:], incl_no_coast=True)
                 if loc.upper() in valid_dests and not [1 for unit in enemy_units if unit[2:5] == loc.upper()]]

        # No valid dests, skipping
        if not dests:
            continue

        # Otherwise moving to loc closest to unoccupied SC or territory
        dest_distance = [(dest, get_distance(game, unit[0], dest, unoccupied_sc + unoccupied_terr, True))
                         for dest in dests]
        dest_distance.sort(key=itemgetter(1))
        for dest, _ in dest_distance:
            if not game.map.abuts(unit[0], unit[2:], '-', dest) or \
                    [1 for order in orders if '{} - {}'.format(dest, unit[2:]) in order]:
                continue

            # Moving there
            orders += ['{} - {}'.format(unit, dest)]
            unordered_units.remove(unit)
            valid_dests = [loc for loc in valid_dests if loc[:3] != dest[:3]]
            valid_dests += game.map.find_coasts(unit[2:5])
            break

    # Assigning orders based on priority
    # Only assigning priority 3+ here
    for unit, dest, current_priority in priority:
        if current_priority <= 2:
            continue
        if unit in unordered_units and dest in valid_dests:

            # Direct move
            # Skipping if we can't move there, or if this would create a two-way bounce
            if not game.map.abuts(unit[0], unit[2:], '-', dest) or \
                    [1 for order in orders if '{} - {}'.format(dest, unit[2:]) in order]:
                continue
            unordered_units.remove(unit)
            valid_dests = [loc for loc in valid_dests if loc[:3] != dest[:3]]
            valid_dests += game.map.find_coasts(unit[2:5])
            orders += ['{} - {}'.format(unit, dest)]
            continue

    # Finally, moving in direction of closest occupied SC
    for unit in unordered_units[:]:
        dests = [loc.upper() for loc in game.map.abut_list(unit[2:], incl_no_coast=True)
                 if loc.upper() in valid_dests]

        # No valid dests, holding
        if not dests:
            unordered_units.remove(unit)
            orders += ['{} H'.format(unit)]
            continue

        # Otherwise moving to loc closest to unoccupied SC or territory
        dest_distance = [(dest, get_distance(game, unit[0], dest, other_centers, True)) for dest in dests]
        dest_distance.sort(key=itemgetter(1))
        for dest, _ in dest_distance:
            if not game.map.abuts(unit[0], unit[2:], '-', dest) or \
                    [1 for order in orders if '{} - {}'.format(dest, unit[2:]) in order]:
                continue

            # Moving there
            orders += ['{} - {}'.format(unit, dest)]
            unordered_units.remove(unit)
            valid_dests = [loc for loc in valid_dests if loc[:3] != dest[:3]]
            valid_dests += game.map.find_coasts(unit[2:5])
            break

    # Returning orders
    return orders

def _run_retreats_phase(game, power_name):
    """ Gets the move for the given power according to the ruleset. (for the retreats phase)
        :param game: The game object
        :param power_name: The name of the power we are playing
        :return: A list of orders for that power.
        :type game: diplomacy.Game
    """
    orders = []

    # Retreats phase
    power = game.get_power(power_name)
    own_units = power.units
    for retreating_unit, allowed_locs in power.retreats.items():

        # No valid retreat locations, disbanding
        # If only one location, going there
        # Otherwise finding the one with the most nearby friendly units
        if not allowed_locs:
            orders += ['{} D'.format(retreating_unit)]

        elif len(allowed_locs) == 1:
            orders += ['{} R {}'.format(retreating_unit, allowed_locs[0])]

        else:
            friends_nearby = []
            for retreat_loc in allowed_locs:
                nb_friends = 0
                for neighbour_loc in game.map.abut_list(retreat_loc, incl_no_coast=True):
                    neighbour_loc = neighbour_loc.upper()
                    if 'A {}'.format(neighbour_loc) in own_units or 'F {}'.format(neighbour_loc) in own_units:
                        nb_friends += 1
                friends_nearby += [(retreat_loc, nb_friends)]
            friends_nearby.sort(key=itemgetter(1), reverse=True)
            orders += ['{} R {}'.format(retreating_unit, friends_nearby[0][0])]

    # Returning
    return orders

def _run_adjustment_phase(game, power_name):
    """ Gets the move for the given power according to the ruleset. (for the adjustment phase)
        :param game: The game object
        :param power_name: The name of the power we are playing
        :return: A list of orders for that power.
        :type game: diplomacy.Game
    """
    orders = []

    # Adjustment / Build phase
    power = game.get_power(power_name)
    orderable_locations = game.get_orderable_locations(power_name)

    nb_builds = len(power.centers) - len(power.units)
    nb_armies = len([unit for unit in power.units if unit[0] == 'A'])
    nb_fleets = len([unit for unit in power.units if unit[0] == 'F'])

    # We can build units
    if nb_builds > 0:
        # Maintaining a 40% fleet, 60% army
        nb_units_after = nb_armies + nb_fleets + nb_builds
        fleet_builds = max(0, int(round(nb_units_after * 0.4, 0)) - nb_fleets)
        army_builds = nb_builds - fleet_builds

        # Finding our centers and the enemy centers
        own_centers = power.centers
        other_centers = [center for center in game.map.scs if center not in own_centers]

        # Calculating distance to closest enemy center from every buildable location
        build_loc_distance = [(build_loc, min(get_distance(game, 'A', build_loc, other_centers, False),
                                              get_distance(game, 'F', build_loc, other_centers, False)))
                              for build_loc in orderable_locations]
        build_loc_distance.sort(key=itemgetter(1))

        # Building from units with smallest distance first
        for build_loc, _ in build_loc_distance:
            area_type = game.map.area_type(build_loc)
            if fleet_builds and area_type == 'WATER':
                fleet_builds -= 1
                orders += ['F {} B'.format(build_loc)]
            elif army_builds and area_type == 'LAND':
                army_builds -= 1
                orders += ['A {} B'.format(build_loc)]
            else:
                if fleet_builds:
                    fleet_builds -= 1
                    orders += ['F {} B'.format(build_loc)]
                elif army_builds:
                    army_builds -= 1
                    orders += ['A {} B'.format(build_loc)]

    # We need to disband units
    elif nb_builds < 0:
        own_influence = [loc for loc in power.influence if (game.map.area_type(loc) in ['LAND', 'COAST']
                                                            and loc.upper()[:3] not in power.centers)]
        other_influence = [loc.upper() for loc in game.map.locs if (game.map.area_type(loc) in ['LAND', 'COAST']
                                                                    and loc.upper() not in own_influence)]

        # Finding distance to nearest enemy territory
        units = power.units
        unit_distance = [(unit, get_distance(game, unit[0], unit[2:], other_influence, True)) for unit in units]
        unit_distance.sort(key=itemgetter(1), reverse=True)

        # Removing units that are further from enemy territory
        nb_disband_left = abs(nb_builds)
        for unit, _ in unit_distance:
            if nb_disband_left:
                orders += ['{} D'.format(unit)]
                nb_disband_left -= 1

    # Returning orders
    return orders
