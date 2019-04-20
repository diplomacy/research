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
""" Ruleset utils
    - Contains various utility functions for the rulesets

"""
from diplomacy.utils import PriorityDict

def get_distance(game, unit_type, start, targets, skip_occupied):
    """ Calculate the distance from a unit to the closest target
        :param game: A `diplomacy.Game` instance.
        :param unit_type: The unit type to calculate distance (e.g. 'A' or 'F')
        :param start: The start location of the unit (e.g. 'LON')
        :param targets: The list of targets (first one reached calculates the distance)
        :param skip_occupied: Boolean flag. If set, doesn't calculate distance if a unit is blocking the path
        :return: The minimum distance from unit to one of the targets
        :type game: diplomacy.Game
    """
    visited = []
    if not targets:
        return 99999

    # Modified Djikstra
    to_check = PriorityDict()
    to_check[start] = 0
    while to_check:
        distance, current = to_check.smallest()
        del to_check[current]

        # Found smallest distance
        if current[:3] in targets:
            return distance

        # Marking visited
        if current in visited:
            continue
        visited += [current]

        # Finding neighbors and updating distance
        for loc in game.map.abut_list(current, incl_no_coast=True):
            loc = loc.upper()
            if loc in visited:
                continue
            elif skip_occupied and (game._unit_owner('A {}'.format(loc[:3]), coast_required=False)      # pylint: disable=protected-access
                                    or game._unit_owner('F {}'.format(loc[:3]), coast_required=False)): # pylint: disable=protected-access
                continue

            # Calculating distance
            if game._abuts(unit_type, current, '-', loc):                                               # pylint: disable=protected-access
                loc_distance = to_check[loc] if loc in to_check else 99999
                to_check[loc] = min(distance + 1, loc_distance)

    # Could not find destination
    return 99999
