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
""" TestPlayer
    - Contains the test cases for the Player object
"""
from tornado import gen
from diplomacy import Game
from diplomacy_research.models.state_space import extract_state_proto
from diplomacy_research.players.player import Player

class FakePlayer(Player):
    """ Test Case Player """
    @property
    def is_trainable(self):
        """ Test class - See documentation on Player object """
        return False

    @gen.coroutine
    def get_orders_details_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto,
                                      **kwargs):
        """ Test class - See documentation on Player object """

    @gen.coroutine
    def get_state_value_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto=None,
                                   **kwargs):
        """ Test class - See documentation on Player object """

def test_get_nb_centers():
    """ Testing if the number of supply centers is correct """
    game = Game()
    player = FakePlayer()
    state_proto = extract_state_proto(game)

    # Checking every power
    power_names = [power_name for power_name in game.powers]
    for power_name in power_names:
        assert player.get_nb_centers(state_proto, power_name) == len(game.get_power(power_name).centers)

def test_get_orderable_locations():
    """ Testing if the number of orderable locations is correct """
    game = Game()
    player = FakePlayer()
    state_proto = extract_state_proto(game)

    # Checking every power
    power_names = [power_name for power_name in game.powers]
    for power_name in power_names:
        expected_locs = [unit.replace('*', '')[2:5] for unit in state_proto.units[power_name].value]
        expected_locs += state_proto.builds[power_name].homes
        assert sorted(player.get_orderable_locations(state_proto, power_name)) == sorted(expected_locs)
