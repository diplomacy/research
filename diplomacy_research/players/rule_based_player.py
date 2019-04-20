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
""" Rule-Based Player
    - Contains a class representing a player that follows a specific ruleset to choose its moves
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from tornado import gen, concurrent
from diplomacy_research.players.player import Player

# Constants
LOGGER = logging.getLogger(__name__)

class RuleBasedPlayer(Player):
    """ Rule-Based Player Class """

    def __init__(self, ruleset, name=None):
        """ Constructor
            :param ruleset: The ruleset to use to compute the next moves
            :param name: Optional. The name of this player.
        """
        name = name or '%s/%s' % (self.__class__.__name__, ruleset.__module__.split('.')[-1])
        Player.__init__(self, name)
        max_workers = (os.cpu_count() or 1) * 5
        self.ruleset = ruleset
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def is_trainable(self):
        """ Returns a boolean that indicates if the player wants to be trained or not """
        return False

    @gen.coroutine
    def get_orders_details_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto,
                                      **kwargs):
        """ Gets the orders (and the corresponding policy details) for the locs the power should play.
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power we are playing
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
                - with_state_value: Boolean that indicates to also query the value function.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: If with_state_value=False (default), a tuple consisting of:
                        1) The list of orders the power should play (e.g. ['A PAR H', 'A MAR - BUR', ...])
                        2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                     If with_state_value=True, a tuple consisting of:
                        1) The list of orders the power should play (e.g. ['A PAR H', 'A MAR - BUR', ...])
                        2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                        3) The state value for the given state
        """
        orders = yield self._run_ruleset(state_proto, power_name)

        if kwargs.get('with_state_value', False):
            return orders, None, self.get_nb_centers(state_proto, power_name)
        return orders, None

    @gen.coroutine
    def get_state_value_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto=None,
                                   **kwargs):
        """ Calculates the player's value of the state of the game for a given power
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want to retrieve the value
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: A float representing the value of the state of the game to the specified power
        """
        # pylint: disable=unused-argument
        LOGGER.warning('There are no models available to query state value. Returning the number of centers.')
        return self.get_nb_centers(state_proto, power_name)

    @concurrent.run_on_executor
    def _run_ruleset(self, state_proto, power_name):
        """ Gets the move for the given power according to the ruleset.
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power we are playing
            :return: A list of orders for that power.
        """
        return self.ruleset(state_proto, power_name)
