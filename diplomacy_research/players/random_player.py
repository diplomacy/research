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
""" Random Player
    - Contains a class representing a player that chooses to move randomly
"""
from collections import OrderedDict
import logging
from numpy.random import choice
from tornado import gen
from diplomacy_research.models.state_space import get_order_frequency
from diplomacy_research.players.player import Player

# Constants
LOGGER = logging.getLogger(__name__)

class RandomPlayer(Player):
    """ Random Player Class """

    def __init__(self, weighting_func=None, name=None):
        """ Constructor
            :param weighting_func: Optional. Weighting function (takes phase and order) to use custom weights.
            :param name: Optional. The name of this player.
        """
        if weighting_func:
            name = name or '%s/%s' % (self.__class__.__name__, weighting_func.__name__)
        Player.__init__(self, name)
        self.weighting_func = weighting_func

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
        orderable_locs = self.get_orderable_locations(state_proto, power_name)

        # Calculating the weights of each possible orders of each orderable locations
        # possible_orders_probs = {loc: list of probs}
        orders_probs_by_loc = self._get_possible_orders_probs(orderable_locs,
                                                              state_proto,
                                                              possible_orders_proto,
                                                              weighting_func=self.weighting_func)

        # Randomly selecting an order at each location according to their weights
        orders = []
        for loc in orderable_locs:
            orders += [choice(possible_orders_proto[loc].value, p=orders_probs_by_loc[loc])]

        # Returning orders
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

    @staticmethod
    def _get_possible_orders_probs(orderable_locs, state_proto, possible_orders_proto, weighting_func=None):
        """ Calculates the probability of each possible order for each orderable location
            :param orderable_locs: The list of orderable locations for power.
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param weighting_func: Optional. Weighting function (takes phase and order) to use custom weights.
            :return: A dictionary containing locs as key and a list of probs as value
        """
        # Getting possible orders for each location
        phase_type = state_proto.name[-1]

        # Calculating weights
        orders_probs_by_loc = OrderedDict()
        for loc in orderable_locs:
            loc_possible_orders = possible_orders_proto[loc].value
            if loc_possible_orders:
                nb_orders = len(loc_possible_orders)

                # Non-Uniform Weight
                if weighting_func:
                    order_probs = [weighting_func(phase_type, order) for order in loc_possible_orders]
                    order_probs = [prob / sum(order_probs) for prob in order_probs]
                # Uniform Weight
                else:
                    order_probs = [1. / nb_orders] * nb_orders

                orders_probs_by_loc[loc] = order_probs

        # Returning
        return orders_probs_by_loc

# =-=-=-=-=--=-=-=-=-=--=-=-=-=-=--=-=-=-=-=--=-=-=-=-=--=-=-=-=-=--
# Weighting functions

def weight_by_order_type(phase_type, order):
    """ Calculate the new weight for the order using the perc of orders with this order type for the given phase
        as calculated in the dataset. (e.g. 20% of orders in a movement phase are H)
        :param phase_type: The current phase type ('M', 'R', 'A')
        :param order: The order to evaluate
        :return: The new weight for the order
    """
    # Average of 10,000 games
    # M --- ['C: 1.535%', '-: 49.738%', 'D: 0.0%', 'R: 0.0%', 'B: 0.0%', 'H: 19.798%', 'S: 28.929%']
    # R --- ['C: 0.0%', '-: 0.0%', 'D: 2.428%', 'R: 3.213%', 'B: 0.0%', 'H: 94.36%', 'S: 0.0%']
    # A --- ['C: 0.0%', '-: 0.0%', 'D: 3.798%', 'R: 0.0%', 'B: 11.252%', 'H: 84.95%', 'S: 0.0%']
    probs = {
        'M': {'H': 0.19798, '-': 0.49738, 'S': 0.28929, 'C': 0.01535, 'R': 0.00000, 'B': 0.00000, 'D': 0.00000},
        'R': {'H': 0.94360, '-': 0.00000, 'S': 0.00000, 'C': 0.00000, 'R': 0.03213, 'B': 0.00000, 'D': 0.02427},
        'A': {'H': 0.84950, '-': 0.00000, 'S': 0.00000, 'C': 0.00000, 'R': 0.00000, 'B': 0.11252, 'D': 0.03798}
    }
    parts = order.split()
    if len(parts) < 3:
        return 0.00001
    order_type = parts[2]
    if phase_type in probs and order_type in probs[phase_type]:
        return probs[phase_type][order_type]
    return 0.00001

def weight_by_order_freq(phase_type, order):
    """ Calculate the new weight for the order using the number of times the order has been seen in the dataset.
        :param phase_type: The current phase type ('M', 'R', 'A')
        :param order: The order to evaluate
        :return: The new weight for the order
    """
    del phase_type      # Unused argument
    return get_order_frequency(order, no_press_only=True) + 1
