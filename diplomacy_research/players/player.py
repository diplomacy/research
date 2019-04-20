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
""" Player
    - Contains the abstract class representing a player
"""
from abc import abstractmethod, ABCMeta
import logging
from tornado import gen
from diplomacy import Game
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto, get_orderable_locs_for_powers
from diplomacy_research.utils.model import strip_keys
from diplomacy_research.utils.openings import get_standard_openings

# Constants
LOGGER = logging.getLogger(__name__)

class Player(metaclass=ABCMeta):
    """ Abstract Player Class """

    def __init__(self, name=None):
        """ Constructor """
        self._player_seed = 0
        self._noise = 0.
        self._temperature = 0.
        self._dropout_rate = 0.
        self.name = name or self.__class__.__name__

    # ---------- Properties -------------
    @property
    @abstractmethod
    def is_trainable(self):
        """ Returns a boolean that indicates if the player wants to be trained or not """
        raise NotImplementedError()

    @property
    def player_seed(self):
        """ Getter - player_seed """
        return self._player_seed

    @player_seed.setter
    def player_seed(self, value):
        """ Setter - player_seed """
        self._player_seed = int(value)

    @property
    def noise(self):
        """ Getter - noise """
        return self._noise

    @property
    def temperature(self):
        """ Getter - temperature """
        return self._temperature

    @property
    def dropout_rate(self):
        """ Getter - dropout_rate """
        return self._dropout_rate

    @property
    def kwargs(self):
        """ Getter - kwargs"""
        return {'player_seed': self.player_seed,
                'noise': self.noise,
                'temperature': self.temperature,
                'dropout_rate': self.dropout_rate}

    # ---------- Methods -------------
    @gen.coroutine
    def get_orders(self, game, power_names, *, retry_on_failure=True, **kwargs):
        """ Gets the orders the power(s) should play.
            :param game: The game object
            :param power_names: A list of power names we are playing, or alternatively a single power name.
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
                - with_draw: If set, also returns whether to accept a draw or not
            :return: One of the following:
                1) If power_name is a string and with_draw == False (or is not set):
                    - A list of orders the power should play
                2) If power_name is a list and with_draw == False (or is not set):
                    - A list of list, which contains orders for each power
                3) If power_name is a string and with_draw == True:
                    - A tuple of 1) the list of orders for the power, 2) a boolean to accept a draw or not
                4) If power_name is a list and with_draw == True:
                    - A list of tuples, each tuple having the list of orders and the draw boolean
            :type game: diplomacy.Game
        """
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)

        # Determining if we have a single or multiple powers
        if not isinstance(power_names, list):
            is_single_power = True
            power_names = [power_names]
        else:
            is_single_power = False

        # Getting orders (and optional draw)
        orders_with_maybe_draw = yield [self.get_orders_with_proto(state_proto,
                                                                   power_name,
                                                                   phase_history_proto,
                                                                   possible_orders_proto,
                                                                   retry_on_failure=retry_on_failure,
                                                                   **kwargs) for power_name in power_names]

        # Returning a single instance, or a list
        orders_with_maybe_draw = orders_with_maybe_draw[0] if is_single_power else orders_with_maybe_draw
        return orders_with_maybe_draw

    @gen.coroutine
    def get_orders_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Gets the orders the power should play
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power we are playing
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
                - with_draw: If set, also returns if we should accept a draw or not.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: If with_draw is not set or is False:
                        - the list of orders the power should play (e.g. ['A PAR H', 'A MAR - BUR', ...])
                     If with_draw is set, a tuple consisting of
                        1) the list of orders the power should play (e.g. ['A PAR H', 'A MAR - BUR', ...])
                        2) a boolean that indicates if we should accept a draw or not
        """
        with_draw = kwargs.get('with_draw', False)
        orders, policy_details = yield self.get_orders_details_with_proto(state_proto,
                                                                          power_name,
                                                                          phase_history_proto,
                                                                          possible_orders_proto,
                                                                          **strip_keys(kwargs, ['with_state_value']))

        # Returning orders, or orders and draw_action
        if not with_draw:
            return orders
        return orders, policy_details['draw_action']

    @gen.coroutine
    def get_policy_details(self, game, power_names, *, retry_on_failure=True, **kwargs):
        """ Gets the details of the current policy
            :param game: The game object
            :param power_names: A list of power names we are playing, or alternatively a single power name.
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
            :return: 1) If power_names is a string, the policy details
                        ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                     2) If power_names is a list, a list of policy details, one for each power.
            :type game: diplomacy.Game
        """
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)

        # Determining if we have a single or multiple powers
        if not isinstance(power_names, list):
            is_single_power = True
            power_names = [power_names]
        else:
            is_single_power = False

        # Getting policy details
        policy_details = yield [self.get_policy_details_with_proto(state_proto,
                                                                   power_name,
                                                                   phase_history_proto,
                                                                   possible_orders_proto,
                                                                   retry_on_failure=retry_on_failure,
                                                                   **kwargs)
                                for power_name in power_names]
        policy_details = policy_details[0] if is_single_power else policy_details
        return policy_details

    @gen.coroutine
    def get_policy_details_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto,
                                      **kwargs):
        """ Gets the details of the current policy
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power we are playing
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
        """
        _, policy_details = yield self.get_orders_details_with_proto(state_proto,
                                                                     power_name,
                                                                     phase_history_proto,
                                                                     possible_orders_proto,
                                                                     **strip_keys(kwargs, ['with_state_value']))
        return policy_details

    @abstractmethod
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
        raise NotImplementedError()

    @gen.coroutine
    def get_state_value(self, game, power_names, *, retry_on_failure=True, **kwargs):
        """ Calculates the player's value of the state of the game for the given power(s)
            :param game: A game object
            :param power_names: A list of power names for which we want the value, or alternatively a single power name.
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :param kwargs: Additional optional kwargs:
                - player_seed: If set. Override the player_seed to use for the model based player.
                - noise: If set. Override the noise to use for the model based player.
                - temperature: If set. Override the temperature to use for the model based player.
                - dropout_rate: If set. Override the dropout_rate to use for the model based player.
            :return: 1) If power_names is a string, a single float representing the value of the state for the power
                     2) If power_names is a list, a list of floats representing the value for each power.
            :type game: diplomacy.Game
        """
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_order_proto = extract_possible_orders_proto(game)

        # Determining if we have a single or multiple powers
        if not isinstance(power_names, list):
            is_single_power = True
            power_names = [power_names]
        else:
            is_single_power = False

        # Getting state value
        state_value = yield [self.get_state_value_with_proto(state_proto,
                                                             power_name,
                                                             phase_history_proto,
                                                             possible_order_proto,
                                                             retry_on_failure=retry_on_failure,
                                                             **kwargs)
                             for power_name in power_names]
        state_value = state_value[0] if is_single_power else state_value
        return state_value

    @abstractmethod
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
        raise NotImplementedError()

    @gen.coroutine
    def get_opening_orders(self):
        """ Returns a dictionary of power_name: [orders] for each power
            The orders represent the opening orders that would have been submitted by the player
        """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)

        # Retrieving all orders
        # Not using kwargs - Using default player_seed, noise, temperature, and dropout_rate.
        power_orders = yield [self.get_orders_with_proto(state_proto,
                                                         power_name,
                                                         phase_history_proto,
                                                         possible_orders_proto,
                                                         retry_on_failure=False) for power_name in game.powers]
        return {power_name: orders for power_name, orders in zip(game.powers.keys(), power_orders)}

    @gen.coroutine
    def check_openings(self):
        """ Validates the opening move of a player against the database of standard openings """
        nb_pass, nb_total = 0, 0
        opening_orders = yield self.get_opening_orders()

        # Validating against database
        for power_name in opening_orders:
            standard_openings = get_standard_openings(power_name)
            normalized_orders = sorted([order for order in opening_orders[power_name] if len(order.split()) >= 2],
                                       key=lambda order_string: order_string.split()[1])

            # Displaying status
            if normalized_orders in standard_openings:
                LOGGER.info('%s - OK', power_name)
                nb_pass += 1
            else:
                LOGGER.info('%s - FAIL -- %s', power_name, normalized_orders)
            nb_total += 1

        # Displaying total success
        LOGGER.info('Summary: %d / %d powers submitted orders in the database.', nb_pass, nb_total)

    @staticmethod
    def get_nb_centers(state_proto, power_name):
        """ Calculates the number of supply centers a power has
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power for which we want the value of the state
            :return: A integer representing the number of supply centers the power controls
        """
        return len(state_proto.centers[power_name].value)

    @staticmethod
    def get_orderable_locations(state_proto, power_name):
        """ Calculates the list of locations where unit can receive orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The name of the power for which we want orderable locations
            :return: A list of locations where units can receive orders
        """
        _, orderable_locs = get_orderable_locs_for_powers(state_proto, [power_name])
        return orderable_locs[power_name]
