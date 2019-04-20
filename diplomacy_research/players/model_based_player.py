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
""" Model-Based Player
    - Contains a class representing a player that chooses to move by sampling and conditional
      its next move based on already selected orders
"""
import logging
import random
from tornado import gen
from diplomacy import Game
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto
from diplomacy_research.players.player import Player
from diplomacy_research.utils.model import merge_dicts

# Constants
LOGGER = logging.getLogger(__name__)

class ModelBasedPlayer(Player):
    """ ModelBased Player Class"""

    def __init__(self, policy_adapter, player_seed=None, noise=None, temperature=None, schedule=None,
                 dropout_rate=None, use_beam=None, name=None):
        """ Constructor
            :param policy_adapter: The policy adapter (instance) to evaluate the action to select
            :param player_seed: The seed to apply to the player to compute a deterministic mask.
            :param noise: The sigma of the additional noise to apply to the intermediate layers.
            :param temperature: The temperature to apply to the logits. (Defaults to schedule, otherwise uses 0.)
            :param schedule: The temperature schedule to use. List of (prob, temperature)
                             e.g. [(0.75, 1.), (1., 0.)] means
                                - 1) 75% chance of using a temperature of 1
                                - 2) if not 1), then 100% of using temperature of 0.
            :param dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
            :param use_beam: Boolean that indicates that we want to use a beam search,
            :param name: Optional. The name of this player.
            :type policy_adapter: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter
        """
        # pylint: disable=too-many-arguments
        Player.__init__(self, name)
        self.policy_adapter = policy_adapter
        self._player_seed = player_seed or 0
        self._noise = noise or 0.
        self._temperature = None
        self._schedule = [(1., 0.)]
        self._dropout_rate = dropout_rate or 0.
        self._use_beam = use_beam

        # Using a temperature of 1. if using beam search without a temperature
        if use_beam and temperature is None:
            temperature = 1.

        # Use temperature if provided, otherwise use schedule, otherwise defaults to greedy
        if temperature is not None:
            self._temperature = temperature
            self._schedule = [(1., temperature)]
        elif schedule is not None:
            self._schedule = schedule

    # ---------- Properties -------------
    @property
    def is_trainable(self):
        """ Returns a boolean that indicates if the player wants to be trained or not """
        if self.policy_adapter is not None and self.policy_adapter.is_trainable:
            return True
        return False

    @property
    def temperature(self):
        """ Getter - temperature """
        if self._temperature is not None:
            return self._temperature

        # Otherwise, computing it from schedule
        remaining = 1.
        weighted_temp = 0.
        for prob, temp in self._schedule:
            weighted_temp += remaining * prob * temp
            remaining -= max(0, remaining * prob)
        return weighted_temp

    # ---------- Methods -------------
    @gen.coroutine
    def get_beam_orders(self, game, power_names, *, retry_on_failure=True, **kwargs):
        """ Finds all the beams with their probabilities returned by the diverse beam search for the selected power(s)
            Beams are ordered by score (highest first).
            :param game: The game object
            :param power_names: A list of power names we are playing, or alternatively a single power name.
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
            :return: 1) If power_names is a string, a tuple of beam orders, and of beam probabilities
                     2) If power_names is a list, a list of list which contains beam orders and beam probabilities
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

        # Getting beam orders
        beam_orders_probs = yield [self.get_beam_orders_with_proto(state_proto,
                                                                   power_name,
                                                                   phase_history_proto,
                                                                   possible_orders_proto,
                                                                   retry_on_failure=retry_on_failure,
                                                                   **kwargs) for power_name in power_names]
        beam_orders_probs = beam_orders_probs[0] if is_single_power else beam_orders_probs
        return beam_orders_probs

    @gen.coroutine
    def get_beam_orders_with_proto(self, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Finds all the beams with their probabilities returned by the diverse beam search for the selected power
            Beams are ordered by score (highest first).
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return:A tuple consisting of
                     1) A list of beams (i.e. a list of selected orders for each beam)
                     2) A list of probability (the probability of selecting each beam)
        """
        orderable_locs = self.get_orderable_locations(state_proto, power_name)
        return (yield self.policy_adapter.get_beam_orders(orderable_locs,
                                                          state_proto,
                                                          power_name,
                                                          phase_history_proto,
                                                          possible_orders_proto,
                                                          **self._get_kwargs(**kwargs)))

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
                - use_beam: If set. Override the use_beam to use for the model based player.
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
        return (yield self.policy_adapter.get_orders(orderable_locs,
                                                     state_proto,
                                                     power_name,
                                                     phase_history_proto,
                                                     possible_orders_proto,
                                                     **self._get_kwargs(**kwargs)))

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
        # Trying to query actor-critic model for state-value
        if self.policy_adapter and self.policy_adapter.has_value_model:
            return (yield self.policy_adapter.get_state_value(state_proto,
                                                              power_name,
                                                              phase_history_proto,
                                                              possible_orders_proto,
                                                              **self._get_kwargs(**kwargs)))

        # Otherwise, returning 0. with a warning
        LOGGER.warning('There are no models available to query state value. Returning a value of 0.')
        return 0.

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
        # Using default player_seed, noise, temperature, and dropout_rate.
        # power_orders is a list of tuples (orders, policy_details)
        power_orders = yield [self.policy_adapter.get_orders(self.get_orderable_locations(state_proto, power_name),
                                                             state_proto,
                                                             power_name,
                                                             phase_history_proto,
                                                             possible_orders_proto,
                                                             retry_on_failure=False) for power_name in game.powers]
        return {power_name: orders[0] for power_name, orders in zip(game.powers.keys(), power_orders)}

    def _get_kwargs(self, player_seed=None, noise=None, temperature=None, dropout_rate=None, use_beam=None,
                    **other_kwargs):
        """ Selects between the default value provided at initialization and the potential override in kwargs """
        # Selecting temperature
        if temperature is None:
            for prob, temp in self._schedule:
                if random.random() <= prob:
                    temperature = temp
                    break
            else:
                temperature = 0.

        # Starting with player.kwargs, then overriding fields
        kwargs = self.kwargs
        if player_seed is not None:
            kwargs['player_seed'] = player_seed
        if noise is not None:
            kwargs['noise'] = noise
        kwargs['temperature'] = temperature
        if dropout_rate is not None:
            kwargs['dropout_rate'] = dropout_rate

        # Setting use_beam
        if use_beam is not None:
            kwargs['use_beam'] = use_beam
        elif self._use_beam is not None:
            kwargs['use_beam'] = self._use_beam

        # Merging with other kwargs and returning
        return merge_dicts(kwargs, other_kwargs)
