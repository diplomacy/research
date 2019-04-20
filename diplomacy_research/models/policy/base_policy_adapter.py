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
""" Policy Adapter
    - Implements an instance of a policy adapter using an active or a frozen Tensorflow graph
"""
from abc import ABCMeta, abstractmethod
import logging
from tornado import gen
from diplomacy_research.models.base_adapter import BaseAdapter
from diplomacy_research.models.datasets.queue_dataset import QueueDataset

# Constants
LOGGER = logging.getLogger(__name__)

class BasePolicyAdapter(BaseAdapter, metaclass=ABCMeta):
    """ Allows the evaluation of a policy adapter from a TensorFlow graph and session """

    @property
    def has_value_model(self):
        """ Indicates if the policy model has a value model (actor-critic) or is just a policy model """
        return bool('state_value' in self.outputs or not isinstance(self.feedable_dataset, QueueDataset))

    @abstractmethod
    def tokenize(self, order):
        """ Returns the tokens use by the adapter for a specific order """
        raise NotImplementedError()

    @abstractmethod
    def _decode_policy(self, locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Returns the output of the Policy Model decoder
            :param locs: A list of locations for which we want orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - with_state_value: Boolean that indicates to also query the value function.
                - use_beam: Boolean that indicates that we want to use a beam search,
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return: A future (fetches) to yield on.
        """
        # pylint: disable=too-many-arguments
        raise NotImplementedError()

    @abstractmethod
    @gen.coroutine
    def get_orders(self, locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Finds the orders to submit at each location given the current state
            Orderings are calculated by defining an ordering and computing the next unit order conditioned on the
            orders already selected
            :param locs: A list of locations for which we want orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - with_state_value: Boolean that indicates to also query the value function.
                - use_beam: Boolean that indicates that we want to use a beam search,
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False and with_state_value=False (default), a tuple consisting of:
                     1) A list of the selected orders
                     2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                - if prefetch=False and with_state_value=True, a tuple consisting of:
                     1) A list of the selected orders
                     2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                     3) The state value for the given state
        """
        raise NotImplementedError()

    @gen.coroutine
    def get_beam_orders(self, locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Finds all the beams with their probabilities returned by the diverse beam search
            Beams are ordered by score (highest first).
            :param locs: A list of locations for which we want orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - with_state_value: Boolean that indicates to also query the value function.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False and with_state_value=False (default), a tuple consisting of:
                     1) A list of beams (i.e. a list of selected orders for each beam)
                     2) A list of probability (the probability of selecting each beam)
                - if prefetch=False and with_state_value=True, a tuple consisting of:
                     1) A list of beams (i.e. a list of selected orders for each beam)
                     2) A list of probability (the probability of selecting each beam)
                     3) The state value for the given state
        """
        raise NotImplementedError()

    @abstractmethod
    @gen.coroutine
    def get_updated_policy_details(self, state_proto, power_name, phase_history_proto, possible_orders_proto,
                                   old_policy_details=None, submitted_orders=None, **kwargs):
        """ Computes the current policy details (locs, tokens, log_probs) under the current model
            Either one of 1) old_policy_details or 2) submitted_orders must be submitted to extract the locs and tokens

            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param old_policy_details: (Optional) Some policy details
                                        ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
            :param submitted_orders: (Optional) A list of submitted orders ['A PAR - BUR', 'A MAR H']
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False, The corresponding updated policy details
                                    ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
        """
        raise NotImplementedError()

    @abstractmethod
    @gen.coroutine
    def expand(self, confirmed_orders, locs, state_proto, power_name, phase_history_proto, possible_orders_proto,
               **kwargs):
        """ Computes the conditional probability of possible orders for each loc given the confirmed orders.
            :param confirmed_orders: The list of orders on which to condition the probs (e.g. ['A PAR H', 'A MAR - SPA']
            :param locs: The locations for which we want to compute probabilities
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the probabilities
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False,
                    A dictionary with every location in locs as key, and a list of tuples where each tuple is composed
                     of 1) an order, 2) the order conditional probability, 3) the conditional log probs of each token
                        e.g. {'PAR': [('A PAR H', 0.000, [...]), ('A PAR - BUR', 0.000, [...]), ...]}
        """
        raise NotImplementedError()

    @abstractmethod
    @gen.coroutine
    def get_state_value(self, state_proto, power_name, phase_history_proto, possible_orders_proto=None, **kwargs):
        """ Computes the value of the current state for a given power
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want to retrieve the value
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False, a float representing the value of the state of the game to the specified power
        """
        raise NotImplementedError()
