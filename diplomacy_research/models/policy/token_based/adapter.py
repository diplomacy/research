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
""" TokenBased Policy Adapter
    - Implements an instance of a policy adapter to connect to a token_based model
"""
from collections import OrderedDict
import logging
import numpy as np
from numpy.random import choice
from tornado import gen
from diplomacy_research.models.policy.base_policy_adapter import BasePolicyAdapter
from diplomacy_research.models.policy.base_policy_model import OrderProbTokenLogProbs, TRAINING_DECODER, \
    GREEDY_DECODER, SAMPLE_DECODER
from diplomacy_research.models.policy.token_based.model import TokenBasedPolicyModel
from diplomacy_research.models.state_space import token_to_ix, get_order_tokens, TOKENS_PER_ORDER, GO_ID, EOS_ID, \
    PAD_ID, get_orderable_locs_for_powers, ix_to_token
from diplomacy_research.proto.diplomacy_proto.common_pb2 import MapStringList
from diplomacy_research.utils.cluster import CompletedFuture, process_fetches_dict
from diplomacy_research.utils.model import logsumexp, apply_temperature, strip_keys, assert_normalized

# Constants
LOGGER = logging.getLogger(__name__)

class PolicyAdapter(BasePolicyAdapter):
    """ Adapter to connect to a TokenBased model """

    @staticmethod
    def get_signature():
        """ Returns the signature of all the possible calls using this adapter
            Format: { method_signature_name: {'placeholders': {name: (value, numpy_dtype)},
                                              'outputs': [output_name, output_name] } }
            e.g. {'policy_evaluate': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                      'outputs: ['selected_tokens', 'log_probs', 'draw_prob']}}
        """
        return {'policy_evaluate': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                    'outputs': ['selected_tokens',
                                                'log_probs',
                                                'draw_prob']},
                'policy_beam_search': {'placeholders': {'decoder_type': ([GREEDY_DECODER], np.uint8)},
                                       'outputs': ['beam_tokens',
                                                   'beam_log_probs',
                                                   'draw_prob']},
                'policy_evaluate_with_state_value': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                                     'outputs': ['selected_tokens',
                                                                 'log_probs',
                                                                 'draw_prob',
                                                                 'state_value']},
                'policy_beam_search_with_state_value': {'placeholders': {'decoder_type': ([GREEDY_DECODER], np.uint8)},
                                                        'outputs': ['beam_tokens',
                                                                    'beam_log_probs',
                                                                    'draw_prob',
                                                                    'state_value']},
                'policy_expand': {'placeholders': {'decoder_type': ([TRAINING_DECODER], np.uint8)},
                                  'outputs': ['logits']},
                'policy_log_probs': {'placeholders': {'decoder_type': ([TRAINING_DECODER], np.uint8)},
                                     'outputs': ['log_probs', 'draw_prob']},
                'policy_get_value': {'placeholders': {'decoder_type': ([GREEDY_DECODER], np.uint8)},
                                     'outputs': ['state_value']}}

    def tokenize(self, order):
        """ Returns the tokens use by the adapter for a specific order """
        try:
            tokens = [token_to_ix(order_token) for order_token in get_order_tokens(order)] + [EOS_ID]
            tokens += [PAD_ID] * (TOKENS_PER_ORDER - len(tokens))
            return tokens
        except KeyError:
            LOGGER.warning('[tokenize] Order "%s" is invalid. Skipping.')
            return [EOS_ID] + [PAD_ID] * (TOKENS_PER_ORDER - 1)

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
        is_prefetching = kwargs.get('prefetch', False)

        # No locations provided, we can return early
        if not locs:
            ret_val = None
            return CompletedFuture(ret_val) if is_prefetching else ret_val

        # Getting feedable item
        feedable_item = self.feedable_dataset.get_feedable_item(locs,
                                                                state_proto,
                                                                power_name,
                                                                phase_history_proto,
                                                                possible_orders_proto,
                                                                **kwargs)
        if not feedable_item:
            LOGGER.warning('The method .get_feedable_item() did not return an item to feed to the model.')
            LOGGER.warning('Make sure you have provided the correct locs and a list of possible orders')
            ret_val = None
            return CompletedFuture(ret_val) if is_prefetching else ret_val

        # Queue
        with_state_value = kwargs.get('with_state_value', False)
        use_beam = kwargs.get('use_beam', False)
        queue_name = {(False, False): 'policy_evaluate',
                      (False, True): 'policy_evaluate_with_state_value',
                      (True, False): 'policy_beam_search',
                      (True, True): 'policy_beam_search_with_state_value'}[(use_beam, with_state_value)]
        return self.feedable_dataset.get_results(queue_name, feedable_item, **kwargs)

    @staticmethod
    def _process_fetches(decode_fetches):
        """ Decodes the fetches returned by self._decode_policy()
            :param decode_fetches: The fetches returned by self._decode_policy()
            :return: An ordered dict with the location as key, and an OrderProbTokenLogProbs as value
        """
        # If we get an empty list, we can't decode it
        if not decode_fetches:
            return decode_fetches

        tokens, log_probs = decode_fetches[:2]
        decoded_results = TokenBasedPolicyModel._decode(selected_tokens=np.array([tokens]),                             # pylint: disable=protected-access
                                                        log_probs=np.array([log_probs]))
        return decoded_results['decoded_orders'][0]

    @staticmethod
    def _process_single_beam_fetches(decode_fetches, temperature=0.):
        """ Decodes the beam fetches returned self._decode_policy() - This samples the beam to use based on a temp.
            :param decode_fetches: The fetches returned by self._decode_policy()
            :return: An ordered dict with the location as key, and an OrderProbTokenLogProbs as value
        """
        # If we get an empty list, we can't decode it
        if not decode_fetches:
            return decode_fetches

        beam_tokens, beam_log_probs = decode_fetches[:2]

        # Computing probabilities after applying temperature
        probs = np.exp(beam_log_probs - logsumexp(beam_log_probs))
        adj_probs = apply_temperature(probs, temperature=temperature).tolist()
        nb_probs = len(probs)

        # Sampling according to probs
        selected_beam_id = choice(range(nb_probs), p=assert_normalized(adj_probs))

        # Decoding that specific beam
        # Assigning probability mass equally over all orders in beam
        selected_beam_tokens = np.array([beam_tokens[selected_beam_id]])
        selected_beam_log_probs = np.zeros_like(selected_beam_tokens)
        decoded_results = TokenBasedPolicyModel._decode(selected_tokens=selected_beam_tokens,                           # pylint: disable=protected-access
                                                        log_probs=selected_beam_log_probs)['decoded_orders'][0]

        # Adjusting log probs to make it uniform over all locs
        nb_locs = len(decoded_results)
        adj_log_probs = beam_log_probs[selected_beam_id] / max(1, nb_locs * TOKENS_PER_ORDER)
        decoded_results = {loc: OrderProbTokenLogProbs(order=decoded_results[loc].order,
                                                       probability=decoded_results[loc].probability,
                                                       log_probs=[adj_log_probs] * TOKENS_PER_ORDER)
                           for loc in decoded_results}
        return decoded_results

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
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_orders'
        with_state_value = kwargs.get('with_state_value', False)

        # Getting fetches
        if not is_postfetching:
            locs = [loc[:3] for loc in locs]

            # Running policy model
            fetches['%s/decode_fetches' % fetch_prefix] = self._decode_policy(locs,
                                                                              state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              **kwargs)
            # Prefetching - We only return the fetches
            if is_prefetching:
                return fetches

            # Otherwise, we yield on the fetches
            fetches = yield process_fetches_dict(self.feedable_dataset, fetches)

        # Variables
        selected_orders = []
        policy_details = {'locs': [],
                          'tokens': [],
                          'log_probs': [],
                          'draw_action': False,
                          'draw_prob': 0.}
        state_value = 0.

        # Processing
        decode_fetches = fetches['%s/decode_fetches' % fetch_prefix]
        if decode_fetches is None:
            return tuple([selected_orders, policy_details] + ([state_value] if with_state_value else []))

        if kwargs.get('use_beam', False):
            results = self._process_single_beam_fetches(decode_fetches, temperature=kwargs.get('temperature', 0.))
        else:
            results = self._process_fetches(decode_fetches)

        # Building policy details based on returned locations
        for loc in results:
            order_prob_token_log_probs = results[loc]

            # Splitting
            order = order_prob_token_log_probs.order
            log_probs = order_prob_token_log_probs.log_probs

            # Building the policy details
            selected_orders += [order]
            policy_details['locs'] += [loc]
            policy_details['tokens'] += self.tokenize(order)
            policy_details['log_probs'] += list(log_probs)

        # Getting draw action and probability
        policy_details['draw_action'] = bool(decode_fetches[2] >= 0.5)
        policy_details['draw_prob'] = decode_fetches[2]

        # Getting state value
        if with_state_value:
            state_value = decode_fetches[-1]

        # Returning sampled orders with the policy details
        return tuple([selected_orders, policy_details] + ([state_value] if with_state_value else []))

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
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_orders'
        with_state_value = kwargs.get('with_state_value', False)

        # Getting fetches
        if not is_postfetching:
            locs = [loc[:3] for loc in locs]

            # Running policy model
            fetches['%s/decode_fetches' % fetch_prefix] = self._decode_policy(locs,
                                                                              state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              use_beam=True,
                                                                              **strip_keys(kwargs, ['use_beam']))
            # Prefetching - We only return the fetches
            if is_prefetching:
                return fetches

            # Otherwise, we yield on the fetches
            fetches = yield process_fetches_dict(self.feedable_dataset, fetches)

        # Variables
        beams, adj_probs, state_value = [], [], 0.

        # Processing
        decode_fetches = fetches['%s/decode_fetches' % fetch_prefix]        # (beam_orders, beam_log_probs, draw, value)
        if decode_fetches is None:
            return tuple([beams, adj_probs] + ([state_value] if with_state_value else []))

        # Computing adj probabilities
        beam_orders, beam_log_probs = decode_fetches[:2]
        probs = np.exp(beam_log_probs - logsumexp(beam_log_probs))
        adj_probs = apply_temperature(probs, temperature=1.).tolist()

        # Decoding
        for beam_candidates in beam_orders:
            orders_for_this_candidate = []
            nb_locs = len(beam_candidates) // TOKENS_PER_ORDER

            # Decoding each token
            for loc_ix in range(nb_locs):
                order_tokens = beam_candidates[loc_ix * TOKENS_PER_ORDER:(loc_ix + 1) * TOKENS_PER_ORDER]
                order_str = ' '.join([ix_to_token(token) for token in order_tokens if token > EOS_ID])
                if order_str:
                    orders_for_this_candidate += [order_str]
            beams += [orders_for_this_candidate]

        # Getting state value
        if with_state_value:
            state_value = decode_fetches[-1]

        # Returning
        return tuple([beams, adj_probs] + ([state_value] if with_state_value else []))

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
        assert self.feedable_dataset.has_queue('policy_log_probs'), 'Unable to get supervised log probs'

        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_updated_policy_details'

        # Setting tokens and actual locs
        if old_policy_details:
            actual_locs = old_policy_details['locs']
            tokens = old_policy_details['tokens']

        # Using submitted orders
        else:
            actual_locs, tokens = [], []
            for order in submitted_orders:
                actual_locs += [order.split()[1][:3]] if len(order.split()) >= 2 else []
                tokens += self.tokenize(order)

        # Getting fetches
        if not is_postfetching:

            if not old_policy_details and not submitted_orders:
                LOGGER.warning('Unable to compute policy details without old policy details or submitted orders.')
                ret_val = {'locs': [],
                           'tokens': [],
                           'log_probs': [],
                           'draw_action': False,
                           'draw_prob': 0.}
                return {'%s/ret_val' % fetch_prefix: CompletedFuture(ret_val)} if is_prefetching else ret_val

            # In adjustment phase, the locs are all the orderable locs
            if state_proto.name[-1] == 'A':
                locs, _ = get_orderable_locs_for_powers(state_proto, [power_name])
            elif old_policy_details:
                locs = old_policy_details['locs']
            else:
                locs = [order.split()[1][:3] for order in submitted_orders if len(order.split()) >= 2]

            # Getting feedable item
            feedable_item = self.feedable_dataset.get_feedable_item(locs,
                                                                    state_proto,
                                                                    power_name,
                                                                    phase_history_proto,
                                                                    possible_orders_proto,
                                                                    **kwargs)
            if not feedable_item:
                ret_val = {'locs': [],
                           'tokens': [],
                           'log_probs': [],
                           'draw_action': False,
                           'draw_prob': 0.}
                return {'%s/ret_val' % fetch_prefix: CompletedFuture(ret_val)} if is_prefetching else ret_val

            feedable_item['decoder_inputs'] = [GO_ID] + tokens
            feedable_item['decoder_lengths'] = len(tokens)

            # Querying model
            queue_name = 'policy_log_probs'
            fetches['%s/log_probs_fetches' % fetch_prefix] = self.feedable_dataset.get_results(queue_name,
                                                                                               feedable_item,
                                                                                               **kwargs)

            # Prefetching - We only return the fetches
            if is_prefetching:
                return fetches

            # Otherwise, we yield on the fetches
            fetches = yield process_fetches_dict(self.feedable_dataset, fetches)

        # Processing fetches
        if '%s/ret_val' % fetch_prefix in fetches:
            return fetches['%s/ret_val' % fetch_prefix]

        new_log_probs, new_draw_prob = fetches['%s/log_probs_fetches' % fetch_prefix]
        new_log_probs = new_log_probs[:len(actual_locs) * TOKENS_PER_ORDER].tolist()

        # Validating
        assert submitted_orders is not None or len(new_log_probs) == len(old_policy_details['log_probs'])

        # Returning
        return {'locs': actual_locs,
                'tokens': tokens,
                'log_probs': new_log_probs,
                'draw_action': old_policy_details['draw_action'] if old_policy_details else bool(new_draw_prob >= 0.5),
                'draw_prob': new_draw_prob}

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
        # pylint: disable=too-many-nested-blocks
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'expand'

        # Locations
        locs = [loc[:3] for loc in locs]
        confirmed_locs = [order.split()[1][:3] for order in confirmed_orders]

        # Getting fetches
        if not is_postfetching:

            confirmed_tokens = []
            for order in confirmed_orders:
                confirmed_tokens += self.tokenize(order)

            # Building a list of conditional probs we want to expand
            # e.g. A PAR H, A PAR - MAR, A PAR - BUR
            # we want P('H' | 'A PAR'), P('MAR', 'A PAR -'), ...

            # 1) We compute all the prefix (RHS of the prob) and we only expand where count is > 1
            prefix_count = {}                     # {prefix => count}
            for loc in locs:
                for order in possible_orders_proto[loc].value:
                    tokens = [-1 + -1 * locs.index(loc)] + self.tokenize(order)     # e.g. [-2, 25, 4, 25, ...]
                    nb_tokens = len(tokens)
                    for token_ix in range(1, nb_tokens):
                        prefix = tuple(tokens[:token_ix])
                        prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

            # 2) Building the list of feedable items only for probs we need to expand
            feedable_items = OrderedDict()      # {prefix => feedable_item}
            items_to_expand = OrderedDict()     # {prefix => set of next available token}
            for loc in locs:                    # pylint: disable=too-many-nested-blocks
                for order in possible_orders_proto[loc].value:
                    tokens = [-1 + -1 * locs.index(loc)] + self.tokenize(order)     # e.g. [-2, 25, 4, 25, ...]
                    nb_tokens = len(tokens)

                    for token_ix in range(1, nb_tokens):
                        prefix = tuple(tokens[:token_ix])
                        if prefix_count.get(prefix) > 1 and prefix not in feedable_items:
                            feedable_item = self.feedable_dataset.get_feedable_item(confirmed_locs + [loc],
                                                                                    state_proto,
                                                                                    power_name,
                                                                                    phase_history_proto,
                                                                                    possible_orders_proto,
                                                                                    **kwargs)

                            # The teacher forcing is GO_ID, the confirmed orders prefix, the actual prefix, and a dummy
                            feedable_item['decoder_inputs'] = [GO_ID] + confirmed_tokens + list(prefix[1:]) + [PAD_ID]
                            feedable_item['decoder_lengths'] = len(confirmed_tokens) + len(prefix[1:]) + 1

                            # Keeping a list of orders using the prefix
                            for possible_order in possible_orders_proto[loc].value:
                                new_tokens = [-1 + -1 * locs.index(loc)] + self.tokenize(possible_order)
                                if prefix == tuple(new_tokens[:token_ix]):
                                    items_to_expand.setdefault(prefix, set())
                                    items_to_expand[prefix].add(new_tokens[token_ix])

                            # Storing feedable item
                            feedable_items[prefix] = feedable_item

            # 3) Removing items_to_expand with only 1 items
            # We know for sure the probability will be 100%
            for prefix in list(items_to_expand.keys()):
                if len(items_to_expand[prefix]) == 1:
                    del items_to_expand[prefix]
                    del feedable_items[prefix]

            # 4) Running all the feedable items
            queue_name = 'policy_expand'
            fetches['%s/items_to_expand' % fetch_prefix] = CompletedFuture(items_to_expand)
            fetches['%s/results' % fetch_prefix] = [self.feedable_dataset.get_results(queue_name, item, **kwargs)
                                                    for item in feedable_items.values()]

            # Prefetching - We only return the fetches
            if is_prefetching:
                return fetches

            # Otherwise, we yield on the fetches
            fetches = yield process_fetches_dict(self.feedable_dataset, fetches)

        # Processing fetches
        def softmax(softmax_logits):
            """ Compute softmax values for the logits """
            e_x = np.exp(softmax_logits - softmax_logits.max(axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        items_to_expand = fetches['%s/items_to_expand' % fetch_prefix]
        results = fetches['%s/results' % fetch_prefix]
        (logits, ) = zip(*results)

        # 5) Computing probs
        probs = {}                      # {prefix: {loc: prob}}
        for prefix, logit in zip(items_to_expand.keys(), logits):

            # IndexError - Ignoring prefix
            if TOKENS_PER_ORDER * len(confirmed_locs) + len(prefix) - 1 >= len(logit):
                LOGGER.error('Got %d logits, but trying to access logit at index %d. Ignoring prefix.',
                             len(logit), TOKENS_PER_ORDER * len(confirmed_locs) + len(prefix) - 1)
                LOGGER.error('Prefix: %s - Confirmed locs: %s', prefix, confirmed_locs)
                continue

            tokens_to_expand = list(sorted(items_to_expand[prefix]))
            token_logits = logit[TOKENS_PER_ORDER * len(confirmed_locs) + len(prefix) - 1]

            # Only selecting the logits that we expect
            # There is currently a bug in the tokenization that could return additional tokens (Issue #331)
            masked_logits = []
            for token in tokens_to_expand:
                masked_logits += [token_logits[token]]
            token_probs = softmax(np.array(masked_logits, dtype=np.float32))

            # Computing the correct probabilities
            probs[prefix] = {}
            for token_ix, token in enumerate(tokens_to_expand):
                probs[prefix][token] = token_probs[token_ix]

        # 6) Computing the prob of each order at each location
        results = {}
        for loc in locs:
            results[loc] = []

            # Processing each possible order
            for order in possible_orders_proto[loc].value:
                tokens = [-1 + -1 * locs.index(loc)] + self.tokenize(order)
                nb_tokens = len(tokens)
                order_prob = 1.
                order_log_probs = []

                # Computing the total order probability and each token log probs
                for token_ix in range(1, nb_tokens):
                    prefix = tuple(tokens[:token_ix])
                    if prefix in probs and tokens[token_ix] in probs[prefix]:
                        order_prob *= probs[prefix][tokens[token_ix]]
                        order_log_probs += [np.log(np.maximum(probs[prefix][tokens[token_ix]], 1e-8))]
                    else:
                        order_log_probs += [0.]

                results[loc] += [OrderProbTokenLogProbs(order=order,
                                                        probability=order_prob,
                                                        log_probs=order_log_probs)]

            # Sorting loc by probability
            results[loc] = list(sorted(results[loc], key=lambda item: item.probability, reverse=True))

        # Returning
        return results

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
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_state_value'

        # Getting fetches
        if not is_postfetching:

            if not self.has_value_model:
                LOGGER.error('This model does not have a value function. Returning a value of 0.')
                return {'%s/ret_val' % fetch_prefix: CompletedFuture(0.)} if is_prefetching else 0.

            # Finding orderable locations
            locs, _ = get_orderable_locs_for_powers(state_proto, [power_name])

            # Building a list of empty possible orders
            # The value function should at most only use the first step of the decoder (time step 0)
            # So, we don't need to apply a mask
            if possible_orders_proto is None:
                possible_orders_proto = MapStringList().value                           # pylint: disable=no-member
                for loc in locs:
                    possible_orders_proto[loc].value.extend([])

            # Getting feedable item
            feedable_item = self.feedable_dataset.get_feedable_item(locs,
                                                                    state_proto,
                                                                    power_name,
                                                                    phase_history_proto,
                                                                    possible_orders_proto,
                                                                    **kwargs)
            if not feedable_item:
                LOGGER.warning('The method .get_feedable_item() did not return an item to feed to the model.')
                LOGGER.warning('Make sure you have provided the correct locs and a list of possible orders')
                LOGGER.warning('Returning a value of 0.')
                return {'%s/ret_val' % fetch_prefix: CompletedFuture(0.)} if is_prefetching else 0.

            # Selecting queue
            queue_name = 'policy_get_value'
            fetches['%s/state_value' % fetch_prefix] = self.feedable_dataset.get_results(queue_name,
                                                                                         feedable_item,
                                                                                         **kwargs)
            # Prefetching - We only return the fetches
            if is_prefetching:
                return fetches

            # Otherwise, we yield on the fetches
            fetches = yield process_fetches_dict(self.feedable_dataset, fetches)

        # Processing fetches
        if '%s/ret_val' % fetch_prefix in fetches:
            return fetches['%s/ret_val' % fetch_prefix]

        # Returning the fetched state value
        (state_value,) = fetches['%s/state_value' % fetch_prefix]
        return state_value
