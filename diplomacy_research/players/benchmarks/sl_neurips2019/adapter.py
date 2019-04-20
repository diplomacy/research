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
""" SL (NeurIPS 2019) Policy Adapter
    - Implements an instance of a policy adapter to connect to a order_based model
"""
import logging
import numpy as np
from numpy.random import choice
from tornado import gen
from diplomacy_research.models.policy.base_policy_adapter import BasePolicyAdapter
from diplomacy_research.models.policy.base_policy_model import OrderProbTokenLogProbs, TRAINING_DECODER, \
    GREEDY_DECODER, SAMPLE_DECODER
from diplomacy_research.models.policy.order_based.model import OrderBasedPolicyModel
from diplomacy_research.models.state_space import EOS_ID, PAD_ID, order_to_ix, ix_to_order
from diplomacy_research.utils.cluster import CompletedFuture, process_fetches_dict
from diplomacy_research.utils.model import logsumexp, apply_temperature, strip_keys, assert_normalized

# Constants
LOGGER = logging.getLogger(__name__)

class PolicyAdapter(BasePolicyAdapter):
    """ Adapter to connect to an OrderBased model """

    @staticmethod
    def get_signature():
        """ Returns the signature of all the possible calls using this adapter
            Format: { method_signature_name: {'placeholders': {name: (value, numpy_dtype)},
                                              'outputs': [output_name, output_name] } }
            e.g. {'policy_evaluate': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                      'outputs: ['selected_tokens', 'log_probs', 'draw_prob']}}
        """
        return {'policy_evaluate': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                    'outputs': ['selected_tokens', 'log_probs']},
                'policy_beam_search': {'placeholders': {'decoder_type': ([GREEDY_DECODER], np.uint8)},
                                       'outputs': ['beam_tokens', 'beam_log_probs']},
                'policy_expand': {'placeholders': {'decoder_type': ([TRAINING_DECODER], np.uint8)},
                                  'outputs': ['logits']},
                'policy_log_probs': {'placeholders': {'decoder_type': ([TRAINING_DECODER], np.uint8)},
                                     'outputs': ['log_probs']},
                'policy_get_value': {'placeholders': {'decoder_type': ([GREEDY_DECODER], np.uint8)},
                                     'outputs': ['state_value']}}

    def tokenize(self, order):
        """ Returns the tokens use by the adapter for a specific order """
        return [order_to_ix(order) or PAD_ID]

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
        use_beam = kwargs.get('use_beam', False)
        queue_name = {False: 'policy_evaluate',
                      True: 'policy_beam_search'}[use_beam]
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
        decoded_results = OrderBasedPolicyModel._decode(selected_tokens=np.array([tokens]),                             # pylint: disable=protected-access
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
        decoded_results = OrderBasedPolicyModel._decode(selected_tokens=selected_beam_tokens,                           # pylint: disable=protected-access
                                                        log_probs=selected_beam_log_probs)['decoded_orders'][0]

        # Adjusting log probs to make it uniform over all locs
        nb_locs = len(decoded_results)
        adj_log_probs = beam_log_probs[selected_beam_id] / max(1, nb_locs)
        decoded_results = {loc: OrderProbTokenLogProbs(order=decoded_results[loc].order,
                                                       probability=decoded_results[loc].probability,
                                                       log_probs=[adj_log_probs]) for loc in decoded_results}
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
                - use_beam: Boolean that indicates that we want to use a beam search, (Default; False)
                - retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
                - prefetch: Boolean that indicates to return a dictionary of fetches (str: PrefetchedItem/Future)
                - fetches: Dictionary of (str: future_results) that was computed with prefetch=True
            :return:
                - if prefetch=True, a dictionary of fetches (key as string, value is a future (or list) to yield on)
                - if prefetch=False, a tuple consisting of:
                     1) A list of the selected orders
                     2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
        """
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_orders'

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

        # Processing
        decode_fetches = fetches['%s/decode_fetches' % fetch_prefix]
        if decode_fetches is None:
            return selected_orders, policy_details

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
        policy_details['draw_action'] = False
        policy_details['draw_prob'] = 0.

        # Returning sampled orders with the policy details
        return selected_orders, policy_details

    @gen.coroutine
    def get_beam_orders(self, locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Finds all the beams with their probabilities returned by the diverse beam search
            Beams are ordered by score (highest first).
            :param locs: A list of locations for which we want orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
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
                - if prefetch=False, a tuple consisting of:
                     1) A list of beams (i.e. a list of selected orders for each beam)
                     2) A list of probability (the probability of selecting each beam)
        """
        # Determining if we need to prefetch or postfetch
        fetches = kwargs.get('fetches', {})
        is_prefetching = kwargs.get('prefetch', False)
        is_postfetching = fetches and not is_prefetching
        fetch_prefix = 'get_orders'

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
        beams, adj_probs = [], []

        # Processing
        decode_fetches = fetches['%s/decode_fetches' % fetch_prefix]        # (beam_orders, beam_log_probs)
        if decode_fetches is None:
            return beams, adj_probs

        # Computing adj probabilities
        beam_orders, beam_log_probs = decode_fetches[:2]
        probs = np.exp(beam_log_probs - logsumexp(beam_log_probs))
        adj_probs = apply_temperature(probs, temperature=1.).tolist()

        # Decoding
        for beam_candidates in beam_orders:
            beams += [[ix_to_order(order_ix) for order_ix in beam_candidates if order_ix > EOS_ID]]

        # Returning
        return beams, adj_probs

    @gen.coroutine
    def get_updated_policy_details(self, state_proto, power_name, phase_history_proto, possible_orders_proto,
                                   old_policy_details=None, submitted_orders=None, **kwargs):
        """ Computes the current policy details (locs, tokens, log_probs) under the current model """
        raise RuntimeError('Not available for this benchmark.')

    @gen.coroutine
    def expand(self, confirmed_orders, locs, state_proto, power_name, phase_history_proto, possible_orders_proto,
               **kwargs):
        """ Computes the conditional probability of possible orders for each loc given the confirmed orders. """
        raise RuntimeError('Not available for this benchmark.')

    @gen.coroutine
    def get_state_value(self, state_proto, power_name, phase_history_proto, possible_orders_proto=None, **kwargs):
        """ Computes the value of the current state for a given power """
        raise RuntimeError('Not available for this benchmark.')
