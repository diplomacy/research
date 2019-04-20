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
""" RL (NeurIPS 2019) Dataset Builder
    - Base class responsible for generating the protocol buffers to be used by the model
"""
import logging
import numpy as np
from diplomacy import Map
from diplomacy_research.models.datasets.base_builder import FixedProtoField, VarProtoField
from diplomacy_research.models.policy.base_policy_builder import BasePolicyBuilder
from diplomacy_research.models.state_space import get_order_based_mask, proto_to_board_state, GO_ID, NB_NODES, \
    NB_SUPPLY_CENTERS, POWER_VOCABULARY_KEY_TO_IX, MAX_CANDIDATES, NB_FEATURES, NB_ORDERS_FEATURES, NB_PREV_ORDERS, \
    get_board_alignments, get_orderable_locs_for_powers, get_current_season, proto_to_prev_orders_state

# Constants
LOGGER = logging.getLogger(__name__)

class BaseDatasetBuilder(BasePolicyBuilder):
    """ This object is responsible for maintaining the data and feeding it into the model """

    @staticmethod
    def get_proto_fields():
        """ Returns the proto fields used by this dataset builder """
        # Creating proto fields
        proto_fields = {
            'request_id': FixedProtoField([], None),
            'player_seed': FixedProtoField([], np.int32),
            'board_state': FixedProtoField([NB_NODES, NB_FEATURES], np.uint8),
            'board_alignments': VarProtoField([NB_NODES * NB_SUPPLY_CENTERS], np.uint8),
            'prev_orders_state': FixedProtoField([NB_PREV_ORDERS, NB_NODES, NB_ORDERS_FEATURES], np.uint8),
            'decoder_inputs': VarProtoField([1 + NB_SUPPLY_CENTERS], np.int32),
            'decoder_lengths': FixedProtoField([], np.int32),
            'candidates': VarProtoField([None, MAX_CANDIDATES], np.int32),
            'noise': FixedProtoField([], np.float32),
            'temperature': FixedProtoField([], np.float32),
            'dropout_rate': FixedProtoField([], np.float32),
            'current_power': FixedProtoField([], np.int32),
            'current_season': FixedProtoField([], np.int32),
            'value_targets': FixedProtoField([], np.float32),
            'context': VarProtoField([256 * 2 * 8], np.float32),
            'messages': VarProtoField([1 + 1000], np.int32),
            'message_lengths': FixedProtoField([], np.int32),
            'senders': VarProtoField([1000], np.uint8),
            'recipients': VarProtoField([1000], np.uint8),
            'next_conversant': FixedProtoField([2], np.int32)
        }
        return proto_fields

    @staticmethod
    def get_feedable_item(locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Computes and return a feedable item (to be fed into the feedable queue)
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
            :return: A feedable item, with feature names as key and numpy arrays as values
        """
        # pylint: disable=too-many-branches
        # Converting to state space
        map_object = Map(state_proto.map)
        board_state = proto_to_board_state(state_proto, map_object)

        # Building the decoder length
        # For adjustment phase, we restrict the number of builds/disbands to what is allowed by the game engine
        in_adjustment_phase = state_proto.name[-1] == 'A'
        nb_builds = state_proto.builds[power_name].count
        nb_homes = len(state_proto.builds[power_name].homes)

        # If we are in adjustment phase, making sure the locs are the orderable locs (and not the policy locs)
        if in_adjustment_phase:
            orderable_locs, _ = get_orderable_locs_for_powers(state_proto, [power_name])
            if sorted(locs) != sorted(orderable_locs):
                if locs:
                    LOGGER.warning('Adj. phase requires orderable locs. Got %s. Expected %s.', locs, orderable_locs)
                locs = orderable_locs

        # WxxxA - We can build units
        # WxxxA - We can disband units
        # Other phase
        if in_adjustment_phase and nb_builds >= 0:
            decoder_length = min(nb_builds, nb_homes)
        elif in_adjustment_phase and nb_builds < 0:
            decoder_length = abs(nb_builds)
        else:
            decoder_length = len(locs)

        # Computing the candidates for the policy
        if possible_orders_proto:

            # Adjustment Phase - Use all possible orders for each location.
            if in_adjustment_phase:

                # Building a list of all orders for all locations
                adj_orders = []
                for loc in locs:
                    adj_orders += possible_orders_proto[loc].value

                # Computing the candidates
                candidates = [get_order_based_mask(adj_orders)] * decoder_length

            # Regular phase - Compute candidates for each location
            else:
                candidates = []
                for loc in locs:
                    candidates += [get_order_based_mask(possible_orders_proto[loc].value)]

        # We don't have possible orders, so we cannot compute candidates
        # This might be normal if we are only getting the state value or the next message to send
        else:
            candidates = []
            for _ in range(decoder_length):
                candidates.append([])

        # Prev orders state
        prev_orders_state = []
        for phase_proto in reversed(phase_history_proto):
            if len(prev_orders_state) == NB_PREV_ORDERS:
                break
            if phase_proto.name[-1] == 'M':
                prev_orders_state = [proto_to_prev_orders_state(phase_proto, map_object)] + prev_orders_state
        for _ in range(NB_PREV_ORDERS - len(prev_orders_state)):
            prev_orders_state = [np.zeros((NB_NODES, NB_ORDERS_FEATURES), dtype=np.uint8)] + prev_orders_state
        prev_orders_state = np.array(prev_orders_state)

        # Building (order) decoder inputs [GO_ID]
        decoder_inputs = [GO_ID]

        # kwargs
        player_seed = kwargs.get('player_seed', 0)
        noise = kwargs.get('noise', 0.)
        temperature = kwargs.get('temperature', 0.)
        dropout_rate = kwargs.get('dropout_rate', 0.)

        # Building feedable data
        item = {
            'player_seed': player_seed,
            'board_state': board_state,
            'board_alignments': get_board_alignments(locs,
                                                     in_adjustment_phase=in_adjustment_phase,
                                                     tokens_per_loc=1,
                                                     decoder_length=decoder_length),
            'prev_orders_state': prev_orders_state,
            'decoder_inputs': decoder_inputs,
            'decoder_lengths': decoder_length,
            'candidates': candidates,
            'noise': noise,
            'temperature': temperature,
            'dropout_rate': dropout_rate,
            'current_power': POWER_VOCABULARY_KEY_TO_IX[power_name],
            'current_season': get_current_season(state_proto)
        }

        # Return
        return item

    @property
    def proto_generation_callable(self):
        """ Returns a callable required for proto files generation.
            e.g. return generate_proto(saved_game_bytes, is_validation_set)

            Note: Callable args are - saved_game_bytes: A `.proto.game.SavedGame` object from the dataset
                                    - phase_ix: The index of the phase we want to process
                                    - is_validation_set: Boolean that indicates if we are generating the validation set

            Note: Used bytes_to_proto from diplomacy_research.utils.proto to convert bytes to proto
                  The callable must return a list of tf.train.Example to put in the protocol buffer file
        """
        raise NotImplementedError()
