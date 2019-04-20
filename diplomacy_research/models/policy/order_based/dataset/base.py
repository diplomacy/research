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
""" Order Based Base Dataset Builder
    - Base class responsible for generating the protocol buffers to be used by the model
"""
import logging
import numpy as np
from diplomacy import Map
from diplomacy_research.models.datasets.base_builder import FixedProtoField, VarProtoField
from diplomacy_research.models.policy.base_policy_builder import BasePolicyBuilder
from diplomacy_research.models.self_play.reward_functions import DefaultRewardFunction, DEFAULT_GAMMA
from diplomacy_research.models.state_space import get_order_tokens, get_order_based_mask, \
    get_possible_orders_for_powers, get_issued_orders_for_powers, proto_to_board_state, GO_ID, NB_NODES, \
    NB_SUPPLY_CENTERS, POWER_VOCABULARY_KEY_TO_IX, order_to_ix, MAX_CANDIDATES, NB_FEATURES, NB_ORDERS_FEATURES, \
    NB_PREV_ORDERS, NB_PREV_ORDERS_HISTORY, get_board_alignments, get_orderable_locs_for_powers, get_current_season, \
    proto_to_prev_orders_state

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
            'draw_target': FixedProtoField([], np.float32),
            'value_target': FixedProtoField([], np.float32)
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


# ---------- Multiprocessing methods to generate proto buffer ----------------
def get_policy_data(saved_game_proto, power_names, top_victors):
    """ Computes the proto to save in tf.train.Example as a training example for the policy network
        :param saved_game_proto: A `.proto.game.SavedGame` object from the dataset.
        :param power_names: The list of powers for which we want the policy data
        :param top_victors: The list of powers that ended with more than 25% of the supply centers
        :return: A dictionary with key: the phase_ix
                              with value: A dict with the power_name as key and a dict with the example fields as value
    """
    nb_phases = len(saved_game_proto.phases)
    policy_data = {phase_ix: {} for phase_ix in range(nb_phases - 1)}
    game_id = saved_game_proto.id
    map_object = Map(saved_game_proto.map)

    # Determining if we have a draw
    nb_sc_to_win = len(map_object.scs) // 2 + 1
    has_solo_winner = max([len(saved_game_proto.phases[-1].state.centers[power_name].value)
                           for power_name in saved_game_proto.phases[-1].state.centers]) >= nb_sc_to_win
    survivors = [power_name for power_name in saved_game_proto.phases[-1].state.centers
                 if saved_game_proto.phases[-1].state.centers[power_name].value]
    has_draw = not has_solo_winner and len(survivors) >= 2

    # Processing all phases (except the last one)
    current_year = 0
    for phase_ix in range(nb_phases - 1):

        # Building a list of orders of previous phases
        previous_orders_states = [np.zeros((NB_NODES, NB_ORDERS_FEATURES), dtype=np.uint8)] * NB_PREV_ORDERS
        for phase_proto in saved_game_proto.phases[max(0, phase_ix - NB_PREV_ORDERS_HISTORY):phase_ix]:
            if phase_proto.name[-1] == 'M':
                previous_orders_states += [proto_to_prev_orders_state(phase_proto, map_object)]
        previous_orders_states = previous_orders_states[-NB_PREV_ORDERS:]
        prev_orders_state = np.array(previous_orders_states)

        # Parsing each requested power in the specified phase
        phase_proto = saved_game_proto.phases[phase_ix]
        phase_name = phase_proto.name
        state_proto = phase_proto.state
        phase_board_state = proto_to_board_state(state_proto, map_object)

        # Increasing year for every spring or when the game is completed
        if phase_proto.name == 'COMPLETED' or (phase_proto.name[0] == 'S' and phase_proto.name[-1] == 'M'):
            current_year += 1

        for power_name in power_names:
            phase_issued_orders = get_issued_orders_for_powers(phase_proto, [power_name])
            phase_possible_orders = get_possible_orders_for_powers(phase_proto, [power_name])
            phase_draw_target = 1. if has_draw and phase_ix == (nb_phases - 2) and power_name in survivors else 0.

            # Data to use when not learning a policy
            blank_policy_data = {'board_state': phase_board_state,
                                 'prev_orders_state': prev_orders_state,
                                 'draw_target': phase_draw_target}

            # Power is not a top victor - We don't want to learn a policy from him
            if power_name not in top_victors:
                policy_data[phase_ix][power_name] = blank_policy_data
                continue

            # Finding the orderable locs
            orderable_locations = list(phase_issued_orders[power_name].keys())

            # Skipping power for this phase if we are only issuing Hold
            for order_loc, order in phase_issued_orders[power_name].items():
                order_tokens = get_order_tokens(order)
                if len(order_tokens) >= 2 and order_tokens[1] != 'H':
                    break
            else:
                policy_data[phase_ix][power_name] = blank_policy_data
                continue

            # Removing orderable locs where orders are not possible (i.e. NO_CHECK games)
            for order_loc, order in phase_issued_orders[power_name].items():
                if order not in phase_possible_orders[order_loc] and order_loc in orderable_locations:
                    if 'NO_CHECK' not in saved_game_proto.rules:
                        LOGGER.warning('%s not in all possible orders. Phase %s - Game %s.', order, phase_name, game_id)
                    orderable_locations.remove(order_loc)

                # Remove orderable locs where the order is either invalid or not frequent
                if order_to_ix(order) is None and order_loc in orderable_locations:
                    orderable_locations.remove(order_loc)

            # Determining if we are in an adjustment phase
            in_adjustment_phase = state_proto.name[-1] == 'A'
            nb_builds = state_proto.builds[power_name].count
            nb_homes = len(state_proto.builds[power_name].homes)

            # WxxxA - We can build units
            # WxxxA - We can disband units
            # Other phase
            if in_adjustment_phase and nb_builds >= 0:
                decoder_length = min(nb_builds, nb_homes)
            elif in_adjustment_phase and nb_builds < 0:
                decoder_length = abs(nb_builds)
            else:
                decoder_length = len(orderable_locations)

            # Not all units were disbanded - Skipping this power as we can't learn the orders properly
            if in_adjustment_phase and nb_builds < 0 and len(orderable_locations) < abs(nb_builds):
                policy_data[phase_ix][power_name] = blank_policy_data
                continue

            # Not enough orderable locations for this power, skipping
            if not orderable_locations or not decoder_length:
                policy_data[phase_ix][power_name] = blank_policy_data
                continue

            # decoder_inputs [GO, order1, order2, order3]
            decoder_inputs = [GO_ID]
            decoder_inputs += [order_to_ix(phase_issued_orders[power_name][loc]) for loc in orderable_locations]
            if in_adjustment_phase and nb_builds > 0:
                decoder_inputs += [order_to_ix('WAIVE')] * (min(nb_builds, nb_homes) - len(orderable_locations))
            decoder_length = min(decoder_length, NB_SUPPLY_CENTERS)

            # Adjustment Phase - Use all possible orders for each location.
            if in_adjustment_phase:
                build_disband_locs = list(get_possible_orders_for_powers(phase_proto, [power_name]).keys())
                phase_board_alignments = get_board_alignments(build_disband_locs,
                                                              in_adjustment_phase=in_adjustment_phase,
                                                              tokens_per_loc=1,
                                                              decoder_length=decoder_length)

                # Building a list of all orders for all locations
                adj_orders = []
                for loc in build_disband_locs:
                    adj_orders += phase_possible_orders[loc]

                # Not learning builds for BUILD_ANY
                if nb_builds > 0 and 'BUILD_ANY' in state_proto.rules:
                    adj_orders = []

                # No orders found - Skipping
                if not adj_orders:
                    policy_data[phase_ix][power_name] = blank_policy_data
                    continue

                # Computing the candidates
                candidates = [get_order_based_mask(adj_orders)] * decoder_length

            # Regular phase - Compute candidates for each location
            else:
                phase_board_alignments = get_board_alignments(orderable_locations,
                                                              in_adjustment_phase=in_adjustment_phase,
                                                              tokens_per_loc=1,
                                                              decoder_length=decoder_length)
                candidates = []
                for loc in orderable_locations:
                    candidates += [get_order_based_mask(phase_possible_orders[loc])]

            # Saving results
            # No need to return temperature, current_power, current_season
            policy_data[phase_ix][power_name] = {'board_state': phase_board_state,
                                                 'board_alignments': phase_board_alignments,
                                                 'prev_orders_state': prev_orders_state,
                                                 'decoder_inputs': decoder_inputs,
                                                 'decoder_lengths': decoder_length,
                                                 'candidates': candidates,
                                                 'draw_target': phase_draw_target}
    # Returning
    return policy_data

def get_value_data(saved_game_proto, power_names):
    """ Computes the proto to save in tf.train.Example as a training example for the value network
        :param saved_game_proto: A `.proto.game.SavedGame` object from the dataset.
        :param power_names: The list of powers for which we want the policy data
        :return: A dictionary with key: the phase_ix
                              with value: A dict with the power_name as key and a dict with the example fields as value
    """
    nb_phases = len(saved_game_proto.phases)
    value_data = {phase_ix: {} for phase_ix in range(nb_phases - 1)}

    # Computing the value of each phase
    for power_name in power_names:
        value_targets = []
        current_value = 0.
        rewards = DefaultRewardFunction().get_episode_rewards(saved_game_proto, power_name)
        for reward in reversed(rewards):
            current_value = reward + DEFAULT_GAMMA * current_value
            value_targets += [current_value]
        value_targets += [0]

        # Computing the value data
        for phase_ix in range(nb_phases - 1):
            value_data[phase_ix][power_name] = {'value_target': value_targets[phase_ix]}

    # Returning the value of the specified phase for each power
    return value_data
