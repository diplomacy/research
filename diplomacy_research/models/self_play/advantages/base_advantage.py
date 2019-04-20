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
""" Base Advantages
    - Compute the TD error for a certain number of phases in a saved game / trajectory
"""
from abc import ABCMeta, abstractmethod
import logging
from diplomacy_research.models.self_play.transition import Transition, TransitionDetails
from diplomacy_research.models.state_space import NB_PREV_ORDERS_HISTORY, ALL_POWERS

# Constants
LOGGER = logging.getLogger(__name__)

class BaseAdvantage(metaclass=ABCMeta):
    """ Base Advantage - Overrided by the different calculation methods """

    def __init__(self, *, gamma, penalty_per_phase=0.):
        """ Constructor
            :param gamma: The discount factor to use
            :param penalty_per_phase: The penalty to add to each transition.
        """
        self.gamma = gamma
        self.penalty_per_phase = penalty_per_phase

    @abstractmethod
    def get_returns(self, rewards, state_values, last_state_value):
        """ Computes the returns for all transitions
            :param rewards: A list of rewards received (after each env.step())
            :param state_values: A list of values for each state (before env.step())
            :param last_state_value: The value of the last state
            :return: A list of returns (same length as rewards)
        """
        raise NotImplementedError()

    @staticmethod
    def get_transition_details(saved_game_proto, power_name, **kwargs):
        """ Calculates the transition details for the saved game.
            :param saved_game_proto: A `.proto.game.SavedGame` object.
            :param power_name: The name of the power for which we want the target.
            :param kwargs: Additional optional kwargs:
                - player_seed: The seed to apply to the player to compute a deterministic mask.
                - noise: The sigma of the additional noise to apply to the intermediate layers (i.e. sigma * epsilon)
                - temperature: The temperature to apply to the logits. (Default to 0. for deterministic/greedy)
                - dropout_rate: The amount of dropout to apply to the inputs/outputs of the decoder.
            :return: The transition detail for the phase, including:
                        1) a `.models.self_play.transition.Transition` transition.
                        2) the power that issued orders
                        3) the corresponding reward target to be used for the policy gradient update,
                        4) the corresponding value target to be used for the critic update,
                        5) a list of log importance sampling ratio for each output token or None
                        6) the updated (current) log probs for each token in the model
            :type transitions: List[diplomacy_research.models.self_play.transition.Transition]
        """
        del kwargs          # Unused args
        transition_details = []

        # We need at least 2 phases to have 1 transition
        nb_phases = len(saved_game_proto.phases)
        if nb_phases < 2:
            return transition_details

        # Looping over all phases
        current_year = 0
        for phase_ix in range(nb_phases - 1):
            current_state = saved_game_proto.phases[phase_ix].state
            current_phase = saved_game_proto.phases[phase_ix]
            policy_details = saved_game_proto.phases[phase_ix].policy       # Policy details (locs, tokens, log_probs)
            orders = saved_game_proto.phases[phase_ix].orders
            phase_history = [saved_game_proto.phases[phase_order_ix]
                             for phase_order_ix in range(max(0, phase_ix - NB_PREV_ORDERS_HISTORY), phase_ix)]
            possible_orders = saved_game_proto.phases[phase_ix].possible_orders
            rewards = {power_name: saved_game_proto.rewards[power_name].value[phase_ix] for power_name in ALL_POWERS}
            current_return = saved_game_proto.returns[power_name].value[phase_ix]

            # Increasing year for every spring or when the game is completed
            if current_phase.name == 'COMPLETED' or (current_phase.name[0] == 'S' and current_phase.name[-1] == 'M'):
                current_year += 1

            # Building transition
            transition = Transition(phase_ix=phase_ix,
                                    state=current_state,
                                    phase=current_phase,
                                    policy=policy_details,
                                    rewards=rewards,
                                    orders=orders,
                                    phase_history=phase_history,
                                    possible_orders=possible_orders)

            # Building transition details
            transition_details += \
                [TransitionDetails(transition=transition,
                                   power_name=power_name,
                                   draw_action=policy_details[power_name].draw_action,
                                   reward_target=current_return,
                                   value_target=current_return,
                                   log_p_t=None,
                                   log_probs=policy_details[power_name].log_probs)]

        # Returning transition details
        return transition_details
