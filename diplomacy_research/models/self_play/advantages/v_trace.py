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
""" V-Trace
    - Based on 1802.01561 (IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures)
"""
import logging
from diplomacy_research.models.self_play.advantages.base_advantage import BaseAdvantage
from diplomacy_research.models.self_play.transition import Transition, TransitionDetails
from diplomacy_research.models.state_space import ALL_POWERS, NB_PREV_ORDERS_HISTORY

# Constants
LOGGER = logging.getLogger(__name__)

class VTrace(BaseAdvantage):
    """ V-Trace - Calculates the V-Trace targets over a full trajectory """

    def __init__(self, *, lambda_, gamma, c_bar=1., p_bar=1., penalty_per_phase=0.):
        """ Constructor
            :param lambda_: The c_ lambda
            :param gamma: The discount factor to use
            :param c_bar: The parameter c_ has described in the paper.
            :param p_bar: The parameter p_ has described in the paper.
            :param penalty_per_phase: The penalty to add to each transition.
        """
        BaseAdvantage.__init__(self, gamma=gamma, penalty_per_phase=penalty_per_phase)
        self.lambda_ = lambda_
        self.c_bar = c_bar
        self.p_bar = max(c_bar, p_bar)                   # To ensure that p_bar >= c_bar

    def _get_vtrace_targets(self, rewards, state_values, last_state_value, rhos=None):
        """ Computes the returns and value targets for all transitions
            :param rewards: A list of rewards received (after each env.step())
            :param state_values: A list of values for each state (before env.step())
            :param last_state_value: The value of the last state
            :param rhos: Optional. List of importance sampling weights.
            :return: A list of returns (same length as rewards), and a list of value targets
        """
        assert len(rewards) == len(state_values), 'Expected rewards to be the same length as state_values'
        assert rhos is None or len(rhos) == len(rewards), 'Expected rhos to be the same length as rewards'

        # Note - Because this is computed on the actors - using log_rhos of 0. (importance sampling weights of 1.)
        # The actual adjustment will be done on the learner.
        if rhos is None:
            rhos = [1.] * len(rewards)
        discounts = [self.lambda_] * len(rewards)
        clip_rho_threshold = self.p_bar
        clip_ph_rho_threshold = self.c_bar
        adj_rewards = [reward - self.penalty_per_phase for reward in rewards]

        # Computing V-Trace returns
        clipped_rhos = [min(clip_rho_threshold, rho) for rho in rhos]
        clipped_pg_rhos = [min(clip_ph_rho_threshold, rho) for rho in rhos]
        c_s = [min(1., rho) for rho in rhos]
        values_t_plus_1 = state_values[1:] + [last_state_value]
        deltas = [clipped_rho * (reward + discount * value_t_plus_1 - value)
                  for (clipped_rho, reward, discount, value_t_plus_1, value)
                  in zip(clipped_rhos, adj_rewards, discounts, values_t_plus_1, state_values)]

        # Computation starts from the back
        sequences = (reversed(discounts), reversed(c_s), reversed(deltas))

        # Computing returns
        acc = 0.
        v_s_minus_v_xs = []
        for discount_t, c_t, delta_t in zip(*sequences):
            acc = delta_t + discount_t * c_t * acc
            v_s_minus_v_xs += [acc]
        v_s_minus_v_xs = list(reversed(v_s_minus_v_xs))

        # Add V(x_s) to get v_s.
        v_s = [vs_minus_v_xs_t + value_t for (vs_minus_v_xs_t, value_t) in zip(v_s_minus_v_xs, state_values)]
        v_s_t_plus_1 = v_s[1:] + [last_state_value]

        # Returns for policy gradients
        vtrace_returns = [clipped_pg_rho * (reward + discount * v_s_t_plus_1_t)
                          for (clipped_pg_rho, reward, discount, v_s_t_plus_1_t)
                          in zip(clipped_pg_rhos, adj_rewards, discounts, v_s_t_plus_1)]
        return vtrace_returns, v_s

    def get_returns(self, rewards, state_values, last_state_value):
        """ Computes the returns for all transitions
            :param rewards: A list of rewards received (after each env.step())
            :param state_values: A list of values for each state (before env.step())
            :param last_state_value: The value of the last state
            :return: A list of returns (same length as rewards)
        """
        vtrace_returns, _ = self._get_vtrace_targets(rewards, state_values, last_state_value)
        return vtrace_returns

    def get_transition_details(self, saved_game_proto, power_name, **kwargs):
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

        # Recomputing the returns and value targets
        rewards = saved_game_proto.rewards[power_name].value
        state_values = [phase.state_value[power_name] for phase in saved_game_proto.phases[:-1]]
        last_state_value = saved_game_proto.phases[-1].state_value[power_name]
        vtrace_returns, value_targets = self._get_vtrace_targets(rewards, state_values, last_state_value)

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
            vtrace_return = vtrace_returns[phase_ix]
            value_target = value_targets[phase_ix]

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
                                   reward_target=vtrace_return,
                                   value_target=value_target,
                                   log_p_t=None,
                                   log_probs=policy_details[power_name].log_probs)]

        # Returning transition details
        return transition_details
