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
""" Generalized Advantage Estimation
    - Based on 1506.02438
"""
import logging
from diplomacy_research.models.self_play.advantages.base_advantage import BaseAdvantage

# Constants
LOGGER = logging.getLogger(__name__)

class GAE(BaseAdvantage):
    """ Generalized Advantage - Calculates the GAE targets over a full trajectory """

    def __init__(self, *, lambda_, gamma, penalty_per_phase=0.):
        """ Constructor
            :param lambda_: The GAE lambda
            :param gamma: The discount factor to use
            :param penalty_per_phase: The penalty to add to each transition.
        """
        BaseAdvantage.__init__(self, gamma=gamma, penalty_per_phase=penalty_per_phase)
        self.lambda_ = lambda_

    def get_returns(self, rewards, state_values, last_state_value):
        """ Computes the returns for all transitions
            :param rewards: A list of rewards received (after each env.step())
            :param state_values: A list of values for each state (before env.step())
            :param last_state_value: The value of the last state
            :return: A list of returns (same length as rewards)
        """
        assert len(rewards) == len(state_values), 'Expected rewards to be the same length as state_values'
        returns = []

        # Computing the GAE returns
        last_gae_lam = 0
        nb_phases = len(rewards)
        adj_rewards = [reward - self.penalty_per_phase for reward in rewards]

        for phase_ix in reversed(range(nb_phases)):
            if phase_ix == nb_phases - 1:
                next_non_terminal = 0.
                next_values = last_state_value
            else:
                next_non_terminal = 1.
                next_values = state_values[phase_ix + 1]
            delta = adj_rewards[phase_ix] + self.gamma * next_values * next_non_terminal - state_values[phase_ix]
            advantage = last_gae_lam = delta + self.gamma * self.lambda_ * next_non_terminal * last_gae_lam
            returns += [advantage + state_values[phase_ix]]

        # Returning
        return list(reversed(returns))
