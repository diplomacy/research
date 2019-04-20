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
""" N-Phase Advantage Calculator
    - Compute the TD error for a certain number of phases in a saved game / trajectory
"""
import logging
from diplomacy_research.models.self_play.advantages.base_advantage import BaseAdvantage

# Constants
LOGGER = logging.getLogger(__name__)

class NStep(BaseAdvantage):
    """ N-Step - Calculates the N-Step returns over a full trajectory """

    def __init__(self, *, nb_steps, gamma, penalty_per_phase=0.):
        """ Constructor
            :param nb_steps: The number of steps (phases) for N-Step.
            :param gamma: The discount factor to use
            :param penalty_per_phase: The penalty to add to each transition.
        """
        BaseAdvantage.__init__(self, gamma=gamma, penalty_per_phase=penalty_per_phase)
        self.nb_steps = nb_steps

    def get_returns(self, rewards, state_values, last_state_value):
        """ Computes the returns for all transitions
            :param rewards: A list of rewards received (after each env.step())
            :param state_values: A list of values for each state (before env.step())
            :param last_state_value: The value of the last state
            :return: A list of returns (same length as rewards)
        """
        assert len(rewards) == len(state_values), 'Expected rewards to be the same length as state_values'
        returns = []

        # List of final values for bootstrap
        bootstrap = state_values + [last_state_value]
        adj_rewards = [reward - self.penalty_per_phase for reward in rewards]

        # Computing the N-Step returns
        nb_transitions = len(rewards)
        for transition_ix in range(nb_transitions):
            nb_steps = min(self.nb_steps, nb_transitions - transition_ix)
            current_return = bootstrap[transition_ix + nb_steps]
            for step_ix in reversed(range(nb_steps)):
                current_return = adj_rewards[transition_ix + step_ix] + self.gamma * current_return
            returns += [current_return]
        return returns
