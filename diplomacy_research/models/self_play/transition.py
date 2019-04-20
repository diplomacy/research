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
""" Transition
    - This module is responsible for defining the transition classes
"""
import collections

class Transition(
        collections.namedtuple('Transition', ('phase_ix',               # The phase ix
                                              'state',                  # The proto representation of the state
                                              'phase',                  # The proto representation of the phase
                                              'policy',                 # The details of the behaviour policy
                                              'rewards',                # The rewards received by each power
                                              'orders',                 # {power => orders during the phase}
                                              'phase_history',          # The phase_history_proto
                                              'possible_orders'))):     # The possible_orders_proto
    """ Represents a transition (state, action, reward) for all powers """

class TransitionDetails(
        collections.namedtuple('TransitionDetails', ('transition',      # The transition named tuple
                                                     'power_name',      # The power that issued orders
                                                     'draw_action',     # Whether the power wanted a draw or not
                                                     'reward_target',   # The reward target for the policy update
                                                     'value_target',    # The value target for the baseline update
                                                     'log_p_t',         # None or the log of the import sampling ratio
                                                     'log_probs'))):    # Log probs of each token under the model
    """ Contains the details of a transition (as computed by the advantage function) for a specific power """

class ReplaySample(
        collections.namedtuple('ReplaySample', ('saved_game_proto',     # A SavedGameProto instance
                                                'power_phases_ix'))):   # {power_name: [phases_ix]}
    """ Contains a saved game proto with a list of prioritized transition ix """

class ReplayPriority(
        collections.namedtuple('ReplayPriority', ('game_id',            # The game id containing a transition priority
                                                  'power_name',         # The power issuing orders
                                                  'phase_ix',           # The phase ix
                                                  'priority'))):        # The priority assigned to the transition/phase
    """ Contains the priority for a given transition ix """
