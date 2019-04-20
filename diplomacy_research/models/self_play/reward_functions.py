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
""" Reward function.
    - Collection of reward functions for the gym environment.
"""
from abc import ABCMeta, abstractmethod
import logging
from diplomacy import Map
from diplomacy_research.models.gym.environment import DoneReason
from diplomacy_research.models.state_space import ALL_POWERS

# Constants
LOGGER = logging.getLogger(__name__)

class AbstractRewardFunction(metaclass=ABCMeta):
    """ Abstract class representing a reward function """

    @property
    @abstractmethod
    def name(self):
        """ Returns a unique name for the reward function """
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        raise NotImplementedError()

    def get_episode_rewards(self, saved_game_proto, power_name):
        """ Compute the list of all rewards for a saved game.
            :param saved_game_proto: A `.proto.SavedGame` representation
            :param power_name: The name of the power for which we want to compute rewards
            :return: A list of rewards (one for each transition in the saved game)
        """
        # Restoring cached rewards
        if saved_game_proto.reward_fn == self.name and saved_game_proto.rewards[power_name].value:
            return list(saved_game_proto.rewards[power_name].value)

        # Otherwise, computing them
        episode_rewards = []
        done_reason = DoneReason(saved_game_proto.done_reason) if saved_game_proto.done_reason != '' else None

        # Making sure we have at least 2 phases (1 transition)
        nb_phases = len(saved_game_proto.phases)
        if nb_phases < 2:
            return episode_rewards

        # Computing the reward of each phase (transition)
        for phase_ix in range(nb_phases - 1):
            current_state = saved_game_proto.phases[phase_ix].state
            next_state = saved_game_proto.phases[phase_ix + 1].state
            is_terminal = phase_ix == nb_phases - 2
            if is_terminal:
                episode_rewards += [self.get_reward(current_state,
                                                    next_state,
                                                    power_name,
                                                    is_terminal_state=True,
                                                    done_reason=done_reason)]
            else:
                episode_rewards += [self.get_reward(current_state,
                                                    next_state,
                                                    power_name,
                                                    is_terminal_state=False,
                                                    done_reason=None)]
        return episode_rewards

class NbCentersReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.

        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the number of supply centers in control by the player at the end of the game.
        The reward is only given at a terminal state.
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'nb_centers_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.
        return len(state_proto.centers[power_name].value)

class NormNbCentersReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.

        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the number of supply centers in control by the player at the end of the game, divided
        by the number of supply centers required to win (to normalize the reward between 0. and 1.).
        The reward is only given at a terminal state.
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'norm_nb_centers_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        return min(1., max(0., len(state_proto.centers[power_name].value) / nb_centers_req_for_win))

class IntNormNbCentersReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.

        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the gain/loss of supply centers between the previous and current phase divided by
        the number of supply centers required to win (to normalize the reward between 0. and 1.).
        The reward is given at every phase (i.e. it is an intermediary reward).
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'int_norm_nb_centers_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers or power_name not in prev_state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        current_centers = set(state_proto.centers[power_name].value)
        prev_centers = set(prev_state_proto.centers[power_name].value)
        sc_diff = len(current_centers) - len(prev_centers)

        if done_reason == DoneReason.THRASHED and current_centers:
            return -1.

        return sc_diff / nb_centers_req_for_win

class CustomIntNbCentersReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.

        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the gain/loss of supply centers between the previous and current phase (+1 / -1)
        Gaining or losing a home multiplies by 2x the gains/losses.
        The reward is given at every phase (i.e. it is an intermediary reward).
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'custom_int_nb_centers_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers or power_name not in prev_state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.

        homes = map_object.homes[power_name]
        current_centers = set(state_proto.centers[power_name].value)
        prev_centers = set(prev_state_proto.centers[power_name].value)
        gained_centers = current_centers - prev_centers
        lost_centers = prev_centers - current_centers

        if done_reason == DoneReason.THRASHED and current_centers:
            return -1. * nb_centers_req_for_win

        # Computing reward
        reward = 0.
        for center in gained_centers:
            reward += 2. if center in homes else 1.
        for center in lost_centers:
            reward -= 2. if center in homes else 1.
        return reward

class CustomIntUnitReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.

        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the gain/loss of supply centers between the previous and current phase (+1 / -1)
        The reward is given as soon as a SC is touched (rather than just during the Adjustment phase)
        Homes are also +1/-1.

        The reward is given at every phase (i.e. it is an intermediary reward).
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'custom_int_unit_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers or power_name not in prev_state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        current_centers = set(state_proto.centers[power_name].value)
        prev_centers = set(prev_state_proto.centers[power_name].value)
        all_scs = map_object.scs

        if done_reason == DoneReason.THRASHED and current_centers:
            return -1. * nb_centers_req_for_win

        # Adjusting supply centers for the current phase
        # Dislodged units don't count for adjustment
        for unit_power in state_proto.units:
            if unit_power == power_name:
                for unit in state_proto.units[unit_power].value:
                    if '*' in unit:
                        continue
                    unit_loc = unit[2:5]
                    if unit_loc in all_scs and unit_loc not in current_centers:
                        current_centers.add(unit_loc)
            else:
                for unit in state_proto.units[unit_power].value:
                    if '*' in unit:
                        continue
                    unit_loc = unit[2:5]
                    if unit_loc in all_scs and unit_loc in current_centers:
                        current_centers.remove(unit_loc)

        # Adjusting supply centers for the previous phase
        # Dislodged units don't count for adjustment
        for unit_power in prev_state_proto.units:
            if unit_power == power_name:
                for unit in prev_state_proto.units[unit_power].value:
                    if '*' in unit:
                        continue
                    unit_loc = unit[2:5]
                    if unit_loc in all_scs and unit_loc not in prev_centers:
                        prev_centers.add(unit_loc)
            else:
                for unit in prev_state_proto.units[unit_power].value:
                    if '*' in unit:
                        continue
                    unit_loc = unit[2:5]
                    if unit_loc in all_scs and unit_loc in prev_centers:
                        prev_centers.remove(unit_loc)

        # Computing difference
        gained_centers = current_centers - prev_centers
        lost_centers = prev_centers - current_centers

        # Computing reward
        return float(len(gained_centers) - len(lost_centers))

class PlusOneMinusOneReward(AbstractRewardFunction):
    """ Reward function: Standard win/lose.

        This reward function rewards winning/drawing (1), and gives -1 for losing.
        All non-terminal states receive 0.
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'plus_one_minus_one_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return -1.
        return -1. if not state_proto.centers[power_name].value else 1.

class DrawSizeReward(AbstractRewardFunction):
    """ Draw Size scoring system. Divides pot size by number of survivors.

        This reward function gives a return of (pot_size / nb_survivors) at the terminal state (0 if eliminated)
        All non-terminal states receive 0.
    """
    def __init__(self, pot_size=34):
        """ Constructor
            :param pot_size: The number of points to split across survivors (default to 34, which is the nb of centers)
        """
        assert pot_size > 0, 'The size of the pot must be positive.'
        self.pot_size = pot_size

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'draw_size_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        victors = [power for power in state_proto.centers
                   if len(state_proto.centers[power].value) >= nb_centers_req_for_win]
        survivors = [power for power in state_proto.centers if state_proto.centers[power].value]

        # if there is a victor, winner takes all
        # else survivors get points uniformly
        if victors:
            split_factor = 1. if power_name in victors else 0.
        else:
            split_factor = 1. / len(survivors) if power_name in survivors else 0.
        return self.pot_size * split_factor

class ProportionalReward(AbstractRewardFunction):
    """ Proportional scoring system.
        For win - The winner takes all, the other parties get nothing
        For draw - The pot size is divided by the number of nb of sc^i / sum of nb of sc^i.
        All non-terminal states receive 0.
    """
    def __init__(self, pot_size=34, exponent=1):
        """ Constructor
            :param pot_size: The number of points to split across survivors (default to 34, which is the nb of centers)
            :param exponent: The exponent to use in the draw calculation.
        """
        assert pot_size > 0, 'The size of the pot must be positive.'
        self.pot_size = pot_size
        self.i = exponent

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'proportional_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        victors = [power for power in state_proto.centers
                   if len(state_proto.centers[power].value) >= nb_centers_req_for_win]

        # if there is a victor, winner takes all
        # else survivors get points according to nb_sc^i / sum of nb_sc^i
        if victors:
            split_factor = 1. if power_name in victors else 0.
        else:
            denom = {power: len(state_proto.centers[power].value) ** self.i for power in state_proto.centers}
            split_factor = denom[power_name] / float(sum(denom.values()))
        return self.pot_size * split_factor

class SumOfSquares(ProportionalReward):
    """ Sum of Squares scoring system.
        For win - The winner takes all, the other parties get nothing
        For draw - The pot size is divided by the number of nb of sc^2 / sum of nb of sc^2.
        All non-terminal states receive 0.
    """
    def __init__(self, pot_size=34):
        """ Constructor
            :param pot_size: The number of points to split across survivors (default to 34, which is the nb of centers)
        """
        super(SumOfSquares, self).__init__(pot_size=pot_size, exponent=2)

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'sum_of_squares_reward'

class SurvivorWinReward(AbstractRewardFunction):
    """ If win:
            nb excess SC = nb of centers of winner - nb of centers required to win (e.g. 20 - 18)
            nb controlled SC = nb of centers owned by all powers - nb excess SC

            Reward for winner = nb of centers required to win / nb controlled SC * pot size
            Reward for survivors = nb of centers / nb controlled SC * pot size

            e.g. 20, 6, 4, 2, 1, 0, 0
            nb excess SC = 20 - 18 = 2
            nb controlled SC = 20 + 6 + 4 + 2 + 1 - 2 = 31
            Winner: 18/31 * pot
            Others: 6/31 * pot, 4/31 * pot, 2/31 * pot, 1/31 * pot, 0, 0

        If draw:
            Pot is split equally among survivors (independently of nb of SC)
    """
    def __init__(self, pot_size=34):
        """ Constructor
            :param pot_size: The number of points to split across survivors (default to 34, which is the nb of centers)
        """
        assert pot_size > 0, 'The size of the pot must be positive.'
        self.pot_size = pot_size

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'survivor_win_reward'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state_proto.centers:
            if power_name not in ALL_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.

        map_object = Map(state_proto.map)
        nb_centers_req_for_win = len(map_object.scs) // 2 + 1.
        victors = [power for power in state_proto.centers
                   if len(state_proto.centers[power].value) >= nb_centers_req_for_win]

        if victors:
            nb_scs = {power: len(state_proto.centers[power].value) for power in state_proto.centers}
            nb_excess_sc = nb_scs[victors[0]] - nb_centers_req_for_win
            nb_controlled_sc = sum(nb_scs.values()) - nb_excess_sc
            if power_name in victors:
                split_factor = nb_centers_req_for_win / nb_controlled_sc
            else:
                split_factor = nb_scs[power_name] / nb_controlled_sc
        else:
            survivors = [power for power in state_proto.centers if state_proto.centers[power].value]
            split_factor = 1. / len(survivors) if power_name in survivors else 0.
        return self.pot_size * split_factor

class MixProportionalCustomIntUnitReward(AbstractRewardFunction):
    """ 50% of ProportionalReward
        50% of CustomIntUnitReward
    """
    def __init__(self):
        """ Constructor """
        self.proportional = ProportionalReward()
        self.custom_int_unit = CustomIntUnitReward()

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'mix_proportional_custom_int_unit'

    def get_reward(self, prev_state_proto, state_proto, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param prev_state_proto: The `.proto.State` representation of the last state of the game (before .process)
            :param state_proto: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        return 0.5 * self.proportional.get_reward(prev_state_proto,
                                                  state_proto,
                                                  power_name,
                                                  is_terminal_state,
                                                  done_reason) \
               + 0.5 * self.custom_int_unit.get_reward(prev_state_proto,
                                                       state_proto,
                                                       power_name,
                                                       is_terminal_state,
                                                       done_reason)

# --- Defaults ---
DEFAULT_PENALTY = 0.
DEFAULT_GAMMA = 0.99

class DefaultRewardFunction(MixProportionalCustomIntUnitReward):
    """ Default reward function class """
