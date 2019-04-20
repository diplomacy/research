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
""" OpenAI Gym Environment for Diplomacy
    - Implements a gym environment to train the RL model
"""
from enum import Enum
import logging
import gym
from gym.utils import seeding
from diplomacy import Game
from diplomacy_research.models.state_space import get_game_id, get_player_seed, get_map_powers

# Constants
LOGGER = logging.getLogger(__name__)

class DoneReason(Enum):
    """ Enumeration of reasons why the environment has terminated """
    NOT_DONE = 'not_done'                                       # Partial game - Not yet completed
    GAME_ENGINE = 'game_engine'                                 # Triggered by the game engine
    AUTO_DRAW = 'auto_draw'                                     # Game was automatically drawn accord. to some prob.
    PHASE_LIMIT = 'phase_limit'                                 # The maximum number of phases was reached
    THRASHED = 'thrashed'                                       # A loop was detected.

class DiplomacyEnv(gym.Env):
    """ Gym environment wrapper for the Diplomacy board game. """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """ Constructor """
        self.game = None
        self.curr_seed = 0
        self._last_known_phase = 'S1901M'

    @property
    def current_year(self):
        """ Returns the current year of the game in normalized format
            e.g. S1901M = year 1
                 F1903M = year 3
                 COMPLETED = year of last phase
        """
        current_phase = self.game.get_current_phase()
        if current_phase == 'COMPLETED':
            current_phase = self._last_known_phase
        return int(current_phase[1:5]) - self.game.map.first_year + 1

    @property
    def game_id(self):
        """ Returns the current game game_id """
        if self.game:
            return self.game.game_id
        return ''

    @property
    def players(self):
        """ Returns a list of players instances playing the game """
        raise NotImplementedError()

    @property
    def is_done(self):
        """ Determines if the game is done """
        return self.game.is_game_done

    @property
    def done_reason(self):
        """ Returns the reason why the game was terminated """
        if self.is_done:
            return DoneReason.GAME_ENGINE
        return None

    def process(self):
        """ Requests that the game processes the current orders """
        self.game.process()
        current_phase = self.game.get_current_phase()
        if current_phase != 'COMPLETED':
            self._last_known_phase = current_phase

    def seed(self, seed=None):
        """ Sets a random seed """
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [self.curr_seed]

    def step(self, action):
        """ Have one agent interact with the environment once.
            :param action: Tuple containing the POWER name and its corresponding list of orders.
                        (e.g. ('FRANCE', ['A PAR H', 'A MAR - BUR', ...])
            :return: Nothing
        """
        power_name, orders = action
        if self.game.get_current_phase()[-1] == 'R':
            orders = [order.replace(' - ', ' R ') for order in orders]
        orders = [order for order in orders if order != 'WAIVE']
        self.game.set_orders(power_name, orders, expand=False)

    def reset(self):
        """ Resets the game to its starting configuration
            :return: ** None. This is a deviation from the standard Gym API. **
        """
        self.game = Game(game_id=get_game_id())
        self._last_known_phase = self.game.get_current_phase()

    def get_all_powers_name(self):
        """ Returns the power for all players """
        map_object = (self.game or Game()).map
        return get_map_powers(map_object)

    def get_player_seeds(self):
        """ Returns a dictionary of power_name: seed to use for all powers """
        map_object = (self.game or Game()).map
        if not self.game:
            return {power_name: 0 for power_name in get_map_powers(map_object)}
        return {power_name: get_player_seed(self.game.game_id, power_name)
                for power_name in get_map_powers(map_object)}

    @staticmethod
    def get_saved_game():
        """ Returns the last saved game """
