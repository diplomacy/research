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
""" Wrappers
    - Contains wrappers to provide custom functionality to the diplomacy gym environment
"""
import logging
import os
from random import shuffle, uniform
import time
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from gym import Wrapper
import ujson as json
from diplomacy_research.models.gym.environment import DoneReason
from diplomacy_research.utils.proto import proto_to_dict

# Constants
LOGGER = logging.getLogger(__name__)

class DiplomacyWrapper(Wrapper):
    """ Generic Diplomacy Wrapper """
    @property
    def game(self):
        """ The game object """
        return self.env.game

    @property
    def current_year(self):
        """ Returns the current year of the game in normalized format
            e.g. S1901M = year 1
                 F1903M = year 3
                 COMPLETED = year of last phase
        """
        return self.env.current_year

    @property
    def game_id(self):
        """ Returns the current game game_id """
        if self.game:
            return self.game.game_id
        return ''

    @property
    def players(self):
        """ Returns a list of players instances playing the game """
        return self.env.players

    @property
    def is_done(self):
        """ Determines if the game is done """
        return self.env.is_done

    @property
    def done_reason(self):
        """ Returns the reason why the game was terminated """
        return self.env.done_reason

    def process(self, **kwargs):
        """ Requests that the game processes the current orders """
        return self.env.process(**kwargs)

    def step(self, action):
        """ Have one agent interact with the environment once.
            :param action: The action to perform
        """
        # _step was deprecated in gym 0.9.6, method_hidden triggered by gym.core L225
        # pylint: disable=method-hidden
        return self.env.step(action)

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and shuffles the player ordering """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        return self.env.reset(**kwargs)

    def get_all_powers_name(self):
        """ Returns the power for all players """
        if callable(getattr(self.env, 'get_all_powers_name')):
            return self.env.get_all_powers_name()
        raise NotImplementedError()

    def get_player_seeds(self):
        """ Returns a dictionary of power_name: seed to use for all powers """
        if callable(getattr(self.env, 'get_player_seeds')):
            return self.env.get_player_seeds()
        raise NotImplementedError()

    def set_draw_prob(self, power_name, draw_prob):
        """ Sets the probability of accepting a draw for a given power """
        if callable(getattr(self.env, 'set_draw_prob')):
            return self.env.set_draw_prob(power_name, draw_prob)
        raise NotImplementedError()

    def get_draw_actions(self):
        """ Returns a dictionary with power_name as key, and draw_action (True/False) as value """
        if callable(getattr(self.env, 'get_draw_actions')):
            return self.env.get_draw_actions()
        raise NotImplementedError()

    def get_saved_game(self):
        """ Returns the last saved game """
        if callable(getattr(self.env, 'get_saved_game')):
            return self.env.get_saved_game()
        return None

class AutoDraw(DiplomacyWrapper):
    """ Wrapper that automatically draws according to the prob of draw in the supervised dataset """

    # This represents the probability of drawing in a specific year
    # e.g. 1.766% of having a draw in 1906
    # The probabilities were computed by taking the prob of ending a game in a given year * probability of having a draw
    # The probability of having a draw is 43.3% according to the dataset.
    probs = {6: 0.01766, 7: 0.02576, 8: 0.04069, 9: 0.05184, 10: 0.05178, 11: 0.05015, 12: 0.04069, 13: 0.03304,
             14: 0.02495, 15: 0.01942, 16: 0.01434, 17: 0.00979, 18: 0.00670, 19: 0.00572, 20: 0.00390, 21: 0.00218,
             22: 0.00189, 23: 0.00133, 24: 0.00130, 25: 0.00094, 26: 0.00055, 27: 0.00059, 28: 0.00020, 29: 0.00026,
             30: 0.00023, 31: 0.00003, 32: 0.00007, 33: 0.00007, 34: 0.00003, 50: 1.}

    def __init__(self, env):
        """ Constructor
            :param env: The wrapped env
            :type env: gym.core.Env
        """
        super(AutoDraw, self).__init__(env)
        self.draw_probs = {power_name: 0. for power_name in self.env.get_all_powers_name()}
        self.draw_actions = {power_name: False for power_name in self.env.get_all_powers_name()}
        self._is_done = False
        self._draw_year = 0
        self._draw_prob = 0.

    @property
    def is_done(self):
        """ Determines if the game is done """
        if self._is_done:
            self.game.note = 'Auto-Draw. (%d - %.2f%%).' % (self._draw_year, 100. * self._draw_prob)
        return self.env.is_done or self._is_done

    @property
    def done_reason(self):
        """ Returns the reason why the game was terminated """
        if self._is_done:
            return DoneReason.AUTO_DRAW
        return self.env.done_reason

    def process(self, **kwargs):
        """ Requests that the game processes the current orders """
        if self._is_done:
            LOGGER.error('The game was auto-drawn. Please reset the game.')
            return

        # Computing the adjusted draw probability
        draw_prob = self.probs.get(self.current_year, 0.)
        survivors = [power.name for power in self.game.powers.values() if power.centers]
        adj_draw_prob = (draw_prob / 3) ** (1 / len(survivors))         # distributing over S..M, F..M, W..A

        # Sampling for each power if they vote for a draw or not
        # Draws should not happen in the first five years
        for power_name in self.draw_probs:
            if power_name not in survivors or self.current_year <= 5:
                self.draw_actions[power_name] = False
                continue

            # Adjusting the draw prob, 50% on the dataset prob, 50% on the model prob to have some exploration
            # Except when there is no draw prob set (we only use the adj draw prob)
            # Or if we reached the last year, we force a draw
            if draw_prob == 1.:
                power_draw_prob = 1.
            elif not self.draw_probs[power_name]:
                power_draw_prob = adj_draw_prob
            else:
                power_draw_prob = 0.5 * self.draw_probs[power_name] + 0.5 * adj_draw_prob
            self.draw_actions[power_name] = bool(uniform(0, 1) <= power_draw_prob)

        # Declaring a draw if all survivors accepted the draw
        if len([1 for draw_action in self.draw_actions.values() if draw_action]) >= len(survivors):
            self._is_done = True
            self._draw_year = int(self.game.get_current_phase()[1:5])
            self._draw_prob = self.probs.get(self.current_year, 0.)

        # Processing, and resetting draw probs
        self.env.process(**kwargs)
        self.draw_probs = {power_name: 0. for power_name in self.draw_probs}

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration  """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        self.draw_probs = {power_name: 0. for power_name in self.env.get_all_powers_name()}
        self.draw_actions = {power_name: False for power_name in self.env.get_all_powers_name()}
        self._is_done = False
        self._draw_year = 0
        self._draw_prob = 0.
        return self.env.reset(**kwargs)

    def set_draw_prob(self, power_name, draw_prob):
        """ Sets the probability of accepting a draw for a given power """
        self.draw_probs[power_name] = draw_prob

    def get_draw_actions(self):
        """ Returns a dictionary with power_name as key, and draw_action (True/False) as value """
        return self.draw_actions

class LimitNumberYears(DiplomacyWrapper):
    """ Wrapper that limits the length of a game to a certain number of years """
    def __init__(self, env, max_nb_years):
        """ Constructor
            :param env: The wrapped env
            :param max_nb_years: The maximum number of years
            :type env: gym.core.Env
        """
        super(LimitNumberYears, self).__init__(env)
        self._max_nb_years = max_nb_years

    @property
    def is_done(self):
        """ Determines if the game is done """
        if self.current_year == self._max_nb_years + 1:
            self.game.note = 'Limit reached ({} years).'.format(self._max_nb_years)
        return self.env.is_done or self.current_year > self._max_nb_years

    @property
    def done_reason(self):
        """ Returns the reason why the game was terminated """
        if self.current_year > self._max_nb_years:
            return DoneReason.PHASE_LIMIT
        return self.env.done_reason

    def process(self, **kwargs):
        """ Requests that the game processes the current orders """
        if self.current_year > self._max_nb_years:
            LOGGER.error('The maximum number of years have been reached. Please reset() the game.')
            return None
        return self.env.process(**kwargs)

class LoopDetection(DiplomacyWrapper):
    """ Wrapper that ends the game when the same state is detected multiple times (i.e. loops / thrashing) """
    def __init__(self, env, threshold=3):
        """ Constructor
            :param env: The wrapped env
            :param threshold: The minimum number of times to detect the same state before ending the game.
            :type env: gym.core.Env
        """
        super(LoopDetection, self).__init__(env)
        self.hash_count = {}
        self.threshold = threshold

    @property
    def is_done(self):
        """ Determines if the game is done """
        state_count = self.hash_count.get(self.game.get_hash(), 0)
        if state_count == self.threshold:
            self.game.note = 'Loop detected ({} occurrences).'.format(self.threshold)
        return self.env.is_done or state_count >= self.threshold

    @property
    def done_reason(self):
        """ Returns the reason why the game was terminated """
        state_count = self.hash_count.get(self.game.get_hash(), 0)
        if state_count >= self.threshold:
            return DoneReason.THRASHED
        return self.env.done_reason

    def process(self, **kwargs):
        """ Requests that the game processes the current orders """
        if self.hash_count.get(self.game.get_hash(), 0) >= self.threshold:
            LOGGER.error('A loop (%d occurrences) has been detected. Please reset() the game.', self.threshold)
            return
        self.env.process(**kwargs)
        self.hash_count[self.game.get_hash()] = self.hash_count.get(self.game.get_hash(), 0) + 1

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and reset the state count """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        self.hash_count = {}
        return self.env.reset(**kwargs)

class SetInitialState(DiplomacyWrapper):
    """ Wrapper that sets the initial state to a different state of the game """
    def __init__(self, env, state_proto):
        """ Constructor
            :param env: The wrapped env
            :param state_proto: A state proto representing the initial state of the game.
            :type env: gym.core.Env
        """
        super(SetInitialState, self).__init__(env)
        self._initial_state = proto_to_dict(state_proto)

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and resets the initial state """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        return_var = self.env.reset(**kwargs)
        self.game.set_state(self._initial_state)
        return return_var

class AssignPlayers(DiplomacyWrapper):
    """ Wrapper that assign the same power to each player on every reset """
    def __init__(self, env, players, power_assignments):
        """ Constructor
            :param env: The wrapped env
            :param players: A list of player instances
            :param power_assignments: The list of powers (in order) to assign to each player
            :type env: gym.core.Env
            :type players: List[diplomacy_research.players.player.Player]
        """
        super(AssignPlayers, self).__init__(env)
        game = self.game or Game()

        # Making sure we have the correct number of powers
        assert len(power_assignments) == len(game.powers.keys())
        assert sorted(power_assignments) == sorted(game.powers.keys())

        # Setting fixed ordering
        self._powers = power_assignments
        self._players = players

    @property
    def players(self):
        """ Returns a list of players instances playing the game """
        return self._players

    def get_all_powers_name(self):
        """ Returns the power for all players """
        return self._powers

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and re-assign the correct ordering """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        return_var = self.env.reset(**kwargs)
        self.game.note = ' / '.join([power_name[:3] for power_name in self._powers])
        return return_var

class RandomizePlayers(DiplomacyWrapper):
    """ Wrapper that randomizes the power each player is playing at every reset """
    def __init__(self, env, players, clusters=None):
        """ Constructor
            :param env: The wrapped env
            :param players: A list of player instances (all 7 players if no clusters, otherwise 1 player per cluster)
            :param clusters: Optional. Contains a list of clusters, where each cluster is a tuple with
                    - 1) the number of players inside it, 2) a boolean that indicates if the players need to be merged

                    e.g. [(1,False), (3, False), (3, False)] would create a 1 vs 3 vs 3 game (3 clusters), where
                    players[0] is assigned to the first cluster,
                    players[1] is copied 3 times and assigned to the second cluster, and
                    players[2] is copied 3 times and assigned to the third cluster.
            :type env: gym.core.Env
            :type players: List[diplomacy_research.players.player.Player]
        """
        super(RandomizePlayers, self).__init__(env)
        game = self.game or Game()
        self._powers = [power_name for power_name in game.powers]
        self._clusters = clusters if clusters is not None else [(1, False)] * len(self._powers)

        # Making sure we have the right number of players
        if len(players) < len(self._clusters):
            LOGGER.error('The nb of players (%d) must be greater or equal to %d.', len(players), len(self._clusters))
            raise ValueError()

        # Generating the list of players based on the clusters definition
        self._players = []
        for cluster_ix, cluster_def in enumerate(self._clusters):
            nb_players_in_cluster = cluster_def[0]
            self._players += [players[cluster_ix]] * nb_players_in_cluster

        # Generating an ordering to shuffle players
        self._ordering = list(range(len(self._players)))

    @property
    def players(self):
        """ Returns a list of players instances playing the game """
        return self._players

    def get_all_powers_name(self):
        """ Returns the power for all players """
        return [self._powers[self._ordering[player_ix]] for player_ix in range(len(self._players))]

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and shuffles the player ordering """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        shuffle(self._ordering)
        return_var = self.env.reset(**kwargs)
        note_to_display = []

        # Determining power assigned to each cluster, and merging if necessary
        remaining_players, remaining_powers = self._players[:], self.get_all_powers_name()
        for nb_players_in_cluster, merge_cluster in self._clusters:
            cluster_players = remaining_players[:nb_players_in_cluster]
            cluster_powers = remaining_powers[:nb_players_in_cluster]
            remaining_players = remaining_players[nb_players_in_cluster:]
            remaining_powers = remaining_powers[nb_players_in_cluster:]

            # Merging all powers in cluster into merged_power
            if merge_cluster and cluster_players:
                merged_power = None
                for power_ix, power_name in enumerate(cluster_powers):
                    current_power = self.game.get_power(power_name)
                    if power_ix == 0:
                        merged_power = current_power
                    else:
                        merged_power.merge(current_power)

            # Concatenating note
            note_to_display += [' '.join([power_name[:3] for power_name in cluster_powers])]

        # Displaying note and returning
        self.game.note = ' / '.join(note_to_display)
        return return_var

class SetPlayerSeed(DiplomacyWrapper):
    """ Wrapper that automatically sets the player seed on every reset """
    def __init__(self, env, players):
        """ Constructor
            :param env: The wrapped env
            :param players: A list of player instances
            :type env: gym.core.Env
            :type players: List[diplomacy_research.players.player.Player]
        """
        super(SetPlayerSeed, self).__init__(env)
        self._players = players

        # Making sure players are separate objects
        nb_players = len(players)
        for player_ix in range(1, nb_players):
            if players[0] == players[player_ix]:
                raise ValueError('Player 0 and %d are the same object. Different player objects expected.' % player_ix)

    @property
    def players(self):
        """ Returns a list of players instances playing the game """
        return self._players

    def reset(self, **kwargs):
        """ Sets the player seed for each player based on the game id """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        return_var = self.env.reset(**kwargs)
        powers = self.env.get_all_powers_name()
        player_seeds = self.env.get_player_seeds()                          # {power_name: player_seed}
        for power_name, player in zip(powers, self._players):
            player.player_seed = player_seeds[power_name]
        return return_var

class SaveGame(DiplomacyWrapper):
    """ Wrapper that saves game to memory / disk """
    def __init__(self, env, output_directory=None):
        """ Constructor
            :param env: The wrapped env
            :param output_directory: The directory were to save the games (None to only save in memory)
            :type env: gym.core.Env
        """
        super(SaveGame, self).__init__(env)
        self.output_dir = output_directory
        self.saved = False
        self.saved_game = None

        # Creating directory, it might be created multiple times in parallel
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _save_game(self):
        """ Save the buffer to disk """
        # Building saved game
        self.saved_game = to_saved_game_format(self.game)
        self.saved = True

        # Saving to disk
        if self.output_dir:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, '{}_{}.json'.format(timestamp, self.game_id))
            with open(output_path, 'w') as file:
                file.write(json.dumps(self.saved_game))

    def process(self, **kwargs):
        """ Clearing state cache and saving it to buffer """
        return_var = self.env.process(**kwargs)
        if self.is_done:
            self._save_game()
        return return_var

    def reset(self, **kwargs):
        """ Resets the game to its starting configuration and saves it to disk """
        # _reset was deprecated in gym 0.9.6, method_hidden triggered by gym.core L235
        # pylint: disable=method-hidden
        if not self.saved and self.game and self.game.state_history:
            self._save_game()
        return_var = self.env.reset(**kwargs)
        self.saved = False
        return return_var

    def get_saved_game(self):
        """ Returns the last saved game """
        if not self.saved:
            return to_saved_game_format(self.game)
        return self.saved_game
