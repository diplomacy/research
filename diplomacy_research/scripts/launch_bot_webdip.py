#!/usr/bin/env python3
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
""" Small script to run the bot that plays on webdiplomacy.net
    You can stop the bot with keyboard interruption (Ctrl+C).
"""
import argparse
import logging
import os
import sys
import time
import traceback
from diplomacy.integration.webdiplomacy_net.api import API
from diplomacy.utils import exceptions
from tornado import gen, ioloop, util
from diplomacy_research.players.benchmark_player import WebDiplomacyPlayer
from diplomacy_research.utils.openings import get_orders_from_gunboat_openings

# Constants
LOGGER = logging.getLogger('diplomacy_research.scripts.launch_bot_webdip')
PERIOD_SECONDS = 5
PERIOD_ERRORS = 300
MAX_ERRORS = 5
NB_MODELS = 2

class GunboatSamplingPlayer():
    """ Mock player that samples orders according to a distribution """

    @staticmethod
    @gen.coroutine
    def get_orders(_, power_name):
        """ Samples according to the gunboat opening orders distribution """
        return get_orders_from_gunboat_openings(power_name)


class WebDipBot():
    """ WebDipBot class """
    __slots__ = ['period_seconds', 'players', 'max_batch_size', 'api_cd', 'api_users', 'errors', 'known_game_ids']

    def __init__(self, *, period_seconds=PERIOD_SECONDS, max_batch_size=32):
        """ Initialize the bot.
            :param period_seconds: time in second between two consecutive bot queries on server. Default 3 seconds.
            :param max_batch_size: The maximum number of powers to query for orders during a given interval.
        """
        self.period_seconds = period_seconds
        self.players = None
        self.max_batch_size = max_batch_size
        self.api_cd = []                            # Keys stored in API_KEY_CD_01 to API_KEY_CD_20
        self.api_users = []                         # Keys stored in API_KEY_USER_01 to API_KEY_USER_20
        self.errors = {}                            # {'game_id/country_id': [timestamps]}
        self.known_game_ids = set()                 # Set of game ids where we already printed the assignments

        # Loading API keys
        for key_ix in range(1, 21):
            if os.environ.get('API_KEY_CD_%02d' % key_ix, ''):
                self.api_cd += [API(os.environ['API_KEY_CD_%02d' % key_ix])]
                LOGGER.info('Using CD key [%d]: %s', key_ix, os.environ['API_KEY_CD_%02d' % key_ix])
            if os.environ.get('API_KEY_USER_%02d' % key_ix, ''):
                self.api_users += [API(os.environ['API_KEY_USER_%02d' % key_ix])]
                LOGGER.info('Using API key [%d]: %s', key_ix, os.environ['API_KEY_USER_%02d' % key_ix])

    @gen.coroutine
    def run(self):
        """ Main bot code. """
        if not self.api_cd and not self.api_users:
            LOGGER.error('No API keys detected. Exiting')
            return

        # Creating player
        self.players = {'beam_1.00': WebDiplomacyPlayer(temperature=1.00, use_beam=True),
                        'beam_0.50': WebDiplomacyPlayer(temperature=0.50, use_beam=True),
                        'beam_0.25': WebDiplomacyPlayer(temperature=0.25, use_beam=True),
                        'beam_0.10': WebDiplomacyPlayer(temperature=0.10, use_beam=True),
                        'greedy_1.00': WebDiplomacyPlayer(temperature=1.00, use_beam=False),
                        'greedy_0.50': WebDiplomacyPlayer(temperature=0.50, use_beam=False),
                        'greedy_0.25': WebDiplomacyPlayer(temperature=0.25, use_beam=False),
                        'greedy_0.10': WebDiplomacyPlayer(temperature=0.10, use_beam=False),
                        'gunboat_sampler': GunboatSamplingPlayer()}
        yield self.players['greedy_0.10'].check_openings()

        # List of all APIs
        all_apis = self.api_cd + self.api_users

        while True:
            try:
                api_list_games = yield ([api.list_games_with_players_in_cd() for api in self.api_cd] +
                                        [api.list_games_with_missing_orders() for api in self.api_users])

                # Set of game_id, country_id
                set_game_country = set()

                # Extracting tuples of (api, game_id, country_id)
                games_with_api = []
                for api, list_of_games in zip(all_apis, api_list_games):
                    for (game_id, country_id) in list_of_games[:self.max_batch_size]:
                        if (game_id, country_id) in set_game_country:
                            continue
                        games_with_api += [(api, game_id, country_id)]
                        set_game_country.add((game_id, country_id))

                # Submitting orders sequentially
                for api, game_id, country_id in games_with_api:
                    yield self.submit_orders(api, game_id, country_id)

                # Only sleeping if no games were retrieved
                if not games_with_api:
                    yield gen.sleep(self.period_seconds)

            # Server error - Logging, but continuing
            except (exceptions.DiplomacyException, util.TimeoutError, RuntimeError) as error:
                LOGGER.error(error)

    @gen.coroutine
    def submit_orders(self, api, game_id, country_id):
        """ Generate and submit orders for a given game_id / country_id using the API.
            :param api: A API instance to query the webdiplomacy.net server
            :param game_id: An integer representing the game_id to query from the server.
            :param country_id: An integer representing the country id we want to play as.
            :return: Nothing, but submit the orders on the server.
            :type api: API
        """
        errors = self.get_errors(game_id, country_id)

        # Too many recent errors - Skipping to throttle
        if len(errors) >= MAX_ERRORS:
            return

        # Querying the game and power
        # 16 = 3 yrs @ 5 phases/yr + current phase
        game, power_name = yield api.get_game_and_power(game_id, country_id, max_phases=16)
        if game is None:
            self.add_error(game_id, country_id)
            return

        # Querying the assignments
        models = self.get_model_per_power(game)
        current_phase = game.get_current_phase()
        if game.game_id not in self.known_game_ids and game.map_name == 'standard' and current_phase == 'S1901M':
            LOGGER.info('[%s] Models per power are: %s', game_id, models)
            self.known_game_ids.add(game_id)

        # Querying the model for object
        player = self.get_player(game, power_name, models)
        if player is None:
            LOGGER.error('Unable to retrieve the player for map %s.', game.map_name)
            return
        orders = yield player.get_orders(game, power_name)
        orders = self.adjust_orders(orders, game, power_name)

        # Submitting orders
        success = yield api.set_orders(game, power_name, orders, wait=False)
        if not success:
            self.add_error(game_id, country_id)

    @staticmethod
    def get_model_per_power(game, nb_models=NB_MODELS):
        """ Computes a dictionary of model version per power
            :param game: The current game object
            :param nb_models: The number of models
            :type game: diplomacy.Game
            :return: A dictionary of {power_name: model_id}
        """
        game_id = int(game.game_id) if str(game.game_id).isdigit() else 0
        assignments = (3079 * game_id) % (nb_models ** 7)            # 3079 is a prime for even hashing
        models = {}

        # Decode the assignments (nb_models ** nb_players)
        # Each player has the same probability of being assigned to each model
        # The assignment will always be the same for a given game_id / power_name
        for power_name in game.powers:
            models[power_name] = (assignments % nb_models)
            assignments = assignments // nb_models
        return models

    def get_player(self, game, power_name, models):
        """ Returns the player to query the orders
            :param game: A game instance
            :param power_name: The name of the power we are playing
            :param models: A dictionary of {power_name: model_id}
            :type game: diplomacy.Game
            :return: A player object to query the orders
        """
        # pylint: disable=too-many-return-statements
        # Standard map
        if game.map_name == 'standard':

            # v1 - Uses the new gunboat empirical openings
            if models.get(power_name, 0) == 1:
                if game.get_current_phase() == 'S1901M':                # Uses the empirical openings
                    return self.players['gunboat_sampler']
                if game.get_current_phase() == 'F1901M':                # Beam to get diverse second moves
                    return self.players['beam_0.25']
                if self.game_stuck_in_local_optimum(game, power_name):  # In case the bot gets stuck
                    return self.players['greedy_1.00']
                return self.players['greedy_0.50']                      # Greedy by default

            # v0 - Default model
            if game.get_current_phase() in ('S1901M', 'F1901M'):        # To get a diverse set of openings
                return self.players['beam_0.50']
            if self.game_stuck_in_local_optimum(game, power_name):      # In case the bot gets stuck
                return self.players['beam_0.50']
            return self.players['greedy_0.10']                          # Greedy by default

        # 1v1 map
        if game.map_name in ('standard_france_austria', 'standard_germany_italy'):
            if game.get_current_phase() in ('S1901M', 'F1901M'):    # To get a diverse set of openings
                return self.players['beam_0.50']
            if game.get_current_phase() in ('S1902M', 'F1902M'):    # To get a diverse set of beginning/mid-game
                return self.players['beam_0.10']
            return self.players['greedy_0.10']                      # Greedy by default

        # Unsupported maps
        return None

    @staticmethod
    def game_stuck_in_local_optimum(game, power_name):
        """ Determines if the bots are stuck in a local optimum, to avoid endless loops
            :param game: A game instance
            :param power_name: The name of the power we are playing
            :type game: diplomacy.Game
            :return: A boolean that indicates if a local optimum has been detected.
        """
        # 1) A local optimum can only be detected on the standard map for a valid power
        if game.map_name != 'standard' or power_name not in game.powers:
            return False

        # 2) A local optimum can only happen if a power has 13 or more supply centers
        if max([len(power.centers) for power in game.powers.values()]) < 13:
            return False

        # 3) A local optimum can only happen for an active game
        if game.get_current_phase() == 'COMPLETED':
            return False

        # 4) A local optimum can not happen if the power has the most supply centers
        if len(game.powers[power_name].centers) == max([len(power.centers) for power in game.powers.values()]):
            return False

        # Building a list of units at the start of the last phase of each year for each power
        # e.g. W1901A, W1902A, S1903M if the current phase is S1903M
        # We can use this to detect a power that has not been able to move its units in the last 'x' years
        units = {}                                      # {year: {power_name: set_of_units}}

        # For each phase, for each power
        for phase_name, phase in game.state_history.items():
            phase_year = int(str(phase_name)[1:5])
            units[phase_year] = {power_name: set(phase['units'].get(power_name, [])) for power_name in game.powers}

        # Setting also units in current phase
        current_year = int(game.get_current_phase()[1:5])
        units[current_year] = {power_name: set(game.get_units(power_name)) for power_name in game.powers}

        # Checking if powers have not moved unit in the last 3 years
        powers_stuck = set()
        for pow_name in game.powers:
            units_yr_0 = units.get(current_year, {}).get(pow_name, set())
            units_yr_1 = units.get(current_year - 1, {}).get(pow_name, set())
            units_yr_2 = units.get(current_year - 2, {}).get(pow_name, set())

            # Power can only be stuck if it still has units on the board
            if not units_yr_0:
                continue

            # Power is stuck if (yr_0 == yr_1 and yr_1 == yr_2)
            if units_yr_0 == units_yr_1 == units_yr_2:
                powers_stuck.add(pow_name)

        # 5) A local optimum can only happen if 2 or more powers are stuck (same units in the last 3 years)
        return bool(len(powers_stuck) >= 2 and power_name in powers_stuck)

    @staticmethod
    def adjust_orders(orders, game, power_name):
        """ Performs manual order adjustments to remove edge case scenarios
            :param orders: The list of orders to submit
            :param game: A game instance
            :param power_name: The name of the power we are playing
            :type game: diplomacy.Game
            :return: The adjusted list of orders
        """
        del game, power_name            # Unused args
        adjusted_orders = []
        retreat_locs = set()

        # 1) Only allow one retreat to a given location, convert the others to disband
        for order in orders:
            if ' R ' in order:
                unit, dest = order.split(' R ')
                if dest in retreat_locs:
                    order = '%s D' % unit
                retreat_locs.add(dest)
            adjusted_orders.append(order)

        # Returning adjusted orders
        return adjusted_orders

    def add_error(self, game_id, country_id):
        """ Marks a request as a failure, to throttle if too many failures are detected in a period of time
            :param game_id: An integer representing the game_id to query from the server.
            :param country_id: An integer representing the country id we want to play as.
        """
        error_key = '%s/%s' % (game_id, country_id)
        self.errors.setdefault(error_key, []).append(int(time.time()))

    def get_errors(self, game_id, country_id):
        """ Returns the timestamp of all errors for that game_id/country_id during the PERIOD_ERRORS
            :param game_id: An integer representing the game_id to query from the server.
            :param country_id: An integer representing the country id we want to play as.
            :return: A list of error timestamp in the PERIOD_ERRORS.
        """
        error_key = '%s/%s' % (game_id, country_id)
        errors = [error_time for error_time in self.errors.get(error_key, [])
                  if (int(time.time()) - error_time) <= PERIOD_ERRORS]

        # Updating errors
        if errors:
            self.errors[error_key] = errors
        elif error_key in self.errors:
            del self.errors[error_key]

        # Returning
        return errors

def main():
    """ Main script function. """
    parser = argparse.ArgumentParser(description='Run the bot that plays on webdiplomacy.net.')
    parser.add_argument('--period', type=int, default=PERIOD_SECONDS,
                        help='run every period (in seconds) (default: %d seconds)' % PERIOD_SECONDS)
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='the maximum number of powers to submit orders for. (default: 32 powers)')
    args = parser.parse_args()
    bot = WebDipBot(period_seconds=args.period, max_batch_size=args.max_batch_size)
    io_loop = ioloop.IOLoop.instance()
    while True:
        try:
            io_loop.run_sync(bot.run)
        except KeyboardInterrupt:
            LOGGER.error('Bot interrupted.')
            break
        except Exception as exc:                                                         # pylint: disable=broad-except
            print('--------------------------------------------------------------------------------')
            LOGGER.error(exc)
            traceback.print_exc(file=sys.stdout)
            print('--------------------------------------------------------------------------------')
            LOGGER.info('Restarting bot in 30 secs...')
            time.sleep(30)
            LOGGER.info('Restarting bot now...')

if __name__ == '__main__':
    main()
