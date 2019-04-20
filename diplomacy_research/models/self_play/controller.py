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
""" Controller
    This module is responsible for generating self-play trajectories (and returning saved games)
"""
from collections import OrderedDict
from random import shuffle
import time
from diplomacy import Map
import gym
from tornado import gen
from diplomacy_research.models.gym.environment import DiplomacyEnv, DoneReason
from diplomacy_research.models.gym.wrappers import SaveGame, LimitNumberYears, SetInitialState, AssignPlayers, \
    RandomizePlayers, AutoDraw, LoopDetection, SetPlayerSeed
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto, get_map_powers, proto_to_board_state, proto_to_prev_orders_state, NB_POWERS, \
    NB_PREV_ORDERS_HISTORY
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto, State as StateProto
from diplomacy_research.utils.proto import dict_to_proto, proto_to_bytes, proto_to_zlib, bytes_to_proto


def default_env_constructor(players, hparams, power_assignments, set_player_seed, initial_state_bytes):
    """ Default gym environment constructor
        :param players: A list of instantiated players
        :param hparams: A dictionary of hyper parameters with their values
        :param power_assignments: Optional. The power name we want to play as. (e.g. 'FRANCE') or a list of powers.
        :param set_player_seed: Boolean that indicates that we want to set the player seed on reset().
        :param initial_state_bytes: A `game.State` proto (in bytes format) representing the initial state of the game.
        :rtype: diplomacy_research.models.gym.wrappers.DiplomacyWrapper
    """
    # The keys should be present in hparams if it's not None
    max_nb_years = hparams['max_nb_years'] if hparams else 35
    auto_draw = hparams['auto_draw'] if hparams else False
    power = hparams['power'] if hparams else ''
    nb_thrashing_states = hparams['nb_thrashing_states'] if hparams else 0

    env = gym.make('DiplomacyEnv-v0')
    env = LimitNumberYears(env, max_nb_years)

    # Loop Detection (Thrashing)
    if nb_thrashing_states:
        env = LoopDetection(env, threshold=nb_thrashing_states)

    # Auto-Drawing
    if auto_draw:
        env = AutoDraw(env)

    # Setting initial state
    if initial_state_bytes is not None:
        env = SetInitialState(env, bytes_to_proto(initial_state_bytes, StateProto))

    # 1) If power_assignments is a list, using that to assign powers
    if isinstance(power_assignments, list):
        env = AssignPlayers(env, players, power_assignments)

    # 2a) If power_assignments is a string, using that as our power, and randomly assigning the other powers
    # 2b) Using hparams['power'] if it's set.
    elif isinstance(power_assignments, str) or power:
        our_power = power_assignments if isinstance(power_assignments, str) else power
        other_powers = [power_name for power_name in env.get_all_powers_name() if power_name != our_power]
        shuffle(other_powers)
        env = AssignPlayers(env, players, [our_power] + other_powers)

    # 3) Randomly shuffle the powers
    else:
        env = RandomizePlayers(env, players)

    # Setting player seed
    if set_player_seed:
        env = SetPlayerSeed(env, players)

    # Ability to save game / retrieve saved game
    env = SaveGame(env)
    return env

def get_saved_game_proto(env, players, stored_board_state, stored_prev_orders_state, stored_possible_orders,
                         power_variables, start_phase_ix, reward_fn, advantage_fn, is_partial_game):
    """ Extracts the saved game proto from the environment to send back to the learner
        :param env: The gym environment (needs to implement a SaveGame wrapper)
        :param players: A list of instantiated players
        :param stored_board_state: A dictionary with phase_name as key and board_state as value
        :param stored_prev_orders_state: A dictionary with phase_name as key and prev_orders_state as value
        :param stored_possible_orders: A dictionary with phase_name as key and possible orders as value
        :param power_variables: A dict containing orders, policy details, values, rewards, returns for each power
        :param start_phase_ix: For partial game, the index of the phase from which to start learning
        :param reward_fn: The reward function to use to calculate rewards
        :param advantage_fn: An instance of `.models.self_play.advantages`
        :param is_partial_game: Boolean that indicates that we are processing an incomplete game
        :return: The saved game in proto format
    """
    # pylint: disable=too-many-arguments
    powers = sorted([power_name for power_name in get_map_powers(env.game.map)])
    assigned_powers = env.get_all_powers_name()

    # Computing returns
    for power_name in powers:
        rewards = power_variables[power_name]['rewards']
        state_values = power_variables[power_name]['state_values']
        last_state_value = power_variables[power_name]['last_state_value']
        power_variables[power_name]['returns'] = advantage_fn.get_returns(rewards, state_values, last_state_value)

    # Retrieving saved game
    saved_game = env.get_saved_game()

    # Restoring stored variables on the saved game before converting to proto
    for phase_ix, phase in enumerate(saved_game['phases']):

        # Last phase - Only storing state value
        if phase_ix == len(saved_game['phases']) - 1:
            state_values = {power_name: float(power_variables[power_name]['state_values'][-1]) for power_name in powers}
            phase['state_value'] = state_values
            break

        # Setting shared fields (board_state, prev_orders_state, possible_orders)
        phase['state']['board_state'] = stored_board_state[phase['name']]
        if phase['name'][-1] == 'M':
            phase['prev_orders_state'] = stored_prev_orders_state[phase['name']]
        phase['possible_orders'] = {loc: stored_possible_orders[phase['name']][loc].value
                                    for loc in stored_possible_orders[phase['name']]}

        # Setting orders, policy_details, state_values
        phase['orders'] = {power_name: power_variables[power_name]['orders'][phase_ix] for power_name in powers}
        phase['policy'] = {power_name: power_variables[power_name]['policy_details'][phase_ix] for power_name in powers}
        phase['state_value'] = {power_name: float(power_variables[power_name]['state_values'][phase_ix])
                                for power_name in powers}

    # Adding power assignments, done reason, and kwargs
    done_reason = env.done_reason.value if env.done_reason is not None else ''
    saved_game['done_reason'] = done_reason
    saved_game['assigned_powers'] = assigned_powers
    saved_game['players'] = [player.name for player in players]
    saved_game['kwargs'] = {power_name: players[assigned_powers.index(power_name)].kwargs for power_name in powers}
    saved_game['is_partial_game'] = is_partial_game
    saved_game['start_phase_ix'] = start_phase_ix if is_partial_game else 0
    saved_game['reward_fn'] = reward_fn.name
    saved_game['rewards'] = {power_name: power_variables[power_name]['rewards'] for power_name in powers}
    saved_game['returns'] = {power_name: power_variables[power_name]['returns'] for power_name in powers}

    # Returning
    return dict_to_proto(saved_game, SavedGameProto)

@gen.coroutine
def get_step_args(player, state_proto, power_name, phase_history_proto, possible_orders_proto):
    """ Gets the arguments required to step through the environment.
        :param player: An instantiated player
        :param state_proto: A `.proto.game.State` representation of the state of the game.
        :param power_name: The name of the power we are playing
        :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
        :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
        :return: A tuple with
                1) The list of orders the power should play (e.g. ['A PAR H', 'A MAR - BUR', ...])
                2) The policy details ==> {'locs', 'tokens', 'log_probs', 'draw_action', 'draw_prob'}
                3) The state value for the given state
        :type player: diplomacy_research.players.player.Player
    """
    if not state_proto.units[power_name].value:
        return [], None, 0.

    return (yield player.get_orders_details_with_proto(state_proto,
                                                       power_name,
                                                       phase_history_proto,
                                                       possible_orders_proto,
                                                       with_state_value=True))

@gen.coroutine
def get_state_value(player, state_proto, power_name, phase_history_proto, possible_orders_proto):
    """ Gets the value of the state for the given power (e.g. the last state value)
        :param player: An instantiated player
        :param state_proto: A `.proto.game.State` representation of the state of the game.
        :param power_name: The name of the power we are playing
        :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
        :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
        :return: The state value for the given state
        :type player: diplomacy_research.players.player.Player
    """
    if not state_proto.units[power_name].value:
        return 0.

    return (yield player.get_state_value_with_proto(state_proto,
                                                    power_name,
                                                    phase_history_proto,
                                                    possible_orders_proto))

@gen.coroutine
def generate_trajectory(players, reward_fn, advantage_fn, env_constructor=None, hparams=None, power_assignments=None,
                        set_player_seed=None, initial_state_bytes=None, update_interval=0, update_queue=None,
                        output_format='proto'):
    """ Generates a single trajectory (Saved Gamed Proto) for RL (self-play) with the power assigments
        :param players: A list of instantiated players
        :param reward_fn: The reward function to use to calculate rewards
        :param advantage_fn: An instance of `.models.self_play.advantages`
        :param env_constructor: A callable to get the OpenAI gym environment (args: players)
        :param hparams: A dictionary of hyper parameters with their values
        :param power_assignments: Optional. The power name we want to play as. (e.g. 'FRANCE') or a list of powers.
        :param set_player_seed: Boolean that indicates that we want to set the player seed on reset().
        :param initial_state_bytes: A `game.State` proto (in bytes format) representing the initial state of the game.
        :param update_interval: Optional. If set, a partial saved game is put in the update_queue this every seconds.
        :param update_queue: Optional. If update interval is set, partial games will be put in this queue
        :param output_format: The output format. One of 'proto', 'bytes', 'zlib'
        :return: A SavedGameProto representing the game played (with policy details and power assignments)
                 Depending on format, the output might be converted to a byte array, or a compressed byte array.
        :type players: List[diplomacy_research.players.player.Player]
        :type reward_fn: diplomacy_research.models.self_play.reward_functions.AbstractRewardFunction
        :type advantage_fn: diplomacy_research.models.self_play.advantages.base_advantage.BaseAdvantage
        :type update_queue: multiprocessing.Queue
    """
    # pylint: disable=too-many-arguments
    assert output_format in ['proto', 'bytes', 'zlib'], 'Format should be "proto", "bytes", "zlib"'
    assert len(players) == NB_POWERS

    # Making sure we use the SavedGame wrapper to record the game
    if env_constructor:
        env = env_constructor(players)
    else:
        env = default_env_constructor(players, hparams, power_assignments, set_player_seed, initial_state_bytes)
    wrapped_env = env
    while not isinstance(wrapped_env, DiplomacyEnv):
        if isinstance(wrapped_env, SaveGame):
            break
        wrapped_env = wrapped_env.env
    else:
        env = SaveGame(env)

    # Detecting if we have a Auto-Draw wrapper
    has_auto_draw = False
    wrapped_env = env
    while not isinstance(wrapped_env, DiplomacyEnv):
        if isinstance(wrapped_env, AutoDraw):
            has_auto_draw = True
            break
        wrapped_env = wrapped_env.env

    # Resetting env
    env.reset()

    # Timing vars for partial updates
    time_last_update = time.time()
    year_last_update = 0
    start_phase_ix = 0
    current_phase_ix = 0
    nb_transitions = 0

    # Cache Variables
    powers = sorted([power_name for power_name in get_map_powers(env.game.map)])
    assigned_powers = env.get_all_powers_name()
    stored_board_state = OrderedDict()                              # {phase_name: board_state}
    stored_prev_orders_state = OrderedDict()                        # {phase_name: prev_orders_state}
    stored_possible_orders = OrderedDict()                          # {phase_name: possible_orders}

    power_variables = {power_name: {'orders': [],
                                    'policy_details': [],
                                    'state_values': [],
                                    'rewards': [],
                                    'returns': [],
                                    'last_state_value': 0.} for power_name in powers}

    new_state_proto = None
    phase_history_proto = []
    map_object = Map(name=env.game.map.name)

    # Generating
    while not env.is_done:
        state_proto = new_state_proto if new_state_proto is not None else extract_state_proto(env.game)
        possible_orders_proto = extract_possible_orders_proto(env.game)

        # Computing board_state
        board_state = proto_to_board_state(state_proto, map_object).flatten().tolist()
        state_proto.board_state.extend(board_state)

        # Storing possible orders for this phase
        current_phase = env.game.get_current_phase()
        stored_board_state[current_phase] = board_state
        stored_possible_orders[current_phase] = possible_orders_proto

        # Getting orders, policy details, and state value
        tasks = [(player,
                  state_proto,
                  pow_name,
                  phase_history_proto[-NB_PREV_ORDERS_HISTORY:],
                  possible_orders_proto) for player, pow_name in zip(env.players, assigned_powers)]
        step_args = yield [get_step_args(*args) for args in tasks]

        # Stepping through env, storing power variables
        for power_name, (orders, policy_details, state_value) in zip(assigned_powers, step_args):
            if orders:
                env.step((power_name, orders))
                nb_transitions += 1
            if has_auto_draw and policy_details is not None:
                env.set_draw_prob(power_name, policy_details['draw_prob'])

        # Processing
        env.process()
        current_phase_ix += 1

        # Retrieving draw action and saving power variables
        for power_name, (orders, policy_details, state_value) in zip(assigned_powers, step_args):
            if has_auto_draw and policy_details is not None:
                policy_details['draw_action'] = env.get_draw_actions()[power_name]
            power_variables[power_name]['orders'] += [orders]
            power_variables[power_name]['policy_details'] += [policy_details]
            power_variables[power_name]['state_values'] += [state_value]

        # Getting new state
        new_state_proto = extract_state_proto(env.game)

        # Storing reward for this transition
        done_reason = DoneReason(env.done_reason) if env.done_reason else None
        for power_name in powers:
            power_variables[power_name]['rewards'] += [reward_fn.get_reward(prev_state_proto=state_proto,
                                                                            state_proto=new_state_proto,
                                                                            power_name=power_name,
                                                                            is_terminal_state=done_reason is not None,
                                                                            done_reason=done_reason)]

        # Computing prev_orders_state for the previous state
        last_phase_proto = extract_phase_history_proto(env.game, nb_previous_phases=1)[-1]
        if last_phase_proto.name[-1] == 'M':
            prev_orders_state = proto_to_prev_orders_state(last_phase_proto, map_object).flatten().tolist()
            stored_prev_orders_state[last_phase_proto.name] = prev_orders_state
            last_phase_proto.prev_orders_state.extend(prev_orders_state)
            phase_history_proto += [last_phase_proto]

        # Sending partial game if:
        # 1) We have update_interval > 0 with an update queue, and
        # 2a) The game is completed, or 2b) the update time has elapsted and at least 5 years as passed
        has_update_interval = update_interval > 0 and update_queue is not None
        game_is_completed = env.is_done
        min_time_has_passed = time.time() - time_last_update > update_interval
        current_year = 9999 if env.game.get_current_phase() == 'COMPLETED' else int(env.game.get_current_phase()[1:5])
        min_years_have_passed = current_year - year_last_update >= 5

        if (has_update_interval
                and (game_is_completed
                     or (min_time_has_passed and min_years_have_passed))):

            # Game is completed - last state value is 0
            if game_is_completed:
                for power_name in powers:
                    power_variables[power_name]['last_state_value'] = 0.

            # Otherwise - Querying the model for the value of the last state
            else:
                tasks = [(player,
                          new_state_proto,
                          pow_name,
                          phase_history_proto[-NB_PREV_ORDERS_HISTORY:],
                          possible_orders_proto) for player, pow_name in zip(env.players, assigned_powers)]
                last_state_values = yield [get_state_value(*args) for args in tasks]

                for power_name, last_state_value in zip(assigned_powers, last_state_values):
                    power_variables[power_name]['last_state_value'] = last_state_value

            # Getting partial game and sending it on the update_queue
            saved_game_proto = get_saved_game_proto(env=env,
                                                    players=players,
                                                    stored_board_state=stored_board_state,
                                                    stored_prev_orders_state=stored_prev_orders_state,
                                                    stored_possible_orders=stored_possible_orders,
                                                    power_variables=power_variables,
                                                    start_phase_ix=start_phase_ix,
                                                    reward_fn=reward_fn,
                                                    advantage_fn=advantage_fn,
                                                    is_partial_game=True)
            update_queue.put_nowait((False, nb_transitions, proto_to_bytes(saved_game_proto)))

            # Updating stats
            start_phase_ix = current_phase_ix
            nb_transitions = 0
            if not env.is_done:
                year_last_update = int(env.game.get_current_phase()[1:5])

    # Since the environment is done (Completed game) - We can leave the last_state_value at 0.
    for power_name in powers:
        power_variables[power_name]['last_state_value'] = 0.

    # Getting completed game
    saved_game_proto = get_saved_game_proto(env=env,
                                            players=players,
                                            stored_board_state=stored_board_state,
                                            stored_prev_orders_state=stored_prev_orders_state,
                                            stored_possible_orders=stored_possible_orders,
                                            power_variables=power_variables,
                                            start_phase_ix=0,
                                            reward_fn=reward_fn,
                                            advantage_fn=advantage_fn,
                                            is_partial_game=False)

    # Converting to correct format
    output = {'proto': lambda proto: proto,
              'zlib': proto_to_zlib,
              'bytes': proto_to_bytes}[output_format](saved_game_proto)

    # Returning
    return output
