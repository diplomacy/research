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
""" Runs tests for gym wrappers """
import gym
from diplomacy_research.models.gym import AutoDraw, LimitNumberYears, LoopDetection, SetInitialState, \
    AssignPlayers, RandomizePlayers, SetPlayerSeed, SaveGame
from diplomacy_research.models.state_space import get_player_seed
from diplomacy_research.proto.diplomacy_proto.game_pb2 import State as StateProto
from diplomacy_research.players import DummyPlayer
from diplomacy_research.utils.proto import zlib_to_proto

def test_auto_draw():
    """ Tests the AutoDraw Wrapper """
    env = gym.make('DiplomacyEnv-v0')
    env = AutoDraw(env)
    env.reset()

    # Expecting env to be done after 10 process
    for _ in range(100):
        if env.is_done:
            break
        env.process()
    assert env.is_done
    final_year = int(env.game.get_current_phase()[1:5])
    assert 1905 <= final_year <= 1951

def test_limit_number_years():
    """ Tests the LimitNumberYears Wrapper """
    env = gym.make('DiplomacyEnv-v0')
    env = LimitNumberYears(env, max_nb_years=10)
    env.reset()

    # Expecting env to be done after 10 process
    for _ in range(100):
        if env.is_done:
            break
        env.process()
    assert env.is_done
    assert env.game.get_current_phase() == 'S1911M'

def test_loop_detection():
    """ Test LoopDetection Wrapper """
    env = gym.make('DiplomacyEnv-v0')
    env = LoopDetection(env, threshold=3)
    env.reset()

    assert env.is_done is False
    nb_process = 0
    while env.is_done is False:
        env.process()
        nb_process += 1
    assert nb_process == 3
    assert env.is_done
    assert 'Loop' in env.game.note

    # after thrashing process has no effect
    curr_phase = env.game.get_current_phase()
    env.process()
    assert curr_phase == env.game.get_current_phase()

def test_set_initial_state():
    """ Tests the SetInitialState wrapper """
    state_zlib = b'x\x9cmSMk\xdb@\x105,v\x92u\x8a\xc0\xf8\xd0\x8465J\xdb\x80\xa0\x1fQ/\xado+kc/Z\xed\x8aY\xc9B\xbd' \
                 b'\x08\x87\xdcR\nIz\xe8\xcf\xef\xcc\xca\xb2A\xf42\xa3\xdd}3\xef\xbd\x19\xc4\xcf\x1f\xff\xfez|\xfa' \
                 b'\xfe\xf4-\xde\xdd?\xcf&\xee\xf6\xc7\xd78\xbf<}\xf9\xb3\xfb\xfd\xb0{~\x08\xa7\xc6\xb6\xa9\x14' \
                 b'\xa9VF\x86|\x95\xb6i\x95\xe7J\xbap*t-\x1a\xd7\xd6B\x95\xe1\x99\xb3Z\x95B\x81\x0cO\xb1\xa0\x00' \
                 b'\xe9\\\xf8J\xad\x8d\x05\xd9J\x00\x0b.</l-\xa1]m\xacZ\xc9\xe8\x03\x1fc\x81nfo\xf8X,\xb6\xd2P*' \
                 b'\x1b\xe0\xe3\xbbEY\x19JF\x14\xd1g~\xb2\x96\x90\x0b\xd3\xcc\xae\xe9.\xed\x80Pm(m\xac\xa6\xcbD' \
                 b'\xfa\xb2L\xc9\xe8\x13\x9f\xdc\x810+Ip\xb1H*\xa0\xe4\nA\x80\xc2z\\\x02\x92R.\x80\xe0P9\xa7D' \
                 b'\xd7\xdd\xc9\xad\x07\xd8\x92\xaa\xd6Bw\\9\xa5\xdc\xba\xe8#\x9f\x94\x15dr\xaf:\xa9<`e\xbd' \
                 b'\xdcD\x0b:\t\x93E7\xfcDT\xae\x04\xec\xdb\xd9S\xd2\xcb\xe8t\n\x9dt\xd5)\x01\xa5YkaR\x02\xa2' \
                 b'\xe5z\xedud\xbe\x95\xa9\x1b:\xc9T\xc5W\x07\xe69gH\xc1\x19\xb12\x977\x9c\xa1\x8c\xf8\xdd' \
                 b'\x91qN7)gx\xe0\x8c\x98\x19\xf2\x12\xa0gB\x00\xb6\xe4LS\x07\xbd-8C\xa2\xf8\xfdq\xd2\x17\xd8' \
                 b'\x01\xa5\xb2\x8c\x8as\\\x06\xa3\xb13\x9cv\xfc\xb6_\x1b6\xc1\xfdp\x066G\x16z\xc6\xad\xc5\xd7' \
                 b'\x87yb\x0f\x9c\x18qo1\x94\x88\xac\x05\xb6\xc4a\x12h\xbf#"\x02\xe2\xa0\xa7\xc2\x07\\\x11' \
                 b'\xc3u-/\x0e\x86\x83\x81\xe1\xe5\xe5\xd1k0\xf0Jo\xbd\xcd``\x93\xdez\x87\xc1\xc0\xe1\xf2u' \
                 b'\xef+\x18\xf8Z^\x1d,\xcd\xffg\x89\x84\xee\xdd\x04\x037\xc0\x0f\xa5#\xfa\xde\xc3\xfc\xf7' \
                 b'\xde\xdb\x08\xa6G3\xfe\xd0\xab\xf7\x87^\xee\x08\xcez}\xa3\x9f7\x88_|\xc1\x9f\x040\xe2%F' \
                 b'\xf07X\x8a\x11Y0b\xe5?\x93\x8e\xc9\xa7'
    state_proto = zlib_to_proto(state_zlib, StateProto)
    env = gym.make('DiplomacyEnv-v0')
    env = SetInitialState(env, state_proto)
    env.reset()

    # Checking that the env has been reset
    assert env.game.get_current_phase() == 'S1902M'
    assert env.current_year == 2

def test_assign_players():
    """ Tests the AssignPlayers Wrapper """
    players = [DummyPlayer()] * 7
    powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
    env = gym.make('DiplomacyEnv-v0')
    env = AssignPlayers(env, players, powers)
    env.reset()
    assert env.players == players

    # Resetting 3 times
    env.reset()
    assign_powers_1 = env.get_all_powers_name()
    env.reset()
    assign_powers_2 = env.get_all_powers_name()
    env.reset()
    assign_powers_3 = env.get_all_powers_name()

    # Checking that they are correctly assigned
    assert assign_powers_1 == powers
    assert assign_powers_2 == powers
    assert assign_powers_3 == powers

def test_randomize_players():
    """ Tests the RandomizePlayers Wrapper """
    players = [DummyPlayer()] * 7
    env = gym.make('DiplomacyEnv-v0')
    env = RandomizePlayers(env, players)
    env.reset()
    assert env.players == players

    # Resetting 3 times
    env.reset()
    randomize_powers_1 = env.get_all_powers_name()
    env.reset()
    randomize_powers_2 = env.get_all_powers_name()
    env.reset()
    randomize_powers_3 = env.get_all_powers_name()

    # Checking that they are randomized
    assert randomize_powers_1 != randomize_powers_2
    assert randomize_powers_1 != randomize_powers_3
    assert randomize_powers_2 != randomize_powers_3

def test_set_player_seed():
    """ Tests the SetPlayerSeed Wrapper """
    players = []
    for _ in range(7):
        players += [DummyPlayer()]
    env = gym.make('DiplomacyEnv-v0')
    env = RandomizePlayers(env, players)
    env = SetPlayerSeed(env, players)
    env.reset()
    powers = env.get_all_powers_name()
    assert env.players == players

    # Making sure the player seed has been set
    for power_name, player in zip(powers, players):
        assert player.player_seed == get_player_seed(env.game_id, power_name)

def test_save_game():
    """ Tests the SaveGame Wrapper """
    env = gym.make('DiplomacyEnv-v0')
    env = SaveGame(env)
    env.reset()
    assert env.get_saved_game() is not None

    # Submitting orders, processing and retrieving saved game
    env.step(('FRANCE', ['A PAR H']))
    env.process()
    assert env.get_saved_game() is not None
    env.reset()
    assert env.get_saved_game() is not None
