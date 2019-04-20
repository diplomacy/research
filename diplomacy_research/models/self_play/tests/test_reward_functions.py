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
""" Unit tests for reward functions. """
from diplomacy import Game
from diplomacy_research.models.gym.environment import DoneReason
from diplomacy_research.models.self_play.reward_functions import NbCentersReward, NormNbCentersReward, \
    IntNormNbCentersReward, CustomIntNbCentersReward, CustomIntUnitReward, PlusOneMinusOneReward, \
    DrawSizeReward, ProportionalReward, SumOfSquares, SurvivorWinReward
from diplomacy_research.models.state_space import extract_state_proto

def test_nb_centers_reward():
    """ Tests for NbCentersReward"""
    game = Game()
    rew_fn = NbCentersReward()
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'nb_centers_reward'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 3.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 3.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 3.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 3.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 3.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 4.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 3.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

def test_norm_centers_reward():
    """ Tests for NormNbCentersReward """
    game = Game()
    rew_fn = NormNbCentersReward()
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'norm_nb_centers_reward'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 3./18
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 3./18
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 3./18
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 3./18
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 3./18
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 4./18
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 3./18

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

def test_int_norm_centers_reward():
    """ Tests for InterimNormNbCentersReward """
    game = Game()
    rew_fn = IntNormNbCentersReward()

    # Removing one center from FRANCE and adding it to GERMANY
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name == 'FRANCE':
            power.centers.remove('PAR')
        if power.name == 'GERMANY':
            power.centers.append('PAR')
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'int_norm_nb_centers_reward'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == -1. / 18
    assert get_reward('GERMANY', False, None) == 1. / 18
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == -1. / 18
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 1. / 18
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -1.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -1.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -1.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -1.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -1.

def test_custom_int_nb_centers_reward():
    """ Tests for CustomInterimNbCentersReward """
    game = Game()
    rew_fn = CustomIntNbCentersReward()

    # Removing one center from FRANCE and adding it to GERMANY
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name == 'FRANCE':
            power.centers.remove('PAR')
        if power.name == 'GERMANY':
            power.centers.append('PAR')
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'custom_int_nb_centers_reward'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == -2.
    assert get_reward('GERMANY', False, None) == 1.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == -2.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -18.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -18.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -18.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -18.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -18.

    # Reversing
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name == 'FRANCE':
            power.centers.append('PAR')
        if power.name == 'GERMANY':
            power.centers.remove('PAR')
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 2.
    assert get_reward('GERMANY', False, None) == -1.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 2.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -18.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -18.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -18.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -18.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -18.

def test_custom_int_unit_reward():
    """ Tests for CustomInterimUnitReward """
    game = Game()
    rew_fn = CustomIntUnitReward()

    # Issuing orders
    prev_state_proto = extract_state_proto(game)
    game.set_orders('FRANCE', ['A MAR - SPA', 'A PAR - PIC'])
    game.set_orders('AUSTRIA', ['A VIE - TYR'])
    game.process()
    state_proto = extract_state_proto(game)
    assert game.get_current_phase() == 'F1901M'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # +1 for FRANCE for conquering SPA

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 1.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -18.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -18.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -18.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -18.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -18.

    # Issuing orders
    prev_state_proto = state_proto
    game.set_orders('FRANCE', ['A PIC - BEL', 'A SPA - POR'])
    game.set_orders('AUSTRIA', ['F TRI - VEN', 'A TYR S F TRI - VEN'])
    game.process()
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # +1 for FRANCE for conquering POR
    # -1 for FRANCE for losing SPA
    # +1 for FRANCE for conquering BEL
    # +1 for AUSTRIA for conquering VEN
    # -1 for ITALY for losing VEN

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 1.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 1.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == -1.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -18.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -18.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -18.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -18.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -18.

    # Issuing orders
    prev_state_proto = state_proto
    game.set_orders('FRANCE', ['A PIC - BEL', 'A SPA - POR'])
    game.set_orders('AUSTRIA', ['F TRI - VEN', 'A TYR S F TRI - VEN'])
    game.process()
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # +0 - No new SCs

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -18.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -18.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -18.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -18.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -18.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -18.

def test_plus_minus_one_reward():
    """ Tests for PlusOneMinusOneReward """
    game = Game()
    rew_fn = PlusOneMinusOneReward()
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'plus_one_minus_one_reward'
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 1.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -1.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -1.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -1.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -1.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -1.

    # --- Clearing supply centers
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name != 'FRANCE':
            power.clear_units()
            power.centers = []
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: rew_fn.get_reward(prev_state_proto,
                                                                                state_proto,
                                                                                power_name,
                                                                                is_terminal_state=is_terminal,
                                                                                done_reason=done_reason)

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == 1.
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == -1.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == -1.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == -1.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == -1.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == -1.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == -1.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == -1.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == -1.

def test_draw_size_reward():
    """ Test draw size reward fn """
    game = Game()
    pot_size = 20
    rew_fn = DrawSizeReward(pot_size=pot_size)
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'draw_size_reward'
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # --- Clearing supply centers
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name != 'FRANCE' and power.name != 'ITALY':
            power.clear_units()
            power.centers = []
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size / 2., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == round(pot_size / 2., 8)
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # Move centers in other countries to FRANCE except ENGLAND
    # Winner: FRANCE
    # Survivor: FRANCE, ENGLAND
    game = Game()
    prev_state_proto = extract_state_proto(game)
    game.clear_centers()
    game.set_centers('FRANCE', ['BUD', 'TRI', 'VIE', 'BRE', 'MAR', 'PAR', 'BER', 'KIE', 'MUN', 'NAP', 'ROM', 'VEN',
                                'MOS', 'SEV', 'STP', 'WAR', 'ANK', 'CON', 'SMY'])
    game.set_centers('ENGLAND', ['EDI', 'LON', 'LVP'])
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state -- Victory
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size, 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

def test_proportional_reward():
    """ Test sum of squares reward function """
    game = Game()
    pot_size = 20
    rew_fn = ProportionalReward(pot_size=pot_size, exponent=2)
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'proportional_reward'
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 16/70., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # --- Clearing supply centers
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name != 'FRANCE' and power.name != 'RUSSIA':
            power.clear_units()
            power.centers = []
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/25., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 16/25., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # Move centers in other countries to FRANCE except ENGLAND
    # Winner: FRANCE
    # Survivor: FRANCE, ENGLAND
    game = Game()
    prev_state_proto = extract_state_proto(game)
    game.clear_centers()
    game.set_centers('FRANCE', ['BUD', 'TRI', 'VIE', 'BRE', 'MAR', 'PAR', 'BER', 'KIE', 'MUN', 'NAP', 'ROM', 'VEN',
                                'MOS', 'SEV', 'STP', 'WAR', 'ANK', 'CON', 'SMY'])
    game.set_centers('ENGLAND', ['EDI', 'LON', 'LVP'])
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state -- Victory
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size, 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

def test_sum_of_squares_reward():
    """ Test sum of squares reward function """
    game = Game()
    pot_size = 20
    rew_fn = SumOfSquares(pot_size=pot_size)
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'sum_of_squares_reward'
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 16/70., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/70., 8)

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # --- Clearing supply centers
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name != 'FRANCE' and power.name != 'RUSSIA':
            power.clear_units()
            power.centers = []
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size * 9/25., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size * 16/25., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # Move centers in other countries to FRANCE except ENGLAND
    # Winner: FRANCE
    # Survivor: FRANCE, ENGLAND
    game = Game()
    prev_state_proto = extract_state_proto(game)
    game.clear_centers()
    game.set_centers('FRANCE', ['BUD', 'TRI', 'VIE', 'BRE', 'MAR', 'PAR', 'BER', 'KIE', 'MUN', 'NAP', 'ROM', 'VEN',
                                'MOS', 'SEV', 'STP', 'WAR', 'ANK', 'CON', 'SMY'])
    game.set_centers('ENGLAND', ['EDI', 'LON', 'LVP'])
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state -- Victory
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size, 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

def test_survivor_win_reward():
    """ Test survivor win reward function """
    game = Game()
    pot_size = 20
    rew_fn = SurvivorWinReward(pot_size=pot_size)
    prev_state_proto = extract_state_proto(game)
    state_proto = extract_state_proto(game)
    assert rew_fn.name == 'survivor_win_reward'
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- Not in terminal state
    assert get_reward('AUSTRIA', False, None) == 0.
    assert get_reward('ENGLAND', False, None) == 0.
    assert get_reward('FRANCE', False, None) == 0.
    assert get_reward('GERMANY', False, None) == 0.
    assert get_reward('ITALY', False, None) == 0.
    assert get_reward('RUSSIA', False, None) == 0.
    assert get_reward('TURKEY', False, None) == 0.

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == round(pot_size / 7., 8)

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # --- Clearing supply centers
    prev_state_proto = extract_state_proto(game)
    for power in game.powers.values():
        if power.name != 'FRANCE' and power.name != 'RUSSIA':
            power.clear_units()
            power.centers = []
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # --- In terminal state
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size / 2., 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == round(pot_size / 2., 8)
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.

    # Move centers in other countries to FRANCE except ENGLAND
    # Winner: FRANCE
    # Survivor: FRANCE, ENGLAND
    game = Game()
    prev_state_proto = extract_state_proto(game)
    game.clear_centers()
    game.set_centers('FRANCE', ['BUD', 'TRI', 'VIE', 'BRE', 'MAR', 'PAR', 'BER', 'KIE', 'MUN', 'NAP', 'ROM', 'VEN',
                                'MOS', 'SEV', 'STP', 'WAR', 'ANK', 'CON', 'SMY'])
    game.set_centers('ENGLAND', ['EDI', 'LON', 'LVP'])
    state_proto = extract_state_proto(game)
    get_reward = lambda power_name, is_terminal, done_reason: round(rew_fn.get_reward(prev_state_proto,
                                                                                      state_proto,
                                                                                      power_name,
                                                                                      is_terminal_state=is_terminal,
                                                                                      done_reason=done_reason), 8)

    # France has 19 SC, 18 to win, 1 excess
    # Nb of controlled SC is 19 + 3 - 1 excess = 21
    # Reward for FRANCE is 18 / 21 * pot
    # Reward for ENGLAND is 3 / 21 * pot

    # --- In terminal state -- Victory
    assert get_reward('AUSTRIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ENGLAND', True, DoneReason.GAME_ENGINE) == round(pot_size * 3./21, 8)
    assert get_reward('FRANCE', True, DoneReason.GAME_ENGINE) == round(pot_size * 18./21, 8)
    assert get_reward('GERMANY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('ITALY', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('RUSSIA', True, DoneReason.GAME_ENGINE) == 0.
    assert get_reward('TURKEY', True, DoneReason.GAME_ENGINE) == 0.

    # --- Thrashing
    assert get_reward('AUSTRIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('ENGLAND', True, DoneReason.THRASHED) == 0.
    assert get_reward('FRANCE', True, DoneReason.THRASHED) == 0.
    assert get_reward('GERMANY', True, DoneReason.THRASHED) == 0.
    assert get_reward('ITALY', True, DoneReason.THRASHED) == 0.
    assert get_reward('RUSSIA', True, DoneReason.THRASHED) == 0.
    assert get_reward('TURKEY', True, DoneReason.THRASHED) == 0.
