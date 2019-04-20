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
""" Tests for State Space
    - Contains the tests suite for the diplomacy_research.models.state_space object
"""
import numpy as np
from diplomacy import Game
from diplomacy_research.models import state_space

def test_board_state():
    """ Tests the proto_to_state_space  """
    game = Game()
    game_map = game.map
    state_proto = state_space.extract_state_proto(game)
    new_game = state_space.build_game_from_state_proto(state_proto)

    # Retrieving board_state
    state_proto_2 = state_space.extract_state_proto(new_game)
    board_state_1 = state_space.proto_to_board_state(state_proto, game_map)
    board_state_2 = state_space.proto_to_board_state(state_proto_2, game_map)

    # Checking
    assert np.allclose(board_state_1, board_state_2)
    assert board_state_1.shape == (state_space.NB_NODES, state_space.NB_FEATURES)
    assert game.get_hash() == new_game.get_hash()

def test_adjacency_matrix():
    """ Tests the creation of the adjacency matrix """
    adj_matrix = state_space.get_adjacency_matrix()
    assert adj_matrix.shape == (state_space.NB_NODES, state_space.NB_NODES)

def test_vocabulary():
    """ Tests accessing and converting vocabulary """
    vocabulary = state_space.get_vocabulary()
    assert vocabulary
    for token in vocabulary:
        token_ix = state_space.token_to_ix(token)
        new_token = state_space.ix_to_token(token_ix)
        assert token == new_token

def test_power_vocabulary():
    """ Tests accessing and converting power vocabulary """
    power_vocabulary = state_space.get_power_vocabulary()
    assert power_vocabulary

def test_order_frequency():
    """ Tests order frequency """
    # Not testing valid order, because moves dataset might not have been generated.
    assert state_space.get_order_frequency('A ZZZ H', no_press_only=True) == 0
    assert state_space.get_order_frequency('A ZZZ H', no_press_only=False) == 0
