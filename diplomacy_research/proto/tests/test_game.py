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
""" Tests the game.proto """
import numpy as np
from diplomacy import Game
from diplomacy_research.proto.diplomacy_proto.common_pb2 import MapStringList
from diplomacy_research.proto.diplomacy_proto.game_pb2 import PhaseHistory, State, SavedGame
from diplomacy_research.models.state_space import token_to_ix, get_order_tokens, TOKENS_PER_ORDER
from diplomacy_research.utils.proto import dict_to_proto, proto_to_dict


def test_phase_history():
    """ Tests the game_pb2.PhaseHistory proto """
    # pylint: disable=too-many-statements
    policy_1 = {'locs': ['PAR', 'MAR'],
                'tokens': [token_to_ix(order_token) for order_token in
                           get_order_tokens('A PAR H <EOS> <PAD> <PAD> <PAD> A MAR - PIE <EOS> <PAD> <PAD>')],
                'log_probs': np.random.rand(2 * TOKENS_PER_ORDER).tolist(),
                'draw_action': True,
                'draw_prob': 0.123}
    policy_2 = {'locs': ['xPAR', 'xMAR'],
                'tokens': [token_to_ix(order_token) for order_token in
                           get_order_tokens('A PAR H <EOS> <PAD> <PAD> <PAD> A MAR - PIE <EOS> <PAD> <PAD>')],
                'log_probs': np.random.rand(2 * TOKENS_PER_ORDER).tolist(),
                'draw_action': False,
                'draw_prob': 0.456}
    possible_orders = Game().get_all_possible_orders()

    phase_history_dict = {
        'name': 'S1902M',
        'state': {
            'game_id': '12345',
            'name': 'S1902M',
            'map': 'standard',
            'zobrist_hash': '1234567890',
            'note': 'Note123',
            'rules': ['R2', 'R1'],
            'units': {'AUSTRIA': ['A PAR', 'A MAR'], 'FRANCE': ['F AAA', 'F BBB']},
            'centers': {'AUSTRIA': ['PAR', 'MAR'], 'FRANCE': ['AAA', 'BBB'], 'ENGLAND': []},
            'homes': {'AUSTRIA': ['PAR1', 'MAR1'], 'FRANCE': ['AAA1', 'BBB1']},
            'builds': {'AUSTRIA': {'count': 0, 'homes': []},
                       'FRANCE': {'count': -1, 'homes': []},
                       'RUSSIA': {'count': 2, 'homes': ['PAR', 'MAR']}},
            'board_state': [10, 11, 12, 13, 14, 15, 16, 17]},
        'orders': {'FRANCE': ['A PAR H', 'A MAR - BUR'],
                   'ENGLAND': []},
        'results': {'A PAR': [''], 'A MAR': ['bounce']},

        'policy': {'FRANCE': policy_1,
                   'ENGLAND': policy_2},
        'prev_orders_state': [1, 2, 3, 4, 5, 6, 7, 8],
        'state_value': {'FRANCE': 2.56, 'ENGLAND': 6.78},
        'possible_orders': possible_orders
    }
    phase_history_proto = dict_to_proto(phase_history_dict, PhaseHistory)
    new_phase_history_dict = proto_to_dict(phase_history_proto)

    # Validating proto
    assert phase_history_proto.name == 'S1902M'
    assert phase_history_proto.state.game_id == '12345'
    assert phase_history_proto.state.name == 'S1902M'
    assert phase_history_proto.state.map == 'standard'
    assert phase_history_proto.state.zobrist_hash == '1234567890'
    assert phase_history_proto.state.note == 'Note123'
    assert phase_history_proto.state.rules == ['R2', 'R1']
    assert phase_history_proto.state.units['AUSTRIA'].value == ['A PAR', 'A MAR']
    assert phase_history_proto.state.units['FRANCE'].value == ['F AAA', 'F BBB']
    assert phase_history_proto.state.centers['AUSTRIA'].value == ['PAR', 'MAR']
    assert phase_history_proto.state.centers['FRANCE'].value == ['AAA', 'BBB']
    assert phase_history_proto.state.centers['ENGLAND'].value == []
    assert phase_history_proto.state.homes['AUSTRIA'].value == ['PAR1', 'MAR1']
    assert phase_history_proto.state.homes['FRANCE'].value == ['AAA1', 'BBB1']
    assert phase_history_proto.state.builds['AUSTRIA'].count == 0
    assert phase_history_proto.state.builds['AUSTRIA'].homes == []
    assert phase_history_proto.state.builds['FRANCE'].count == -1
    assert phase_history_proto.state.builds['FRANCE'].homes == []
    assert phase_history_proto.state.builds['RUSSIA'].count == 2
    assert phase_history_proto.state.builds['RUSSIA'].homes == ['PAR', 'MAR']
    assert phase_history_proto.state.board_state == [10, 11, 12, 13, 14, 15, 16, 17]
    assert phase_history_proto.orders['FRANCE'].value == ['A PAR H', 'A MAR - BUR']
    assert phase_history_proto.orders['ENGLAND'].value == []
    assert phase_history_proto.results['A PAR'].value == ['']
    assert phase_history_proto.results['A MAR'].value == ['bounce']
    assert phase_history_proto.policy['FRANCE'].locs == policy_1['locs']
    assert phase_history_proto.policy['FRANCE'].tokens == policy_1['tokens']
    assert np.allclose(phase_history_proto.policy['FRANCE'].log_probs, policy_1['log_probs'])
    assert phase_history_proto.policy['FRANCE'].draw_action == policy_1['draw_action']
    assert np.allclose(phase_history_proto.policy['FRANCE'].draw_prob, policy_1['draw_prob'])
    assert phase_history_proto.policy['ENGLAND'].locs == policy_2['locs']
    assert phase_history_proto.policy['ENGLAND'].tokens == policy_2['tokens']
    assert np.allclose(phase_history_proto.policy['ENGLAND'].log_probs, policy_2['log_probs'])
    assert phase_history_proto.policy['ENGLAND'].draw_action == policy_2['draw_action']
    assert np.allclose(phase_history_proto.policy['ENGLAND'].draw_prob, policy_2['draw_prob'])
    assert phase_history_proto.prev_orders_state == [1, 2, 3, 4, 5, 6, 7, 8]
    assert np.allclose(phase_history_proto.state_value['FRANCE'], 2.56)
    assert np.allclose(phase_history_proto.state_value['ENGLAND'], 6.78)

    for loc in possible_orders:
        assert possible_orders[loc] == phase_history_proto.possible_orders[loc].value

    # Validating resulting dict
    assert new_phase_history_dict['name'] == 'S1902M'
    assert new_phase_history_dict['state']['game_id'] == '12345'
    assert new_phase_history_dict['state']['name'] == 'S1902M'
    assert new_phase_history_dict['state']['map'] == 'standard'
    assert new_phase_history_dict['state']['zobrist_hash'] == '1234567890'
    assert new_phase_history_dict['state']['note'] == 'Note123'
    assert new_phase_history_dict['state']['rules'] == ['R2', 'R1']
    assert new_phase_history_dict['state']['units']['AUSTRIA'] == ['A PAR', 'A MAR']
    assert new_phase_history_dict['state']['units']['FRANCE'] == ['F AAA', 'F BBB']
    assert new_phase_history_dict['state']['centers']['AUSTRIA'] == ['PAR', 'MAR']
    assert new_phase_history_dict['state']['centers']['FRANCE'] == ['AAA', 'BBB']
    assert new_phase_history_dict['state']['centers']['ENGLAND'] == []
    assert new_phase_history_dict['state']['homes']['AUSTRIA'] == ['PAR1', 'MAR1']
    assert new_phase_history_dict['state']['homes']['FRANCE'] == ['AAA1', 'BBB1']
    assert new_phase_history_dict['state']['builds']['AUSTRIA']['count'] == 0
    assert new_phase_history_dict['state']['builds']['AUSTRIA']['homes'] == []
    assert new_phase_history_dict['state']['builds']['FRANCE']['count'] == -1
    assert new_phase_history_dict['state']['builds']['FRANCE']['homes'] == []
    assert new_phase_history_dict['state']['builds']['RUSSIA']['count'] == 2
    assert new_phase_history_dict['state']['builds']['RUSSIA']['homes'] == ['PAR', 'MAR']
    assert new_phase_history_dict['state']['board_state'] == [10, 11, 12, 13, 14, 15, 16, 17]
    assert new_phase_history_dict['orders']['FRANCE'] == ['A PAR H', 'A MAR - BUR']
    assert new_phase_history_dict['orders']['ENGLAND'] == []
    assert new_phase_history_dict['results']['A PAR'] == ['']
    assert new_phase_history_dict['results']['A MAR'] == ['bounce']
    assert new_phase_history_dict['policy']['FRANCE']['locs'] == policy_1['locs']
    assert new_phase_history_dict['policy']['FRANCE']['tokens'] == policy_1['tokens']
    assert np.allclose(new_phase_history_dict['policy']['FRANCE']['log_probs'], policy_1['log_probs'])
    assert new_phase_history_dict['policy']['FRANCE']['draw_action'] == policy_1['draw_action']
    assert np.allclose(new_phase_history_dict['policy']['FRANCE']['draw_prob'], policy_1['draw_prob'])
    assert new_phase_history_dict['policy']['ENGLAND']['locs'] == policy_2['locs']
    assert new_phase_history_dict['policy']['ENGLAND']['tokens'] == policy_2['tokens']
    assert np.allclose(new_phase_history_dict['policy']['ENGLAND']['log_probs'], policy_2['log_probs'])
    assert new_phase_history_dict['policy']['ENGLAND']['draw_action'] == policy_2['draw_action']
    assert np.allclose(new_phase_history_dict['policy']['ENGLAND']['draw_prob'], policy_2['draw_prob'])
    assert new_phase_history_dict['prev_orders_state'] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert np.allclose(new_phase_history_dict['state_value']['FRANCE'], 2.56)
    assert np.allclose(new_phase_history_dict['state_value']['ENGLAND'], 6.78)

def test_possible_orders():
    """ Tests the game_pb2.PossibleOrders proto """
    game = Game()
    possible_orders = game.get_all_possible_orders()
    possible_orders_proto = dict_to_proto(possible_orders, MapStringList)
    new_possible_orders_dict = proto_to_dict(possible_orders_proto)

    # Validating proto
    for loc in possible_orders:
        assert possible_orders[loc] == possible_orders_proto[loc].value

    # Validating resulting dict
    for loc in possible_orders:
        assert possible_orders[loc] == new_possible_orders_dict[loc]

def test_state():
    """ Tests the game_pb2.State proto """
    state_dict = {'game_id': '12345',
                  'name': 'S1901M',
                  'map': 'standard',
                  'zobrist_hash': '1234567890',
                  'note': 'Note123',
                  'rules': ['R2', 'R1'],
                  'units': {'AUSTRIA': ['A PAR', 'A MAR'], 'FRANCE': ['F AAA', 'F BBB']},
                  'centers': {'AUSTRIA': ['PAR', 'MAR'], 'FRANCE': ['AAA', 'BBB'], 'ENGLAND': []},
                  'influence': {'AUSTRIA': ['x1', 'x2']},
                  'civil_disorder': {'FRANCE': 0, 'ENGLAND': 1},
                  'homes': {'AUSTRIA': ['PAR1', 'MAR1'], 'FRANCE': ['AAA1', 'BBB1']},
                  'builds': {'AUSTRIA': {'count': 0, 'homes': []},
                             'FRANCE': {'count': -1, 'homes': []},
                             'RUSSIA': {'count': 2, 'homes': ['PAR', 'MAR']}},
                  'board_state': [1, 2, 3, 4, 5, 6, 7, 8]}
    state_proto = dict_to_proto(state_dict, State)
    new_state_dict = proto_to_dict(state_proto)

    # Validating proto
    assert state_proto.game_id == '12345'
    assert state_proto.name == 'S1901M'
    assert state_proto.map == 'standard'
    assert state_proto.zobrist_hash == '1234567890'
    assert state_proto.note == 'Note123'
    assert state_proto.rules == ['R2', 'R1']
    assert state_proto.units['AUSTRIA'].value == ['A PAR', 'A MAR']
    assert state_proto.units['FRANCE'].value == ['F AAA', 'F BBB']
    assert state_proto.centers['AUSTRIA'].value == ['PAR', 'MAR']
    assert state_proto.centers['FRANCE'].value == ['AAA', 'BBB']
    assert state_proto.centers['ENGLAND'].value == []
    assert state_proto.influence['AUSTRIA'].value == ['x1', 'x2']
    assert state_proto.civil_disorder['FRANCE'] == 0
    assert state_proto.civil_disorder['ENGLAND'] == 1
    assert state_proto.homes['AUSTRIA'].value == ['PAR1', 'MAR1']
    assert state_proto.homes['FRANCE'].value == ['AAA1', 'BBB1']
    assert state_proto.builds['AUSTRIA'].count == 0
    assert state_proto.builds['AUSTRIA'].homes == []
    assert state_proto.builds['FRANCE'].count == -1
    assert state_proto.builds['FRANCE'].homes == []
    assert state_proto.builds['RUSSIA'].count == 2
    assert state_proto.builds['RUSSIA'].homes == ['PAR', 'MAR']
    assert state_proto.board_state == [1, 2, 3, 4, 5, 6, 7, 8]

    # Validating resulting dict
    assert new_state_dict['game_id'] == '12345'
    assert new_state_dict['name'] == 'S1901M'
    assert new_state_dict['map'] == 'standard'
    assert new_state_dict['zobrist_hash'] == '1234567890'
    assert new_state_dict['note'] == 'Note123'
    assert new_state_dict['rules'] == ['R2', 'R1']
    assert new_state_dict['units']['AUSTRIA'] == ['A PAR', 'A MAR']
    assert new_state_dict['units']['FRANCE'] == ['F AAA', 'F BBB']
    assert new_state_dict['centers']['AUSTRIA'] == ['PAR', 'MAR']
    assert new_state_dict['centers']['FRANCE'] == ['AAA', 'BBB']
    assert new_state_dict['centers']['ENGLAND'] == []
    assert new_state_dict['influence']['AUSTRIA'] == ['x1', 'x2']
    assert new_state_dict['civil_disorder']['FRANCE'] == 0
    assert new_state_dict['civil_disorder']['ENGLAND'] == 1
    assert new_state_dict['homes']['AUSTRIA'] == ['PAR1', 'MAR1']
    assert new_state_dict['homes']['FRANCE'] == ['AAA1', 'BBB1']
    assert new_state_dict['builds']['AUSTRIA']['count'] == 0
    assert new_state_dict['builds']['AUSTRIA']['homes'] == []
    assert new_state_dict['builds']['FRANCE']['count'] == -1
    assert new_state_dict['builds']['FRANCE']['homes'] == []
    assert new_state_dict['builds']['RUSSIA']['count'] == 2
    assert new_state_dict['builds']['RUSSIA']['homes'] == ['PAR', 'MAR']
    assert new_state_dict['board_state'] == [1, 2, 3, 4, 5, 6, 7, 8]

def test_saved_game():
    """ Tests the game_pb2.State proto """
    # pylint: disable=too-many-statements
    game = Game()
    state_1 = {'game_id': '12345',
               'name': 'S1901M',
               'map': 'standard',
               'zobrist_hash': '1234567890',
               'note': 'Note123',
               'rules': ['R2', 'R1'],
               'units': {'AUSTRIA': ['A PAR', 'A MAR'], 'FRANCE': ['F AAA', 'F BBB']},
               'centers': {'AUSTRIA': ['PAR', 'MAR'], 'FRANCE': ['AAA', 'BBB'], 'ENGLAND': []},
               'homes': {'AUSTRIA': ['PAR1', 'MAR1'], 'FRANCE': ['AAA1', 'BBB1']},
               'builds': {'AUSTRIA': {'count': 0, 'homes': []},
                          'FRANCE': {'count': -1, 'homes': []},
                          'RUSSIA': {'count': 2, 'homes': ['PAR', 'MAR']}},
               'board_state': [1, 2, 3, 4, 5, 6, 7, 8]}
    state_2 = {'game_id': 'x12345',
               'name': 'xS1901M',
               'map': 'xstandard',
               'zobrist_hash': '0987654321',
               'note': 'xNote123',
               'rules': ['xR2', 'xR1'],
               'units': {'AUSTRIA': ['xA PAR', 'xA MAR'], 'FRANCE': ['xF AAA', 'xF BBB']},
               'centers': {'AUSTRIA': ['xPAR', 'xMAR'], 'FRANCE': ['xAAA', 'xBBB'], 'ENGLAND': []},
               'homes': {'AUSTRIA': ['xPAR1', 'xMAR1'], 'FRANCE': ['xAAA1', 'xBBB1']},
               'builds': {'AUSTRIA': {'count': 0, 'homes': []},
                          'FRANCE': {'count': -1, 'homes': []},
                          'RUSSIA': {'count': 2, 'homes': ['xPAR', 'xMAR']}}}
    policy_1 = {'locs': ['PAR', 'MAR'],
                'tokens': [token_to_ix(order_token) for order_token in
                           get_order_tokens('A PAR H <EOS> <PAD> <PAD> <PAD> A MAR - PIE <EOS> <PAD> <PAD>')],
                'log_probs': np.random.rand(2 * TOKENS_PER_ORDER).tolist(),
                'draw_action': True,
                'draw_prob': 0.123}
    policy_2 = {'locs': ['xPAR', 'xMAR'],
                'tokens': [token_to_ix(order_token) for order_token in
                           get_order_tokens('A PAR H <EOS> <PAD> <PAD> <PAD> A MAR - PIE <EOS> <PAD> <PAD>')],
                'log_probs': np.random.rand(2 * TOKENS_PER_ORDER).tolist(),
                'draw_action': False,
                'draw_prob': 0.456}
    possible_orders = game.get_all_possible_orders()

    saved_game = {'id': '12345',
                  'map': 'standard',
                  'rules': ['R2', 'R1'],
                  'phases': [{'name': 'S1901M',
                              'state': state_1,
                              'orders': {'FRANCE': ['A PAR H', 'A MAR - PIE'],
                                         'ENGLAND': []},
                              'results': {},
                              'policy': {'FRANCE': policy_1, 'ENGLAND': policy_2},
                              'prev_orders_state': [10, 11, 12, 13, 14, 15],
                              'state_value': {'FRANCE': 5.67, 'ENGLAND': 6.78},
                              'possible_orders': possible_orders},

                             {'name': 'xS1901M',
                              'state': state_2,
                              'orders': {'FRANCE': ['xA PAR H', 'xA MAR - PIE'],
                                         'ENGLAND': []},
                              'results': {},
                              'policy': {'FRANCE': policy_2, 'ENGLAND': None}}],
                  'done_reason': 'auto_draw',

                  'assigned_powers': ['FRANCE', 'ENGLAND'],
                  'players': ['rule', 'v0'],
                  'kwargs': {'FRANCE': {'player_seed': 1, 'noise': 0.2, 'temperature': 0.3, 'dropout_rate': 0.4},
                             'ENGLAND': {'player_seed': 6, 'noise': 0.7, 'temperature': 0.8, 'dropout_rate': 0.9}},

                  'is_partial_game': True,
                  'start_phase_ix': 2,

                  'reward_fn': 'default_reward',
                  'rewards': {'FRANCE': [1.1, 2.2, 3.3],
                              'ENGLAND': [1.2, 2.3, 3.4]},
                  'returns': {'FRANCE': [11.1, 12.2, 13.3],
                              'ENGLAND': [11.2, 12.3, 13.4]}}

    # Converting to proto and back
    saved_game_proto = dict_to_proto(saved_game, SavedGame)
    new_saved_game = proto_to_dict(saved_game_proto)

    # --- Validating proto ---
    assert saved_game_proto.id == '12345'
    assert saved_game_proto.map == 'standard'
    assert saved_game_proto.rules == ['R2', 'R1']

    assert saved_game_proto.phases[0].name == 'S1901M'
    assert saved_game_proto.phases[0].state.game_id == '12345'
    assert saved_game_proto.phases[0].state.name == 'S1901M'
    assert saved_game_proto.phases[0].state.map == 'standard'
    assert saved_game_proto.phases[0].state.zobrist_hash == '1234567890'
    assert saved_game_proto.phases[0].state.note == 'Note123'
    assert saved_game_proto.phases[0].state.rules == ['R2', 'R1']
    assert saved_game_proto.phases[0].state.units['AUSTRIA'].value == ['A PAR', 'A MAR']
    assert saved_game_proto.phases[0].state.units['FRANCE'].value == ['F AAA', 'F BBB']
    assert saved_game_proto.phases[0].state.centers['AUSTRIA'].value == ['PAR', 'MAR']
    assert saved_game_proto.phases[0].state.centers['FRANCE'].value == ['AAA', 'BBB']
    assert saved_game_proto.phases[0].state.centers['ENGLAND'].value == []
    assert saved_game_proto.phases[0].state.homes['AUSTRIA'].value == ['PAR1', 'MAR1']
    assert saved_game_proto.phases[0].state.homes['FRANCE'].value == ['AAA1', 'BBB1']
    assert saved_game_proto.phases[0].state.builds['AUSTRIA'].count == 0
    assert saved_game_proto.phases[0].state.builds['AUSTRIA'].homes == []
    assert saved_game_proto.phases[0].state.builds['FRANCE'].count == -1
    assert saved_game_proto.phases[0].state.builds['FRANCE'].homes == []
    assert saved_game_proto.phases[0].state.builds['RUSSIA'].count == 2
    assert saved_game_proto.phases[0].state.builds['RUSSIA'].homes == ['PAR', 'MAR']
    assert saved_game_proto.phases[0].state.board_state == [1, 2, 3, 4, 5, 6, 7, 8]

    assert saved_game_proto.phases[0].orders['FRANCE'].value == ['A PAR H', 'A MAR - PIE']
    assert saved_game_proto.phases[0].orders['ENGLAND'].value == []

    assert saved_game_proto.phases[0].policy['FRANCE'].locs == policy_1['locs']
    assert saved_game_proto.phases[0].policy['FRANCE'].tokens == policy_1['tokens']
    assert np.allclose(saved_game_proto.phases[0].policy['FRANCE'].log_probs, policy_1['log_probs'])
    assert saved_game_proto.phases[0].policy['FRANCE'].draw_action == policy_1['draw_action']
    assert np.allclose(saved_game_proto.phases[0].policy['FRANCE'].draw_prob, policy_1['draw_prob'])
    assert saved_game_proto.phases[0].policy['ENGLAND'].locs == policy_2['locs']
    assert saved_game_proto.phases[0].policy['ENGLAND'].tokens == policy_2['tokens']
    assert np.allclose(saved_game_proto.phases[0].policy['ENGLAND'].log_probs, policy_2['log_probs'])
    assert saved_game_proto.phases[0].policy['ENGLAND'].draw_action == policy_2['draw_action']
    assert np.allclose(saved_game_proto.phases[0].policy['ENGLAND'].draw_prob, policy_2['draw_prob'])

    assert saved_game_proto.phases[0].prev_orders_state == [10, 11, 12, 13, 14, 15]
    assert np.allclose(saved_game_proto.phases[0].state_value['FRANCE'], 5.67)
    assert np.allclose(saved_game_proto.phases[0].state_value['ENGLAND'], 6.78)
    for loc in possible_orders:
        assert possible_orders[loc] == saved_game_proto.phases[0].possible_orders[loc].value

    assert saved_game_proto.phases[1].name == 'xS1901M'
    assert saved_game_proto.phases[1].state.game_id == 'x12345'
    assert saved_game_proto.phases[1].state.name == 'xS1901M'
    assert saved_game_proto.phases[1].state.map == 'xstandard'
    assert saved_game_proto.phases[1].state.zobrist_hash == '0987654321'
    assert saved_game_proto.phases[1].state.note == 'xNote123'
    assert saved_game_proto.phases[1].state.rules == ['xR2', 'xR1']
    assert saved_game_proto.phases[1].state.units['AUSTRIA'].value == ['xA PAR', 'xA MAR']
    assert saved_game_proto.phases[1].state.units['FRANCE'].value == ['xF AAA', 'xF BBB']
    assert saved_game_proto.phases[1].state.centers['AUSTRIA'].value == ['xPAR', 'xMAR']
    assert saved_game_proto.phases[1].state.centers['FRANCE'].value == ['xAAA', 'xBBB']
    assert saved_game_proto.phases[1].state.centers['ENGLAND'].value == []
    assert saved_game_proto.phases[1].state.homes['AUSTRIA'].value == ['xPAR1', 'xMAR1']
    assert saved_game_proto.phases[1].state.homes['FRANCE'].value == ['xAAA1', 'xBBB1']
    assert saved_game_proto.phases[1].state.builds['AUSTRIA'].count == 0
    assert saved_game_proto.phases[1].state.builds['AUSTRIA'].homes == []
    assert saved_game_proto.phases[1].state.builds['FRANCE'].count == -1
    assert saved_game_proto.phases[1].state.builds['FRANCE'].homes == []
    assert saved_game_proto.phases[1].state.builds['RUSSIA'].count == 2
    assert saved_game_proto.phases[1].state.builds['RUSSIA'].homes == ['xPAR', 'xMAR']
    assert saved_game_proto.phases[1].state.board_state == []

    assert saved_game_proto.phases[1].orders['FRANCE'].value == ['xA PAR H', 'xA MAR - PIE']
    assert saved_game_proto.phases[1].orders['ENGLAND'].value == []

    assert saved_game_proto.phases[1].policy['FRANCE'].locs == policy_2['locs']
    assert saved_game_proto.phases[1].policy['FRANCE'].tokens == policy_2['tokens']
    assert np.allclose(saved_game_proto.phases[1].policy['FRANCE'].log_probs, policy_2['log_probs'])
    assert saved_game_proto.phases[1].policy['FRANCE'].draw_action == policy_2['draw_action']
    assert np.allclose(saved_game_proto.phases[1].policy['FRANCE'].draw_prob, policy_2['draw_prob'])
    assert saved_game_proto.phases[1].policy['ENGLAND'].locs == []
    assert saved_game_proto.phases[1].policy['ENGLAND'].tokens == []
    assert saved_game_proto.phases[1].policy['ENGLAND'].log_probs == []
    assert not saved_game_proto.phases[1].policy['ENGLAND'].draw_action
    assert saved_game_proto.phases[1].policy['ENGLAND'].draw_prob == 0.

    assert saved_game_proto.phases[1].prev_orders_state == []
    assert saved_game_proto.phases[1].state_value['FRANCE'] == 0.
    assert saved_game_proto.phases[1].state_value['ENGLAND'] == 0.
    assert not saved_game_proto.phases[1].possible_orders

    assert saved_game_proto.done_reason == 'auto_draw'
    assert saved_game_proto.assigned_powers == ['FRANCE', 'ENGLAND']
    assert saved_game_proto.players == ['rule', 'v0']
    assert saved_game_proto.kwargs['FRANCE'].player_seed == 1
    assert np.allclose(saved_game_proto.kwargs['FRANCE'].noise, 0.2)
    assert np.allclose(saved_game_proto.kwargs['FRANCE'].temperature, 0.3)
    assert np.allclose(saved_game_proto.kwargs['FRANCE'].dropout_rate, 0.4)
    assert saved_game_proto.kwargs['ENGLAND'].player_seed == 6
    assert np.allclose(saved_game_proto.kwargs['ENGLAND'].noise, 0.7)
    assert np.allclose(saved_game_proto.kwargs['ENGLAND'].temperature, 0.8)
    assert np.allclose(saved_game_proto.kwargs['ENGLAND'].dropout_rate, 0.9)

    assert saved_game_proto.is_partial_game
    assert saved_game_proto.start_phase_ix == 2
    assert saved_game_proto.reward_fn == 'default_reward'
    assert np.allclose(saved_game_proto.rewards['FRANCE'].value, [1.1, 2.2, 3.3])
    assert np.allclose(saved_game_proto.rewards['ENGLAND'].value, [1.2, 2.3, 3.4])
    assert np.allclose(saved_game_proto.returns['FRANCE'].value, [11.1, 12.2, 13.3])
    assert np.allclose(saved_game_proto.returns['ENGLAND'].value, [11.2, 12.3, 13.4])

    # --- Validating resulting dict ---
    assert new_saved_game['id'] == '12345'
    assert new_saved_game['map'] == 'standard'
    assert new_saved_game['rules'] == ['R2', 'R1']

    assert new_saved_game['phases'][0]['name'] == 'S1901M'
    assert new_saved_game['phases'][0]['state']['game_id'] == '12345'
    assert new_saved_game['phases'][0]['state']['name'] == 'S1901M'
    assert new_saved_game['phases'][0]['state']['map'] == 'standard'
    assert new_saved_game['phases'][0]['state']['zobrist_hash'] == '1234567890'
    assert new_saved_game['phases'][0]['state']['note'] == 'Note123'
    assert new_saved_game['phases'][0]['state']['rules'] == ['R2', 'R1']
    assert new_saved_game['phases'][0]['state']['units']['AUSTRIA'] == ['A PAR', 'A MAR']
    assert new_saved_game['phases'][0]['state']['units']['FRANCE'] == ['F AAA', 'F BBB']
    assert new_saved_game['phases'][0]['state']['centers']['AUSTRIA'] == ['PAR', 'MAR']
    assert new_saved_game['phases'][0]['state']['centers']['FRANCE'] == ['AAA', 'BBB']
    assert new_saved_game['phases'][0]['state']['centers']['ENGLAND'] == []
    assert new_saved_game['phases'][0]['state']['homes']['AUSTRIA'] == ['PAR1', 'MAR1']
    assert new_saved_game['phases'][0]['state']['homes']['FRANCE'] == ['AAA1', 'BBB1']
    assert new_saved_game['phases'][0]['state']['builds']['AUSTRIA']['count'] == 0
    assert new_saved_game['phases'][0]['state']['builds']['AUSTRIA']['homes'] == []
    assert new_saved_game['phases'][0]['state']['builds']['FRANCE']['count'] == -1
    assert new_saved_game['phases'][0]['state']['builds']['FRANCE']['homes'] == []
    assert new_saved_game['phases'][0]['state']['builds']['RUSSIA']['count'] == 2
    assert new_saved_game['phases'][0]['state']['builds']['RUSSIA']['homes'] == ['PAR', 'MAR']
    assert new_saved_game['phases'][0]['state']['board_state'] == [1, 2, 3, 4, 5, 6, 7, 8]

    assert new_saved_game['phases'][0]['orders']['FRANCE'] == ['A PAR H', 'A MAR - PIE']
    assert new_saved_game['phases'][0]['orders']['ENGLAND'] == []

    assert new_saved_game['phases'][0]['policy']['FRANCE']['locs'] == policy_1['locs']
    assert new_saved_game['phases'][0]['policy']['FRANCE']['tokens'] == policy_1['tokens']
    assert np.allclose(new_saved_game['phases'][0]['policy']['FRANCE']['log_probs'], policy_1['log_probs'])
    assert new_saved_game['phases'][0]['policy']['FRANCE']['draw_action'] == policy_1['draw_action']
    assert np.allclose(new_saved_game['phases'][0]['policy']['FRANCE']['draw_prob'], policy_1['draw_prob'])
    assert new_saved_game['phases'][0]['policy']['ENGLAND']['locs'] == policy_2['locs']
    assert new_saved_game['phases'][0]['policy']['ENGLAND']['tokens'] == policy_2['tokens']
    assert np.allclose(new_saved_game['phases'][0]['policy']['ENGLAND']['log_probs'], policy_2['log_probs'])
    assert new_saved_game['phases'][0]['policy']['ENGLAND']['draw_action'] == policy_2['draw_action']
    assert np.allclose(new_saved_game['phases'][0]['policy']['ENGLAND']['draw_prob'], policy_2['draw_prob'])

    assert new_saved_game['phases'][0]['prev_orders_state'] == [10, 11, 12, 13, 14, 15]
    assert np.allclose(new_saved_game['phases'][0]['state_value']['FRANCE'], 5.67)
    assert np.allclose(new_saved_game['phases'][0]['state_value']['ENGLAND'], 6.78)
    for loc in possible_orders:
        assert possible_orders[loc] == new_saved_game['phases'][0]['possible_orders'][loc]

    assert new_saved_game['phases'][1]['name'] == 'xS1901M'
    assert new_saved_game['phases'][1]['state']['game_id'] == 'x12345'
    assert new_saved_game['phases'][1]['state']['name'] == 'xS1901M'
    assert new_saved_game['phases'][1]['state']['map'] == 'xstandard'
    assert new_saved_game['phases'][1]['state']['zobrist_hash'] == '0987654321'
    assert new_saved_game['phases'][1]['state']['note'] == 'xNote123'
    assert new_saved_game['phases'][1]['state']['rules'] == ['xR2', 'xR1']
    assert new_saved_game['phases'][1]['state']['units']['AUSTRIA'] == ['xA PAR', 'xA MAR']
    assert new_saved_game['phases'][1]['state']['units']['FRANCE'] == ['xF AAA', 'xF BBB']
    assert new_saved_game['phases'][1]['state']['centers']['AUSTRIA'] == ['xPAR', 'xMAR']
    assert new_saved_game['phases'][1]['state']['centers']['FRANCE'] == ['xAAA', 'xBBB']
    assert new_saved_game['phases'][1]['state']['centers']['ENGLAND'] == []
    assert new_saved_game['phases'][1]['state']['homes']['AUSTRIA'] == ['xPAR1', 'xMAR1']
    assert new_saved_game['phases'][1]['state']['homes']['FRANCE'] == ['xAAA1', 'xBBB1']
    assert new_saved_game['phases'][1]['state']['builds']['AUSTRIA']['count'] == 0
    assert new_saved_game['phases'][1]['state']['builds']['AUSTRIA']['homes'] == []
    assert new_saved_game['phases'][1]['state']['builds']['FRANCE']['count'] == -1
    assert new_saved_game['phases'][1]['state']['builds']['FRANCE']['homes'] == []
    assert new_saved_game['phases'][1]['state']['builds']['RUSSIA']['count'] == 2
    assert new_saved_game['phases'][1]['state']['builds']['RUSSIA']['homes'] == ['xPAR', 'xMAR']
    assert new_saved_game['phases'][1]['state']['board_state'] == []

    assert new_saved_game['phases'][1]['orders']['FRANCE'] == ['xA PAR H', 'xA MAR - PIE']
    assert new_saved_game['phases'][1]['orders']['ENGLAND'] == []

    assert new_saved_game['phases'][1]['policy']['FRANCE']['locs'] == policy_2['locs']
    assert new_saved_game['phases'][1]['policy']['FRANCE']['tokens'] == policy_2['tokens']
    assert np.allclose(new_saved_game['phases'][1]['policy']['FRANCE']['log_probs'], policy_2['log_probs'])
    assert 'ENGLAND' not in new_saved_game['phases'][1]['policy']

    assert new_saved_game['phases'][1]['prev_orders_state'] == []
    assert 'FRANCE' not in new_saved_game['phases'][1]['state_value']
    assert 'ENGLAND' not in new_saved_game['phases'][1]['state_value']
    assert not new_saved_game['phases'][1]['possible_orders']

    assert new_saved_game['done_reason'] == 'auto_draw'
    assert new_saved_game['assigned_powers'] == ['FRANCE', 'ENGLAND']
    assert new_saved_game['players'] == ['rule', 'v0']
    assert new_saved_game['kwargs']['FRANCE']['player_seed'] == 1
    assert np.allclose(new_saved_game['kwargs']['FRANCE']['noise'], 0.2)
    assert np.allclose(new_saved_game['kwargs']['FRANCE']['temperature'], 0.3)
    assert np.allclose(new_saved_game['kwargs']['FRANCE']['dropout_rate'], 0.4)
    assert new_saved_game['kwargs']['ENGLAND']['player_seed'] == 6
    assert np.allclose(new_saved_game['kwargs']['ENGLAND']['noise'], 0.7)
    assert np.allclose(new_saved_game['kwargs']['ENGLAND']['temperature'], 0.8)
    assert np.allclose(new_saved_game['kwargs']['ENGLAND']['dropout_rate'], 0.9)

    assert new_saved_game['is_partial_game']
    assert new_saved_game['start_phase_ix'] == 2
    assert new_saved_game['reward_fn'] == 'default_reward'
    assert np.allclose(new_saved_game['rewards']['FRANCE'], [1.1, 2.2, 3.3])
    assert np.allclose(new_saved_game['rewards']['ENGLAND'], [1.2, 2.3, 3.4])
    assert np.allclose(new_saved_game['returns']['FRANCE'], [11.1, 12.2, 13.3])
    assert np.allclose(new_saved_game['returns']['ENGLAND'], [11.2, 12.3, 13.4])
