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
""" Generic class to test an advantage """
import os
import gym
from tornado import gen
from tornado.ioloop import IOLoop
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.gym import LimitNumberYears, RandomizePlayers
from diplomacy_research.models.gym.environment import DoneReason
from diplomacy_research.models.policy.order_based import PolicyAdapter, BaseDatasetBuilder
from diplomacy_research.models.policy.order_based.v012_film_diff_w_board_align_prev_ord import PolicyModel, load_args
from diplomacy_research.models.self_play.controller import generate_trajectory
from diplomacy_research.models.self_play.reward_functions import DefaultRewardFunction
from diplomacy_research.models.state_space import TOKENS_PER_ORDER, ALL_POWERS
from diplomacy_research.players import RuleBasedPlayer, ModelBasedPlayer
from diplomacy_research.players.rulesets import easy_ruleset
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import read_next_proto, write_proto_to_file

# Constants
HOME_DIR = os.path.expanduser('~')
if HOME_DIR == '~':
    raise RuntimeError('Cannot find home directory. Unable to save cache')


class AdvantageSetup():
    """ Tests an advantage """

    def __init__(self, model_type):
        """ Constructor """
        self.saved_game_cache_path = None
        self.model_type = model_type
        self.adapter = None
        self.advantage = None
        self.reward_fn = DefaultRewardFunction()
        self.graph = None

    def create_advantage(self):
        """ Creates the advantage object """
        raise NotImplementedError()

    @staticmethod
    def parse_flags(args):
        """ Parse flags without calling tf.app.run() """
        define = {'bool': lambda x: bool(x),            # pylint: disable=unnecessary-lambda
                  'int': lambda x: int(x),              # pylint: disable=unnecessary-lambda
                  'str': lambda x: str(x),              # pylint: disable=unnecessary-lambda
                  'float': lambda x: float(x),          # pylint: disable=unnecessary-lambda
                  '---': lambda x: x}                   # pylint: disable=unnecessary-lambda

        # Keeping a dictionary of parse args to overwrite if provided multiple times
        flags = {}
        for arg in args:
            arg_type, arg_name, arg_value, _ = arg
            flags[arg_name] = define[arg_type](arg_value)
            if arg_type == '---' and arg_name in flags:
                del flags[arg_name]
        return flags

    def run_tests(self):
        """ Run all tests """
        IOLoop.current().run_sync(self.run_tests_async)

    @gen.coroutine
    def run_tests_async(self):
        """ Run tests in an asynchronous IO Loop """
        from diplomacy_research.utils.tensorflow import tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            yield self.build_adapter()
            saved_game_proto = yield self.get_saved_game_proto()

            # Testing with a full game
            self.test_get_returns(saved_game_proto)
            self.test_get_transition_details(saved_game_proto)

            # Testing with a partial game
            saved_game_proto.done_reason = DoneReason.NOT_DONE.value
            self.test_get_returns(saved_game_proto)
            self.test_get_transition_details(saved_game_proto)

    @gen.coroutine
    def build_adapter(self):
        """ Builds adapter """
        from diplomacy_research.utils.tensorflow import tf
        hparams = self.parse_flags(load_args())

        # Generating model
        dataset = QueueDataset(batch_size=32, dataset_builder=BaseDatasetBuilder())
        model = PolicyModel(dataset, hparams)
        model.finalize_build()
        model.add_meta_information({'state_value': model.outputs['logits'][:, 0, 0]})
        self.adapter = PolicyAdapter(dataset, self.graph, tf.Session(graph=self.graph))
        self.create_advantage()

        # Setting cache path
        filename = '%s_savedgame.pbz' % self.model_type
        self.saved_game_cache_path = os.path.join(HOME_DIR, '.cache', 'diplomacy', filename)

    def test_get_returns(self, saved_game_proto):
        """ Tests the get_returns method """
        reward_fn = DefaultRewardFunction()

        for power_name in ALL_POWERS:
            rewards = reward_fn.get_episode_rewards(saved_game_proto, power_name)
            state_values = [1.] * len(rewards)
            last_state_value = 0.
            returns = self.advantage.get_returns(rewards, state_values, last_state_value)
            assert len(returns) == len(rewards)

    def test_get_transition_details(self, saved_game_proto):
        """ Tests the get_transition_details method """
        for power_name in saved_game_proto.assigned_powers[:-1]:
            result = self.advantage.get_transition_details(saved_game_proto, power_name)
            assert len(result) == len(saved_game_proto.phases) - 1                  # nb_transitions = nb_phases - 1
            for transition_detail in result:
                submitted_orders = transition_detail.transition.orders[power_name].value
                assert transition_detail.power_name == power_name
                assert transition_detail.transition is not None
                assert transition_detail.draw_action in (True, False)
                assert transition_detail.reward_target is not None
                assert transition_detail.value_target is not None
                assert len(transition_detail.log_probs) >= 1 or not submitted_orders

    @gen.coroutine
    def get_saved_game_proto(self):
        """ Tests the generate_saved_game_proto method """
        # Creating players
        player = ModelBasedPlayer(self.adapter)
        rule_player = RuleBasedPlayer(easy_ruleset)
        players = [player, player, player, player, player, player, rule_player]

        def env_constructor(players):
            """ Env constructor """
            env = gym.make('DiplomacyEnv-v0')
            env = LimitNumberYears(env, 5)
            env = RandomizePlayers(env, players)
            return env

        # Generating game
        saved_game_proto = None
        if os.path.exists(self.saved_game_cache_path):
            with open(self.saved_game_cache_path, 'rb') as file:
                saved_game_proto = read_next_proto(SavedGameProto, file, compressed=True)

        if saved_game_proto is None:
            saved_game_proto = yield generate_trajectory(players, self.reward_fn, self.advantage, env_constructor)
            with open(self.saved_game_cache_path, 'wb') as file:
                write_proto_to_file(file, saved_game_proto, compressed=True)

        # Validating game
        assert saved_game_proto.id
        assert len(saved_game_proto.phases) >= 10

        # Validating policy details
        for phase in saved_game_proto.phases:
            for power_name in phase.policy:
                nb_locs = len(phase.policy[power_name].locs)
                assert (len(phase.policy[power_name].tokens) == nb_locs * TOKENS_PER_ORDER          # Token-based
                        or len(phase.policy[power_name].tokens) == nb_locs)                         # Order-based
                assert len(phase.policy[power_name].log_probs) == len(phase.policy[power_name].tokens)
                assert phase.policy[power_name].draw_action in (True, False)
                assert 0. <= phase.policy[power_name].draw_prob <= 1.

        # Returning saved game proto for other tests to use
        return saved_game_proto
