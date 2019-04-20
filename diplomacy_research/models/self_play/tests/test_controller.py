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
""" Generic class to test the controller """
import gym
from tornado import gen
from tornado.ioloop import IOLoop
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.draw.v001_draw_relu import DrawModel, load_args as load_draw_args
from diplomacy_research.models.gym import AutoDraw, LimitNumberYears, RandomizePlayers
from diplomacy_research.models.policy.order_based import PolicyAdapter, BaseDatasetBuilder
from diplomacy_research.models.policy.order_based.v012_film_diff_w_board_align_prev_ord import PolicyModel, load_args
from diplomacy_research.models.self_play.advantages.monte_carlo import MonteCarlo
from diplomacy_research.models.self_play.controller import generate_trajectory
from diplomacy_research.models.self_play.reward_functions import DefaultRewardFunction
from diplomacy_research.models.state_space import TOKENS_PER_ORDER, NB_POWERS
from diplomacy_research.models.value.v001_val_relu_7 import ValueModel, load_args as load_value_args
from diplomacy_research.players import RuleBasedPlayer, ModelBasedPlayer
from diplomacy_research.players.rulesets import easy_ruleset
from diplomacy_research.utils.process import run_in_separate_process

class ControllerSetup():
    """ Tests the controller """

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
        yield self.test_generate_saved_game_proto()

    @gen.coroutine
    def test_generate_saved_game_proto(self):
        """ Tests the generate_saved_game_proto method """
        from diplomacy_research.utils.tensorflow import tf
        hparams = self.parse_flags(load_args() + load_value_args() + load_draw_args())

        # Generating model
        graph = tf.Graph()
        with graph.as_default():
            dataset = QueueDataset(batch_size=32, dataset_builder=BaseDatasetBuilder())
            model = PolicyModel(dataset, hparams)
            model = ValueModel(model, dataset, hparams)
            model = DrawModel(model, dataset, hparams)
            model.finalize_build()
            adapter = PolicyAdapter(dataset, graph, tf.Session(graph=graph))
            advantage_fn = MonteCarlo(gamma=0.99)
            reward_fn = DefaultRewardFunction()

            # Creating players
            player = ModelBasedPlayer(adapter)
            rule_player = RuleBasedPlayer(easy_ruleset)
            players = [player, player, player, player, player, player, rule_player]

            def env_constructor(players):
                """ Env constructor """
                env = gym.make('DiplomacyEnv-v0')
                env = LimitNumberYears(env, 5)
                env = RandomizePlayers(env, players)
                env = AutoDraw(env)
                return env

            # Generating game
            saved_game_proto = yield generate_trajectory(players, reward_fn, advantage_fn, env_constructor)

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

        # Validating assignments
        assert len(saved_game_proto.assigned_powers) == NB_POWERS

        # Validating rewards and returns
        assert saved_game_proto.reward_fn == DefaultRewardFunction().name
        for power_name in saved_game_proto.assigned_powers:
            assert len(saved_game_proto.rewards[power_name].value) == len(saved_game_proto.phases) - 1
            assert len(saved_game_proto.returns[power_name].value) == len(saved_game_proto.phases) - 1

def launch():
    """ Launches the test """
    test_object = ControllerSetup()
    test_object.run_tests()

def test_run():
    """ Runs the test """
    run_in_separate_process(target=launch, timeout=240)
