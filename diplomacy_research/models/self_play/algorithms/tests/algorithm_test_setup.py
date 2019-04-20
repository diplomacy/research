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
""" Generic class to test an algorithm """
import os
import gym
from tornado import gen
from tornado.ioloop import IOLoop
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.draw.v001_draw_relu import DrawModel, load_args as load_draw_args
from diplomacy_research.models.gym import AutoDraw, LimitNumberYears, RandomizePlayers
from diplomacy_research.models.self_play.controller import generate_trajectory
from diplomacy_research.models.self_play.reward_functions import DefaultRewardFunction, DEFAULT_PENALTY
from diplomacy_research.models.state_space import TOKENS_PER_ORDER
from diplomacy_research.models.value.v004_board_state_conv import ValueModel, load_args as load_value_args
from diplomacy_research.players import RuleBasedPlayer, ModelBasedPlayer
from diplomacy_research.players.rulesets import easy_ruleset
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import read_next_proto, write_proto_to_file

# Constants
HOME_DIR = os.path.expanduser('~')
if HOME_DIR == '~':
    raise RuntimeError('Cannot find home directory. Unable to save cache')


class AlgorithmSetup():
    """ Tests an algorithm """

    def __init__(self, algorithm_ctor, algo_load_args, model_type):
        """ Constructor
            :param algorithm_ctor: The constructor class for the Algorithm
            :param algo_args: The method load_args() for the algorithm
            :param model_type: The model type ("order_based", "token_based") of the policy (for caching)
        """
        self.saved_game_cache_path = None
        self.model_type = model_type
        self._algorithm_ctor = algorithm_ctor
        self.get_algo_load_args = algo_load_args
        self.adapter = None
        self.algorithm = None
        self.advantage = None
        self.reward_fn = DefaultRewardFunction()
        self.graph = None

    def create_algorithm(self, feedable_dataset, model, hparams):
        """ Creates the algorithm object """
        self.algorithm = self._algorithm_ctor(feedable_dataset, model, hparams)

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

    @staticmethod
    def get_policy_model():
        """ Returns the PolicyModel """
        raise NotImplementedError()

    @staticmethod
    def get_policy_builder():
        """ Returns the Policy's DatasetBuilder """
        raise NotImplementedError()

    @staticmethod
    def get_policy_adapter():
        """ Returns the PolicyAdapter """
        raise NotImplementedError()

    @staticmethod
    def get_policy_load_args():
        """ Returns the policy args """
        return []

    @staticmethod
    def get_test_load_args():
        """ Overrides common hparams to speed up tests. """
        return [('int', 'nb_graph_conv', 3, 'Number of Graph Conv Layer'),
                ('int', 'word_emb_size', 64, 'Word embedding size.'),
                ('int', 'order_emb_size', 64, 'Order embedding size.'),
                ('int', 'power_emb_size', 64, 'Power embedding size.'),
                ('int', 'season_emb_size', 10, 'Season embedding size.'),
                ('int', 'board_emb_size', 40, 'Embedding size for the board state'),
                ('int', 'gcn_size', 24, 'Size of graph convolution outputs.'),
                ('int', 'lstm_size', 64, 'LSTM (Encoder and Decoder) size.'),
                ('int', 'attn_size', 30, 'LSTM decoder attention size.'),
                ('int', 'value_embedding_size', 64, 'Embedding size.'),
                ('int', 'value_h1_size', 16, 'The size of the first hidden layer in the value calculation'),
                ('int', 'value_h2_size', 16, 'The size of the second hidden layer in the value calculation'),
                ('bool', 'use_v_dropout', True, 'Use variational dropout (same mask across all time steps)'),
                ('bool', 'use_xla', False, 'Use XLA compilation.'),
                ('str', 'mode', 'self-play', 'The RL training mode.')]

    def run_tests(self):
        """ Run all tests """
        IOLoop.current().run_sync(self.run_tests_async)

    @gen.coroutine
    def run_tests_async(self):
        """ Run tests in an asynchronous IO Loop """
        from diplomacy_research.utils.tensorflow import tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            yield self.build_algo_and_adapter()
            saved_game_proto = yield self.get_saved_game_proto()
            yield self.test_clear_buffers()
            yield self.test_learn(saved_game_proto)
            assert self.adapter.session.run(self.algorithm.version_step) == 0
            yield self.test_update()
            yield self.test_init()
            assert self.adapter.session.run(self.algorithm.version_step) == 1
            yield self.test_get_priorities(saved_game_proto)

    @gen.coroutine
    def test_learn(self, saved_game_proto):
        """ Tests the algorithm learn method """
        power_phases_ix = self.algorithm.get_power_phases_ix(saved_game_proto, 1)
        yield self.algorithm.learn([saved_game_proto], [power_phases_ix], self.advantage)

    @gen.coroutine
    def test_get_priorities(self, saved_game_proto):
        """ Tests the algorithm get_priorities method """
        power_phases_ix = self.algorithm.get_power_phases_ix(saved_game_proto, 1)
        yield self.algorithm.clear_buffers()
        yield self.algorithm.learn([saved_game_proto], [power_phases_ix], self.advantage)
        priorities = yield self.algorithm.get_priorities([saved_game_proto], self.advantage)
        assert len(priorities) == len(self.algorithm.list_power_phases_per_game.get(saved_game_proto.id, []))

    @gen.coroutine
    def test_update(self):
        """ Tests the algorithm update method """
        results = yield self.algorithm.update(memory_buffer=None)
        for eval_tag in self.algorithm.get_evaluation_tags():
            assert eval_tag in results
            assert results[eval_tag]

    @gen.coroutine
    def test_init(self):
        """ Tests the algorithm init method """
        yield self.algorithm.init()

    @gen.coroutine
    def test_clear_buffers(self):
        """ Tests the algorithm clear_buffers method """
        yield self.algorithm.clear_buffers()

    @gen.coroutine
    def build_algo_and_adapter(self):
        """ Builds adapter """
        from diplomacy_research.utils.tensorflow import tf

        policy_model_ctor = self.get_policy_model()
        dataset_builder_ctor = self.get_policy_builder()
        policy_adapter_ctor = self.get_policy_adapter()
        extra_proto_fields = self._algorithm_ctor.get_proto_fields()

        hparams = self.parse_flags(self.get_policy_load_args()
                                   + load_value_args()
                                   + load_draw_args()
                                   + self.get_algo_load_args()
                                   + self.get_test_load_args())

        # Generating model
        dataset = QueueDataset(batch_size=32,
                               dataset_builder=dataset_builder_ctor(extra_proto_fields=extra_proto_fields))
        model = policy_model_ctor(dataset, hparams)
        model = ValueModel(model, dataset, hparams)
        model = DrawModel(model, dataset, hparams)
        model.finalize_build()
        self.create_algorithm(dataset, model, hparams)
        self.adapter = policy_adapter_ctor(dataset, self.graph, tf.Session(graph=self.graph))
        self.advantage = self._algorithm_ctor.create_advantage_function(hparams,
                                                                        gamma=0.99,
                                                                        penalty_per_phase=DEFAULT_PENALTY)

        # Setting cache path
        filename = '%s_savedgame.pbz' % self.model_type
        self.saved_game_cache_path = os.path.join(HOME_DIR, '.cache', 'diplomacy', filename)

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
            env = AutoDraw(env)
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

        # Validating rewards
        assert saved_game_proto.reward_fn == DefaultRewardFunction().name
        for power_name in saved_game_proto.assigned_powers:
            assert len(saved_game_proto.rewards[power_name].value) == len(saved_game_proto.phases) - 1

        # Returning saved game proto for other tests to use
        return saved_game_proto
