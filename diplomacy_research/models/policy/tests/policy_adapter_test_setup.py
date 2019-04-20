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
""" Generic class to tests for model and adapter correctness """
# pylint: disable=too-many-lines
import os
import shutil
import tempfile
import numpy as np
from tornado import gen
from tornado.ioloop import IOLoop
from diplomacy import Game
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.policy.base_policy_model import POLICY_BEAM_WIDTH
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto, TOKENS_PER_ORDER, get_orderable_locs_for_powers, get_map_powers, NB_NODES, \
    NB_FEATURES
from diplomacy_research.utils.checkpoint import freeze_graph, build_saved_model
from diplomacy_research.utils.cluster import process_fetches_dict
from diplomacy_research.utils.model import int_prod

class PolicyAdapterTestSetup():
    """ Creates a testable setup to test a model and a constructor """

    def __init__(self, policy_model_ctor, value_model_ctor, draw_model_ctor, dataset_builder, policy_adapter_ctor,
                 load_policy_args, load_value_args, load_draw_args, strict=True):
        """ Constructor
            :param policy_model_ctor: The policy model constructor to create the policy model.
            :param value_model_ctor: The value model constructor to create the value model.
            :param draw_model_ctor: The draw model constructor to create the draw model.
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param policy_adaptor_ctor: The policy adapter constructor to create the policy adapter
            :param load_policy_args: Reference to the callable function required to load policy args
            :param load_value_args: Reference to the callable function required to load value args
            :param load_draw_args: Reference to the callable function required to load draw args
            :param strict: Boolean. Uses strict tests, otherwise allow more variation in results.
            :type policy_model_ctor: diplomacy_research.models.policy.base_policy_model.BasePolicyModel.__class__
            :type value_model_ctor: diplomacy_research.models.value.base_value_model.BaseValueModel.__class__
            :type draw_model_ctor: diplomacy_research.models.draw.base_draw_model.BaseDrawModel.__class__
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type policy_adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        """
        # pylint: disable=too-many-arguments
        # Parsing new flags
        args = load_policy_args()
        if load_value_args is not None:
            args += load_value_args()
        if load_draw_args is not None:
            args += load_draw_args()
        self.hparams = self.parse_flags(args)
        self.strict = strict

        # Other attributes
        self.graph = None
        self.sess = None
        self.adapter = None
        self.queue_dataset = None
        self.policy_model_ctor = policy_model_ctor
        self.value_model_ctor = value_model_ctor
        self.draw_model_ctor = draw_model_ctor
        self.dataset_builder = dataset_builder
        self.policy_adapter_ctor = policy_adapter_ctor

    def build_model(self):
        """ Builds the model """
        from diplomacy_research.utils.tensorflow import tf
        graph = tf.Graph()
        with graph.as_default():

            # Creating dataset
            self.queue_dataset = QueueDataset(batch_size=self.hparams['batch_size'],
                                              dataset_builder=self.dataset_builder)

            # Creating model and validating
            model = self.policy_model_ctor(self.queue_dataset, self.hparams)
            if self.value_model_ctor is not None:
                model = self.value_model_ctor(model, self.queue_dataset, self.hparams)
            if self.draw_model_ctor is not None:
                model = self.draw_model_ctor(model, self.queue_dataset, self.hparams)
            model.finalize_build()
            model.validate()

            # Testing encode_board(), get_board_state_conv() get_board_value()
            self.test_encode_board(model)
            self.test_get_board_state_conv(model)
            self.test_get_board_value(model)

        self.graph = graph
        self.sess = tf.Session(graph=graph)

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
        self.build_model()
        self.adapter = self.policy_adapter_ctor(self.queue_dataset, self.graph, session=self.sess)
        yield self.test_load_from_checkpoint()
        self.test_freeze_and_saved_model()
        self.test_is_trainable()
        self.test_tokenize()
        self.test_get_feedable_item()
        yield self.test_get_orders()
        yield self.test_get_orders_with_beam()
        yield self.test_get_orders_with_value()
        yield self.test_get_beam_orders()
        yield self.test_get_beam_orders_with_value()
        yield self.test_get_updated_policy_details()
        yield self.test_expand()
        yield self.test_get_state_value()
        self.test_graph_size()

    @staticmethod
    def test_encode_board(model):
        """ Tests the encode_board_method """
        from diplomacy_research.utils.tensorflow import tf
        global_vars_before = set(tf.global_variables())
        model.encode_board(tf.placeholder(dtype=tf.float32, shape=[None, NB_NODES, NB_FEATURES], name='fake_board'),
                           name='board_state_conv',
                           reuse=True)
        global_vars_after = set(tf.global_variables())
        allowed_new_vars = {var for var in tf.global_variables() if 'policy' in var.name and '/proj/' in var.name}
        assert not global_vars_after - global_vars_before - allowed_new_vars, 'New variables added when encoding board.'

    @staticmethod
    def test_get_board_state_conv(model):
        """ Tests the get_board_state_conv method """
        from diplomacy_research.utils.tensorflow import tf
        fake_board_conv = tf.placeholder(dtype=tf.float32, shape=[None, NB_NODES, NB_FEATURES], name='fake_board_conv')
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name='fake_is_training')
        model.get_board_state_conv(fake_board_conv, is_training, fake_board_conv)

    @staticmethod
    def test_get_board_value(model):
        """ Tests the get_board_value method (with and/or without a value function) """
        from diplomacy_research.utils.tensorflow import tf
        global_vars_before = set(tf.global_variables())
        board_state = tf.placeholder(dtype=tf.float32, shape=[None, NB_NODES, NB_FEATURES], name='fake_board')
        current_power = tf.placeholder(dtype=tf.int32, shape=[None], name='fake_current_power')
        model.get_board_value(board_state, current_power, reuse=True)
        global_vars_after = set(tf.global_variables())
        assert not global_vars_after - global_vars_before, 'New variables added when getting board value.'

    @gen.coroutine
    def test_load_from_checkpoint(self):
        """ Checks if the model can be frozen to disk and then loaded back """
        from diplomacy_research.utils.tensorflow import tf

        # Freezing model to disk
        temp_dir = tempfile.mkdtemp()
        freeze_graph(temp_dir, 0, graph=self.graph, session=self.sess)

        # Rebuilding a new graph and querying a move from the new graph
        new_graph = tf.Graph()
        with new_graph.as_default():
            new_session = tf.Session(graph=new_graph)
            new_dataset = QueueDataset(batch_size=32,
                                       dataset_builder=self.dataset_builder,
                                       no_iterator=True)
        new_adapter = self.policy_adapter_ctor(new_dataset, new_graph, new_session)
        new_adapter.load_from_checkpoint(os.path.join(temp_dir, 'frozen_graph-v000000000.pb'))

        # Querying moves
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}
        orders, _ = yield self.adapter.get_orders(locs,
                                                  state_proto,
                                                  'FRANCE',
                                                  phase_history_proto,
                                                  possible_orders_proto,
                                                  **kwargs)
        assert orders

        # Deleting tempdir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_freeze_and_saved_model(self):
        """ Tests the freeze_graph and build_saved_model functions """
        freeze_temp_dir = tempfile.mkdtemp()
        freeze_graph(freeze_temp_dir, 0, graph=self.graph, session=self.sess)
        assert os.path.exists(os.path.join(freeze_temp_dir, 'frozen_graph-v000000000.pb'))
        shutil.rmtree(freeze_temp_dir, ignore_errors=True)

        saved_temp_dir = tempfile.mkdtemp()
        build_saved_model(saved_temp_dir,
                          version_id=0,
                          signature=self.adapter.get_signature(),
                          proto_fields=self.dataset_builder.get_proto_fields(),
                          graph=self.graph,
                          session=self.sess)
        assert os.path.exists(os.path.join(saved_temp_dir, '000000000', 'saved_model.pb'))
        shutil.rmtree(saved_temp_dir, ignore_errors=True)

    def test_is_trainable(self):
        """ Checks if the .is_trainable property works """
        assert self.adapter.is_trainable

    def test_tokenize(self):
        """ Checks if .tokenize is implemented """
        tokens = self.adapter.tokenize('A PAR H')
        assert tokens

    def test_get_feedable_item(self):
        """ Checks if the .get_feedable_item method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}
        assert self.adapter.feedable_dataset.get_feedable_item(locs,
                                                               state_proto,
                                                               'FRANCE',
                                                               phase_history_proto,
                                                               possible_orders_proto,
                                                               **kwargs)

    @gen.coroutine
    def test_get_orders(self):
        """ Checks if the .get_orders method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        temp_0_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 0.,
                         'dropout_rate': 0.}
        temp_1_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 1.,
                         'dropout_rate': 0.}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       **temp_1_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_1_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       fetches=fetches,
                                                                       **temp_1_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (policy_details['locs'] == locs) or (policy_details['locs'] == ['PAR', 'MAR'])
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.

        # Temperature == 0.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       **temp_0_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_0_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       fetches=fetches,
                                                                       **temp_0_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (policy_details['locs'] == locs) or (policy_details['locs'] == ['PAR', 'MAR'])
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.

    @gen.coroutine
    def test_get_orders_with_beam(self):
        """ Checks if the .get_orders method works with beam search """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        temp_0_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 0.,
                         'dropout_rate': 0.,
                         'use_beam': True}
        temp_1_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 1.,
                         'dropout_rate': 0.,
                         'use_beam': True}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       **temp_1_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_1_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       fetches=fetches,
                                                                       **temp_1_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.

        # Temperature == 0.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       **temp_0_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_0_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details = yield self.adapter.get_orders(locs,
                                                                       state_proto,
                                                                       'FRANCE',
                                                                       phase_history_proto,
                                                                       possible_orders_proto,
                                                                       fetches=fetches,
                                                                       **temp_0_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.

    @gen.coroutine
    def test_get_orders_with_value(self):
        """ Checks if the .get_orders method with state value works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        temp_0_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 0.,
                         'dropout_rate': 0.,
                         'with_state_value': True}
        temp_1_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 1.,
                         'dropout_rate': 0.,
                         'with_state_value': True}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details, state_value = yield self.adapter.get_orders(locs,
                                                                                    state_proto,
                                                                                    'FRANCE',
                                                                                    phase_history_proto,
                                                                                    possible_orders_proto,
                                                                                    **temp_1_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_1_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details, state_value = yield self.adapter.get_orders(locs,
                                                                                    state_proto,
                                                                                    'FRANCE',
                                                                                    phase_history_proto,
                                                                                    possible_orders_proto,
                                                                                    fetches=fetches,
                                                                                    **temp_1_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (policy_details['locs'] == locs) or (policy_details['locs'] == ['PAR', 'MAR'])
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.
            assert state_value != 0.

        # Temperature == 0.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                orders, policy_details, state_value = yield self.adapter.get_orders(locs,
                                                                                    state_proto,
                                                                                    'FRANCE',
                                                                                    phase_history_proto,
                                                                                    possible_orders_proto,
                                                                                    **temp_0_kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **temp_0_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders, policy_details, state_value = yield self.adapter.get_orders(locs,
                                                                                    state_proto,
                                                                                    'FRANCE',
                                                                                    phase_history_proto,
                                                                                    possible_orders_proto,
                                                                                    fetches=fetches,
                                                                                    **temp_0_kwargs)

            assert (len(orders) == 3 and orders[2] == '') or (len(orders) == 2)
            assert (policy_details['locs'] == locs) or (policy_details['locs'] == ['PAR', 'MAR'])
            assert (len(policy_details['tokens']) == TOKENS_PER_ORDER * len(policy_details['locs'])       # Token-based
                    or len(policy_details['tokens']) == len(policy_details['locs']))                      # Order-based
            assert len(policy_details['log_probs']) == len(policy_details['tokens'])
            assert policy_details['draw_action'] in (True, False)
            assert 0. <= policy_details['draw_prob'] <= 1.
            assert state_value != 0.

    @gen.coroutine
    def test_get_beam_orders(self):
        """ Checks if the .get_beam_orders method works  """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        temp_1_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 1.,
                         'dropout_rate': 0.,
                         'use_beam': True,
                         'with_state_value': True}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                beam_orders, beam_probs, state_value = yield self.adapter.get_beam_orders(locs,
                                                                                          state_proto,
                                                                                          'FRANCE',
                                                                                          phase_history_proto,
                                                                                          possible_orders_proto,
                                                                                          **temp_1_kwargs)
            else:
                fetches = yield self.adapter.get_beam_orders(locs,
                                                             state_proto,
                                                             'FRANCE',
                                                             phase_history_proto,
                                                             possible_orders_proto,
                                                             prefetch=True,
                                                             **temp_1_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                beam_orders, beam_probs, state_value = yield self.adapter.get_beam_orders(locs,
                                                                                          state_proto,
                                                                                          'FRANCE',
                                                                                          phase_history_proto,
                                                                                          possible_orders_proto,
                                                                                          fetches=fetches,
                                                                                          **temp_1_kwargs)

            assert len(beam_orders) == POLICY_BEAM_WIDTH
            assert (len(beam_orders[0]) == 3 and beam_orders[0][2] == '') or len(beam_orders[0]) == 2
            assert len(beam_probs) == POLICY_BEAM_WIDTH
            assert state_value != 0.

    @gen.coroutine
    def test_get_beam_orders_with_value(self):
        """ Checks if the .get_beam_orders method with state value works  """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        temp_1_kwargs = {'player_seed': 0,
                         'noise': 0.,
                         'temperature': 1.,
                         'dropout_rate': 0.,
                         'use_beam': True}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                beam_orders, beam_probs = yield self.adapter.get_beam_orders(locs,
                                                                             state_proto,
                                                                             'FRANCE',
                                                                             phase_history_proto,
                                                                             possible_orders_proto,
                                                                             **temp_1_kwargs)
            else:
                fetches = yield self.adapter.get_beam_orders(locs,
                                                             state_proto,
                                                             'FRANCE',
                                                             phase_history_proto,
                                                             possible_orders_proto,
                                                             prefetch=True,
                                                             **temp_1_kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                beam_orders, beam_probs = yield self.adapter.get_beam_orders(locs,
                                                                             state_proto,
                                                                             'FRANCE',
                                                                             phase_history_proto,
                                                                             possible_orders_proto,
                                                                             fetches=fetches,
                                                                             **temp_1_kwargs)

            assert len(beam_orders) == POLICY_BEAM_WIDTH
            assert (len(beam_orders[0]) == 3 and beam_orders[0][2] == '') or len(beam_orders[0]) == 2
            assert len(beam_probs) == POLICY_BEAM_WIDTH

    @gen.coroutine
    def test_get_updated_policy_details(self):
        """ Checks if the .get_updated_policy_details method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        power_name = 'FRANCE'
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}

        # Testing with and without prefetching
        for use_prefetching in (False, True):
            orderable_locs, _ = get_orderable_locs_for_powers(state_proto, [power_name])

            if not use_prefetching:
                submitted_orders, old_policy_details = yield self.adapter.get_orders(orderable_locs,
                                                                                     state_proto,
                                                                                     power_name,
                                                                                     phase_history_proto,
                                                                                     possible_orders_proto,
                                                                                     **kwargs)
                new_details_1 = yield self.adapter.get_updated_policy_details(state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              old_policy_details=old_policy_details,
                                                                              **kwargs)
                new_details_2 = yield self.adapter.get_updated_policy_details(state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              submitted_orders=submitted_orders,
                                                                              **kwargs)
            else:
                fetches = yield self.adapter.get_orders(orderable_locs,
                                                        state_proto,
                                                        power_name,
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                submitted_orders, old_policy_details = yield self.adapter.get_orders(orderable_locs,
                                                                                     state_proto,
                                                                                     power_name,
                                                                                     phase_history_proto,
                                                                                     possible_orders_proto,
                                                                                     fetches=fetches,
                                                                                     **kwargs)
                fetches = {}
                fetches['a'] = yield self.adapter.get_updated_policy_details(state_proto,
                                                                             power_name,
                                                                             phase_history_proto,
                                                                             possible_orders_proto,
                                                                             old_policy_details=old_policy_details,
                                                                             prefetch=True,
                                                                             **kwargs)
                fetches['b'] = yield self.adapter.get_updated_policy_details(state_proto,
                                                                             power_name,
                                                                             phase_history_proto,
                                                                             possible_orders_proto,
                                                                             submitted_orders=submitted_orders,
                                                                             prefetch=True,
                                                                             **kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                new_details_1 = yield self.adapter.get_updated_policy_details(state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              old_policy_details=old_policy_details,
                                                                              fetches=fetches['a'],
                                                                              **kwargs)
                new_details_2 = yield self.adapter.get_updated_policy_details(state_proto,
                                                                              power_name,
                                                                              phase_history_proto,
                                                                              possible_orders_proto,
                                                                              submitted_orders=submitted_orders,
                                                                              fetches=fetches['b'],
                                                                              **kwargs)

            # Validating policy_details using old_policy_details
            assert new_details_1['locs']
            assert old_policy_details['locs'] == new_details_1['locs']
            assert old_policy_details['tokens'] == new_details_1['tokens']
            assert len(old_policy_details['log_probs']) == len(new_details_1['log_probs'])
            assert old_policy_details['draw_action'] == new_details_1['draw_action']
            assert 0. <= old_policy_details['draw_prob'] <= 1.
            assert 0. <= new_details_1['draw_prob'] <= 1.
            if self.strict:
                assert np.allclose(old_policy_details['log_probs'], new_details_1['log_probs'], atol=1e-4)

            # Validating policy_details using submitted_orders
            assert new_details_2['locs'] == new_details_1['locs']
            assert new_details_2['tokens'] == new_details_1['tokens']
            if self.strict:
                assert np.allclose(new_details_2['log_probs'], new_details_1['log_probs'], atol=1e-4)
            assert new_details_2['draw_action'] in (True, False)
            assert 0. <= new_details_2['draw_prob'] <= 1.

    @gen.coroutine
    def test_expand(self):
        """ Checks if the .expand method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['MAR', 'BUR']
        confirmed_orders = ['A PAR H']
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}

        # With and without prefetching
        for use_prefetching in (False, True):

            if not use_prefetching:
                orders_probs_log_probs = yield self.adapter.expand(confirmed_orders,
                                                                   locs,
                                                                   state_proto,
                                                                   'FRANCE',
                                                                   phase_history_proto,
                                                                   possible_orders_proto,
                                                                   **kwargs)
            else:
                fetches = yield self.adapter.expand(confirmed_orders,
                                                    locs,
                                                    state_proto,
                                                    'FRANCE',
                                                    phase_history_proto,
                                                    possible_orders_proto,
                                                    prefetch=True,
                                                    **kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                orders_probs_log_probs = yield self.adapter.expand(confirmed_orders,
                                                                   locs,
                                                                   state_proto,
                                                                   'FRANCE',
                                                                   phase_history_proto,
                                                                   possible_orders_proto,
                                                                   fetches=fetches,
                                                                   **kwargs)

            # Validating
            assert len(orders_probs_log_probs) == len(locs)
            assert len(orders_probs_log_probs['MAR']) > 1
            assert len(orders_probs_log_probs['BUR']) <= 1

            # Making sure the probability of 'MAR' sums to 1.
            cumulative_prob = sum([item.probability for item in orders_probs_log_probs['MAR']])
            assert 0.999 <= cumulative_prob < 1.001, 'Cumulative prob of %.8f does not sum to 1.' % cumulative_prob

    @gen.coroutine
    def test_deterministic(self):
        """ Makes sure the policy always return the same probs for the same query """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        _, orderable_locs = get_orderable_locs_for_powers(state_proto, get_map_powers(game.map))
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}

        for power_name in get_map_powers(game.map):
            orders_probs_log_probs_1 = yield self.adapter.expand([],
                                                                 orderable_locs[power_name],
                                                                 state_proto,
                                                                 power_name,
                                                                 phase_history_proto,
                                                                 possible_orders_proto,
                                                                 **kwargs)
            orders_probs_log_probs_2 = yield self.adapter.expand([],
                                                                 orderable_locs[power_name],
                                                                 state_proto,
                                                                 power_name,
                                                                 phase_history_proto,
                                                                 possible_orders_proto,
                                                                 **kwargs)

            for loc in orders_probs_log_probs_1:
                assert orders_probs_log_probs_1[loc] == orders_probs_log_probs_2[loc]

    @gen.coroutine
    def test_get_state_value(self):
        """ Checks if the .get_state_value method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        kwargs = {'player_seed': 0,
                  'noise': 0.,
                  'temperature': 0.,
                  'dropout_rate': 0.}

        # With and without prefetching
        for use_prefetching in (False, True):

            if not use_prefetching:
                state_value = yield self.adapter.get_state_value(state_proto,
                                                                 'FRANCE',
                                                                 phase_history_proto,
                                                                 possible_orders_proto,
                                                                 **kwargs)
            else:
                fetches = yield self.adapter.get_state_value(state_proto,
                                                             'FRANCE',
                                                             phase_history_proto,
                                                             possible_orders_proto,
                                                             prefetch=True,
                                                             **kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                state_value = yield self.adapter.get_state_value(state_proto,
                                                                 'FRANCE',
                                                                 phase_history_proto,
                                                                 possible_orders_proto,
                                                                 fetches=fetches,
                                                                 **kwargs)

            assert state_value != 0.

    def test_graph_size(self):
        """ Tests the graph size """
        from diplomacy_research.utils.tensorflow import tf
        total_size = 0
        for var in self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            total_size += int_prod(var.shape.as_list()) * var.dtype.size
        total_size_mb = total_size / (1024. ** 2)
        assert total_size_mb < 750., 'Graph too large - Maximum: 750MB - Currently: %.2f MB' % total_size_mb
