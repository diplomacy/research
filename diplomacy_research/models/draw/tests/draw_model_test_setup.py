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
""" Generic class to tests for draw model correctness """
from tornado import gen
from tornado.ioloop import IOLoop
from diplomacy import Game
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto
from diplomacy_research.utils.cluster import process_fetches_dict


class DrawModelTestSetup():
    """ Creates a testable setup to test a model and a constructor """

    def __init__(self, policy_model_ctor, value_model_ctor, draw_model_ctor, dataset_builder, adapter_ctor,
                 load_policy_args, load_value_args, load_draw_args):
        """ Constructor
            :param policy_model_ctor: The policy model constructor to create the policy.
            :param value_model_ctor: The value model constructor to create the value model.
            :param draw_model_ctor: The draw model constructor to create the draw model.
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param adaptor_ctor: The policy adapter constructor to create the policy adapter
            :param load_policy_args: Reference to the callable function required to load policy args
            :param load_value_args: Reference to the callable function required to load value args
            :param load_draw_args: Reference to the callable function required to load draw args
            :type policy_model_ctor: diplomacy_research.models.policy.base_policy_model.BasePolicyModel.__class__
            :type value_model_ctor: diplomacy_research.models.value.base_value_model.BaseValueModel.__class__
            :type draw_model_ctor: diplomacy_research.models.draw.base_draw_model.BaseDrawModel.__class__
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        """
        # pylint: disable=too-many-arguments
        # Parsing new flags
        args = load_policy_args()
        if load_value_args is not None:
            args += load_value_args()
        args += load_draw_args()
        self.hparams = self.parse_flags(args)

        # Other attributes
        self.graph = None
        self.sess = None
        self.adapter = None
        self.queue_dataset = None
        self.policy_model_ctor = policy_model_ctor
        self.value_model_ctor = value_model_ctor
        self.draw_model_ctor = draw_model_ctor
        self.dataset_builder = dataset_builder
        self.adapter_ctor = adapter_ctor

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
            model = self.draw_model_ctor(model, self.queue_dataset, self.hparams)
            model.finalize_build()
            model.validate()

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
        self.adapter = self.adapter_ctor(self.queue_dataset, self.graph, session=self.sess)
        yield self.test_get_draw_prob()

    @gen.coroutine
    def test_get_draw_prob(self):
        """ Checks if the .get_draw_prob method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        kwargs = {'player_seed': 0, 'noise': 0., 'temperature': 1., 'dropout_rate': 0.}

        # Temperature == 1.
        # With and without prefetching
        for use_prefetching in (False, True):
            if not use_prefetching:
                _, policy_details = yield self.adapter.get_orders(locs,
                                                                  state_proto,
                                                                  'FRANCE',
                                                                  phase_history_proto,
                                                                  possible_orders_proto,
                                                                  **kwargs)
            else:
                fetches = yield self.adapter.get_orders(locs,
                                                        state_proto,
                                                        'FRANCE',
                                                        phase_history_proto,
                                                        possible_orders_proto,
                                                        prefetch=True,
                                                        **kwargs)
                fetches = yield process_fetches_dict(self.queue_dataset, fetches)
                _, policy_details = yield self.adapter.get_orders(locs,
                                                                  state_proto,
                                                                  'FRANCE',
                                                                  phase_history_proto,
                                                                  possible_orders_proto,
                                                                  fetches=fetches,
                                                                  **kwargs)

            assert policy_details['draw_action'] in (True, False)
            assert 0. < policy_details['draw_prob'] < 1.
