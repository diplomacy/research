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
""" Generic class to tests for policy builder correctness """
from diplomacy import Game
from diplomacy_research.models.state_space import extract_state_proto, extract_phase_history_proto, \
    extract_possible_orders_proto

class PolicyBuilderTestSetup():
    """ Creates a testable setup to test a policy dataset builder """

    def __init__(self, dataset_builder):
        """ Constructor
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
        """
        self.dataset_builder = dataset_builder

    def run_tests(self):
        """ Run all tests """
        self.run_tests_sync()

    def run_tests_sync(self):
        """ Run tests """
        self.test_paths()
        self.test_get_feedable_item()
        self.test_proto_generation_callable()
        self.test_required_features()

    def test_paths(self):
        """ Checks if the training, validation and index paths are set """
        assert self.dataset_builder.training_dataset_path
        assert self.dataset_builder.validation_dataset_path
        assert self.dataset_builder.dataset_index_path

    def test_get_feedable_item(self):
        """ Checks if the .get_feedable_item method works """
        game = Game()
        state_proto = extract_state_proto(game)
        phase_history_proto = extract_phase_history_proto(game)
        possible_orders_proto = extract_possible_orders_proto(game)
        locs = ['PAR', 'MAR', 'BUR']
        kwargs = {'player_seed': 0, 'noise': 0., 'temperature': 0., 'dropout_rate': 0.}
        assert self.dataset_builder.get_feedable_item(locs,
                                                      state_proto,
                                                      'FRANCE',
                                                      phase_history_proto,
                                                      possible_orders_proto,
                                                      **kwargs)

    def test_proto_generation_callable(self):
        """ Checks if the .proto_generation_callable is implemented """
        proto_callable = self.dataset_builder.proto_generation_callable
        assert callable(proto_callable)

    def test_required_features(self):
        """ Checks that features required are present """
        proto_fields = self.dataset_builder.get_proto_fields()
        assert 'request_id' in proto_fields
        assert 'board_state' in proto_fields
        assert 'current_power' in proto_fields
