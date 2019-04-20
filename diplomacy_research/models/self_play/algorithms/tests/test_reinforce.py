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
""" Class to test the REINFORCE Algorithm """
from diplomacy_research.models.policy import order_based
from diplomacy_research.models.policy import token_based
import diplomacy_research.models.policy.order_based.v002_markovian_film as order_based_model
import diplomacy_research.models.policy.token_based.v002_markovian_film as token_based_model
from diplomacy_research.models.self_play.algorithms.reinforce.algorithm import ReinforceAlgorithm, load_args
from diplomacy_research.models.self_play.algorithms.tests.algorithm_test_setup import AlgorithmSetup
from diplomacy_research.utils.process import run_in_separate_process


# ----------- Testable Class --------------
class BaseTestClass(AlgorithmSetup):
    """ Tests the algorithm """
    def __init__(self, model_type, model_family_path, model_version_path):
        """ Constructor """
        AlgorithmSetup.__init__(self, ReinforceAlgorithm, load_args, model_type)
        self._model_family_path = model_family_path
        self._model_version_path = model_version_path

    def get_policy_model(self):
        """ Returns the PolicyModel """
        return self._model_version_path.PolicyModel

    def get_policy_builder(self):
        """ Returns the Policy's BaseDatasetBuilder """
        return self._model_family_path.BaseDatasetBuilder

    def get_policy_adapter(self):
        """ Returns the PolicyAdapter """
        return self._model_family_path.PolicyAdapter

    def get_policy_load_args(self):
        """ Returns the policy args """
        return self._model_version_path.load_args()

# ----------- Launch Scripts --------------
def launch_order_based():
    """ Launches tests for order based """
    test_object = BaseTestClass('order_based', order_based, order_based_model)
    test_object.run_tests()

def launch_token_based():
    """ Launches tests for token based """
    test_object = BaseTestClass('token_based', token_based, token_based_model)
    test_object.run_tests()

# ----------- Tests --------------
def test_order_based():
    """ Runs the order_based test """
    run_in_separate_process(target=launch_order_based, timeout=240)

def test_token_based():
    """ Runs the token_based test """
    run_in_separate_process(target=launch_token_based, timeout=240)
