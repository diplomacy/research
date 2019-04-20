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
""" Runs tests for the current model and adapter """
from diplomacy_research.models.policy.tests.policy_adapter_test_setup import PolicyAdapterTestSetup
from diplomacy_research.models.policy.token_based import PolicyAdapter, BaseDatasetBuilder
from diplomacy_research.models.policy.token_based.v002_markovian_film import PolicyModel, load_args
from diplomacy_research.models.value.v001_val_relu_7 import ValueModel, load_args as load_value_args
from diplomacy_research.models.self_play.algorithms.a2c import Algorithm as A2CAlgo, load_args as a2c_args
from diplomacy_research.models.self_play.algorithms.ppo import Algorithm as PPOAlgo, load_args as ppo_args
from diplomacy_research.models.self_play.algorithms.reinforce import Algorithm as ReinforceAlgo,\
    load_args as reinforce_args
from diplomacy_research.models.self_play.algorithms.tests.algorithm_test_setup import AlgorithmSetup
from diplomacy_research.utils.process import run_in_separate_process


# ----------- Testable Class --------------
class BaseTestClass(AlgorithmSetup):
    """ Tests the algorithm """
    def __init__(self, algorithm_ctor, algo_load_args):
        """ Constructor """
        AlgorithmSetup.__init__(self, algorithm_ctor, algo_load_args, 'token_based')

    def get_policy_model(self):
        """ Returns the PolicyModel """
        return PolicyModel

    def get_policy_builder(self):
        """ Returns the Policy's BaseDatasetBuilder """
        return BaseDatasetBuilder

    def get_policy_adapter(self):
        """ Returns the PolicyAdapter """
        return PolicyAdapter

    def get_policy_load_args(self):
        """ Returns the policy args """
        return load_args()

# ----------- Launch Scripts --------------
def launch_a2c():
    """ Launches tests for a2c """
    test_object = BaseTestClass(A2CAlgo, a2c_args)
    test_object.run_tests()

def launch_ppo():
    """ Launches tests for ppo """
    test_object = BaseTestClass(PPOAlgo, ppo_args)
    test_object.run_tests()

def launch_reinforce():
    """ Launches tests for reinforce """
    test_object = BaseTestClass(ReinforceAlgo, reinforce_args)
    test_object.run_tests()

def launch_adapter():
    """ Launches the tests """
    testable_class = PolicyAdapterTestSetup(policy_model_ctor=PolicyModel,
                                            value_model_ctor=ValueModel,
                                            draw_model_ctor=None,
                                            dataset_builder=BaseDatasetBuilder(),
                                            policy_adapter_ctor=PolicyAdapter,
                                            load_policy_args=load_args,
                                            load_value_args=load_value_args,
                                            load_draw_args=None,
                                            strict=False)
    testable_class.run_tests()

# ----------- Tests --------------
def test_run_a2c():
    """ Runs the a2c test """
    run_in_separate_process(target=launch_a2c, timeout=240)

def test_run_ppo():
    """ Runs the ppo test """
    run_in_separate_process(target=launch_ppo, timeout=240)

def test_run_reinforce():
    """ Runs the reinforce test """
    run_in_separate_process(target=launch_reinforce, timeout=240)

def test_run_adapter():
    """ Runs the adapter test """
    run_in_separate_process(target=launch_adapter, timeout=240)
