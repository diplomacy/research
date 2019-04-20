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
""" Class to test the v001_draw_relu DrawModel """
from diplomacy_research.models.policy import order_based
from diplomacy_research.models.policy import token_based
import diplomacy_research.models.policy.order_based.v005_markovian_film_board_align as order_based_model
import diplomacy_research.models.policy.token_based.v005_markovian_film_board_align as token_based_model
from diplomacy_research.models.draw.tests.draw_model_test_setup import DrawModelTestSetup
from diplomacy_research.models.draw.v001_draw_relu import DrawModel, load_args
from diplomacy_research.utils.process import run_in_separate_process


# ----------- Launch Scripts --------------
def launch_order_based():
    """ Launches tests for order based """
    model_family_path = order_based
    model_version_path = order_based_model
    test_object = DrawModelTestSetup(policy_model_ctor=model_version_path.PolicyModel,
                                     value_model_ctor=None,
                                     draw_model_ctor=DrawModel,
                                     dataset_builder=model_family_path.BaseDatasetBuilder(),
                                     adapter_ctor=model_family_path.PolicyAdapter,
                                     load_policy_args=model_version_path.load_args,
                                     load_value_args=None,
                                     load_draw_args=load_args)
    test_object.run_tests()

def launch_token_based():
    """ Launches tests for token based """
    model_family_path = token_based
    model_version_path = token_based_model
    test_object = DrawModelTestSetup(policy_model_ctor=model_version_path.PolicyModel,
                                     value_model_ctor=None,
                                     draw_model_ctor=DrawModel,
                                     dataset_builder=model_family_path.BaseDatasetBuilder(),
                                     adapter_ctor=model_family_path.PolicyAdapter,
                                     load_policy_args=model_version_path.load_args,
                                     load_value_args=None,
                                     load_draw_args=load_args)
    test_object.run_tests()

# ----------- Tests --------------
def test_order_based():
    """ Runs the order_based test """
    run_in_separate_process(target=launch_order_based, timeout=120)

def test_token_based():
    """ Runs the token_based test """
    run_in_separate_process(target=launch_token_based, timeout=120)
