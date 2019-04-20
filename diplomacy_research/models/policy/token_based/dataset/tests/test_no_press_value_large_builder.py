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
""" Runs tests for the NoPressValue (Large) Dataset Builder """
from diplomacy_research.models.policy.tests.policy_builder_test_setup import PolicyBuilderTestSetup
from diplomacy_research.models.policy.token_based.dataset.no_press_value_large import DatasetBuilder
from diplomacy_research.utils.process import run_in_separate_process

def launch():
    """ Launches the tests """
    testable_class = PolicyBuilderTestSetup(DatasetBuilder())
    testable_class.run_tests()

def test_run():
    """ Runs the test """
    run_in_separate_process(target=launch, timeout=60)
