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
""" Class to test the GAE Advantage """
from diplomacy_research.models.self_play.reward_functions import DEFAULT_PENALTY
from diplomacy_research.models.self_play.advantages.gae import GAE
from diplomacy_research.models.self_play.advantages.tests.advantage_test_setup import AdvantageSetup
from diplomacy_research.utils.process import run_in_separate_process

class GAETestClass(AdvantageSetup):
    """ Tests the advantage """

    def create_advantage(self):
        """ Creates the advantage object """
        self.advantage = GAE(lambda_=0.9, gamma=0.99, penalty_per_phase=DEFAULT_PENALTY)

def launch():
    """ Launches the test """
    test_object = GAETestClass(model_type='order_based')
    test_object.run_tests()

def test_run():
    """ Runs the test """
    run_in_separate_process(target=launch, timeout=240)
