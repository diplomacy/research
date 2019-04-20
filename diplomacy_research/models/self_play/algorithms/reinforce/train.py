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
""" REINFORCE Algorithm
    - Trains a model with the REINFORCE algorithm
"""
import importlib
import logging
import os
from diplomacy_research.models.self_play.algorithms.reinforce import Algorithm, load_args
from diplomacy_research.models.training.reinforcement import ReinforcementTrainer
from diplomacy_research.utils.cluster import start_distributed_training
from diplomacy_research.utils.cluster_config.reinforcement import get_cluster_config
from diplomacy_research.utils.model import load_dynamically_from_model_id, find_model_type
from diplomacy_research.utils.model import parse_args_into_flags, run_app
from diplomacy_research.settings import ROOT_DIR

# Constants
LOGGER = logging.getLogger('diplomacy_research.models.self_play.algorithms.reinforce.train')

def main(_):
    """ Trains the RL model """
    def start_rl_training(cluster_config, process_pool):
        """ Callable fn to start training """
        ReinforcementTrainer(policy_constructor=PolicyModel,
                             value_constructor=None,
                             draw_constructor=DrawModel,
                             dataset_builder_constructor=BaseDatasetBuilder,
                             adapter_constructor=PolicyAdapter,
                             algorithm_constructor=Algorithm,
                             reward_fn=reward_fn,
                             flags=FLAGS,
                             cluster_config=cluster_config,
                             process_pool=process_pool).start()

    # Detecting if we need to start regular or distributed training
    start_distributed_training(callable_fn=start_rl_training,
                               flags=FLAGS,
                               get_cluster_config_fn=get_cluster_config,
                               with_process_pool=True)

if __name__ == '__main__':

    # Finding the root dir of the model type
    base_dir, base_import_path = find_model_type()                                                                      # pylint: disable=invalid-name

    # Loading model_constructor and args
    PolicyModel, load_policy_args = load_dynamically_from_model_id(['PolicyModel', 'load_args'],                        # pylint: disable=invalid-name
                                                                   arg_name='model_id',
                                                                   base_dir=base_dir)

    # Detecting DrawModel, and load_args
    DrawModel, load_draw_args = load_dynamically_from_model_id(['DrawModel', 'load_args'],                              # pylint: disable=invalid-name
                                                               arg_name='draw_model_id',
                                                               base_dir=os.path.join(ROOT_DIR, 'models', 'draw'),
                                                               on_error='ignore')

    # REINFORCE does not use a value model
    LOGGER.info('REINFORCE does not a use a parameterized value function. Ignoring "value_model_id".')

    # Loading args
    ARGS = load_policy_args() \
           + (load_draw_args() if load_draw_args is not None else []) \
           + load_args()
    FLAGS = parse_args_into_flags(ARGS)

    # Loading reward functions, policy adapter, and base dataset builder
    PolicyAdapter = importlib.import_module('%s' % base_import_path).PolicyAdapter                                      # pylint: disable=invalid-name
    BaseDatasetBuilder = importlib.import_module('%s' % base_import_path).BaseDatasetBuilder                            # pylint: disable=invalid-name
    reward_fn = getattr(importlib.import_module('diplomacy_research.models.self_play.reward_functions'),                # pylint: disable=invalid-name
                        FLAGS.reward_fn)()

    # Running
    run_app()
