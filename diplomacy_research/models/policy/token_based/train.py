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
""" Policy model (token_based)
    - Contains the training methods for the policy model
"""
import os
from diplomacy_research.models.policy.token_based import PolicyAdapter
from diplomacy_research.models.training.supervised import SupervisedTrainer
from diplomacy_research.utils.cluster import start_distributed_training
from diplomacy_research.utils.cluster_config.supervised import get_cluster_config
from diplomacy_research.utils.model import load_dynamically_from_model_id, load_dataset_builder
from diplomacy_research.utils.model import parse_args_into_flags, run_app
from diplomacy_research.settings import ROOT_DIR

def main(_):
    """ Trains the policy model """
    def start_supervised_training(cluster_config, _):
        """ Callable fn to start training """
        SupervisedTrainer(policy_constructor=PolicyModel,
                          value_constructor=ValueModel,
                          draw_constructor=DrawModel,
                          dataset_builder=DatasetBuilder(),
                          adapter_constructor=PolicyAdapter,
                          cluster_config=cluster_config,
                          flags=FLAGS).start()

    # Detecting if we need to start regular or distributed training
    start_distributed_training(callable_fn=start_supervised_training,
                               flags=FLAGS,
                               get_cluster_config_fn=get_cluster_config,
                               with_process_pool=False)

if __name__ == '__main__':

    # Detecting PolicyModel, and load_args
    PolicyModel, load_args = load_dynamically_from_model_id(['PolicyModel', 'load_args'], arg_name='model_id')          # pylint: disable=invalid-name

    # Detecting ValueModel, and load_args
    ValueModel, load_value_args = load_dynamically_from_model_id(['ValueModel', 'load_args'],                           # pylint: disable=invalid-name
                                                                 arg_name='value_model_id',
                                                                 base_dir=os.path.join(ROOT_DIR, 'models', 'value'),
                                                                 on_error='ignore')

    # Detecting DrawModel, and load_args
    DrawModel, load_draw_args = load_dynamically_from_model_id(['DrawModel', 'load_args'],                              # pylint: disable=invalid-name
                                                               arg_name='draw_model_id',
                                                               base_dir=os.path.join(ROOT_DIR, 'models', 'draw'),
                                                               on_error='ignore')

    # Loading args
    ARGS = load_args() \
           + (load_value_args() if load_value_args is not None else []) \
           + (load_draw_args() if load_draw_args is not None else [])
    FLAGS = parse_args_into_flags(ARGS)

    # Loading DatasetBuilder
    DatasetBuilder = load_dataset_builder(FLAGS.dataset)                                                                # pylint: disable=invalid-name

    # Running
    run_app()
