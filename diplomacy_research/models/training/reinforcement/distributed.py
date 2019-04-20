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
""" Reinforcement - Distributed training
    - Class responsible for training a model in a distributed setting
"""
import logging
from tornado import gen
from diplomacy_research.models.training.memory_buffer.memory_buffer import start_redis_server
from diplomacy_research.models.training.reinforcement.common import build_train_server
from diplomacy_research.models.training.reinforcement.distributed_actor import start_distributed_actor
from diplomacy_research.models.training.reinforcement.distributed_learner import start_distributed_learner
from diplomacy_research.models.training.reinforcement.serving import start_tf_serving_server

# Constants
LOGGER = logging.getLogger(__name__)


@gen.coroutine
def start_distributed_training(trainer):
    """ Starts training in distributed mode.
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    # Making sure debug (tensorflow and batch) and profile are not set
    if trainer.flags.debug:
        raise RuntimeError('Debug (TensorFlow) mode is not supported in distributed mode.')
    if trainer.flags.debug_batch:
        raise RuntimeError('Debug (batch) mode is not supported in distributed mode.')
    if trainer.flags.profile:
        raise RuntimeError('Profile mode is not supported in distributed mode.')

    # Dispatching
    # --- Parameter server ---
    if trainer.cluster_config.job_name == 'ps':
        start_parameter_server(trainer)

    # --- Actor ---
    elif trainer.cluster_config.job_name == 'actor':
        yield start_distributed_actor(trainer)

    # --- Learner ---
    elif trainer.cluster_config.job_name == 'learner':
        yield start_distributed_learner(trainer)

    # --- Evaluator ---
    elif trainer.cluster_config.job_name == 'evaluator':
        trainer.process_pool.shutdown()                                                     # Not yet implemented
        trainer.process_pool = None

    # --- Serving ---
    elif trainer.cluster_config.job_name == 'serving':
        start_tf_serving_server(trainer, force_cpu=False, serving_id=0, config=None)        # Config set by actors

    # --- Redis ---
    elif trainer.cluster_config.job_name == 'redis':
        start_redis_server(trainer)
    else:
        raise RuntimeError('Invalid configuration detected.')

def start_parameter_server(trainer):
    """ Starts a parameter server
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    build_train_server(trainer)
    LOGGER.info('Parameter server is now ready ...')
    trainer.server.join()
