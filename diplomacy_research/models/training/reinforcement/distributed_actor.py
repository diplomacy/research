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
""" Reinforcement - Distributed Actor
    - Class responsible for playing the 'actor' role in the cluster by generating games
"""
import logging
from tornado import gen
from diplomacy_research.models.training.reinforcement.common import create_advantage
from diplomacy_research.models.training.reinforcement.generation import start_training_process
from diplomacy_research.models.training.reinforcement.memory_buffer import build_memory_buffer
from diplomacy_research.models.training.reinforcement.serving import start_tf_serving_server, check_opening_orders, \
    update_serving, wait_for_version, get_training_config

# Constants
LOGGER = logging.getLogger(__name__)


@gen.coroutine
def start_distributed_actor(trainer):
    """ Starts an actor
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    # Creating advantage function
    create_advantage(trainer)

    # Builds the memory buffer and wait for the cluster to be ready
    build_memory_buffer(trainer)
    trainer.memory_buffer.wait_until_ready()

    # Creating a serving - Training (endpoint only)
    start_tf_serving_server(trainer,
                            force_cpu=True,
                            serving_id=0,
                            config=None,
                            endpoint_only=True)

    # Wait for model to be loaded
    update_serving(trainer, serving_id=0, config=get_training_config(trainer))
    wait_for_version(trainer, serving_id=0, model_name='player')

    # Querying the model to make sure to check if it has been trained
    check_opening_orders(trainer, serving_id=0)

    # Launchs the training process to generate training games forever
    start_training_process(trainer, block=True)
