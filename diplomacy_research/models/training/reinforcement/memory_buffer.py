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
""" Reinforcement Learning - Memory Buffer
    - Class responsible for for interacting with the memory buffer
"""
import logging
from tornado import gen
from diplomacy_research.models.training.memory_buffer import MemoryBuffer
from diplomacy_research.models.training.memory_buffer import priority_replay

# Constants
LOGGER = logging.getLogger(__name__)


def build_memory_buffer(trainer):
    """ Builds the memory buffer and connects to it
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    # Creating memory buffer
    trainer.memory_buffer = MemoryBuffer(trainer.cluster_config, trainer.hparams)

    # If debug (batch), we need to flush the Redis cache
    if trainer.flags.debug_batch and (not trainer.cluster_config or trainer.cluster_config.is_chief):
        trainer.memory_buffer.clear()

@gen.coroutine
def update_priorities(trainer, learned_games_proto, replay_samples):
    """ Update the priorities of games and samples in the memory buffer
        :param trainer: A reinforcement learning trainer instance.
        :param learned_games_proto: A list of `SavedGameProto` representing played games by the RL agents
        :param replay_samples: A list of ReplaySamples from the memory buffer.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    if not trainer.algorithm_constructor.can_do_experience_replay or not trainer.flags.experience_replay:
        return
    if learned_games_proto:
        new_priorities = yield trainer.algorithm.get_priorities(learned_games_proto, trainer.advantage_fn)
        priority_replay.update_priorities(trainer.memory_buffer, new_priorities, first_update=True)
    if replay_samples:
        new_priorities = yield trainer.algorithm.get_new_priorities(replay_samples, trainer.advantage_fn)
        priority_replay.update_priorities(trainer.memory_buffer, new_priorities)
        priority_replay.trim_buffer(trainer.memory_buffer)
