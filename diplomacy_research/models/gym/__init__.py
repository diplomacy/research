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
""" OpenAI Gym integration modules """
import warnings
from gym.envs.registration import register
from diplomacy_research.models.gym.wrappers import AutoDraw, LimitNumberYears, LoopDetection, SetInitialState, \
    AssignPlayers, RandomizePlayers, SetPlayerSeed, SaveGame

# Ignore specific warnings
warnings.filterwarnings('ignore', message='Parameters to load are deprecated')

register(
    id='DiplomacyEnv-v0',
    entry_point='diplomacy_research.models.gym.environment:DiplomacyEnv')
