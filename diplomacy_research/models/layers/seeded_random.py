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
""" Seeded Random
    - Contains the seeded_random op
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import array_ops
from diplomacy_research.utils.tensorflow import gen_array_ops
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import load_so_by_name
from diplomacy_research.utils.tensorflow import math_ops
from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import random_seed
from diplomacy_research.utils.tensorflow import gen_user_ops

# Finding the correct implementation of seeded_random
SEEDED_RANDOM_SO = gen_user_ops if hasattr(gen_user_ops, 'seeded_random') else load_so_by_name('seeded_random')


def seeded_random(seeds, offset, shape, dtype, seed=None, name=None):
    """ Outputs random values from a uniform distribution.
        The random values are deterministic given a seed.

        :param seeds: A vector of seeds (Size: [batch,]) - If 0, defaults to seed attr, then graph seed, then random.
        :param offset: Integer to add to the seed to get a deterministic mask.
        :param shape: The shape required for each seed (e.g. [3, 5] with a batch of 10 will return [10, 3, 5]).
        :param dtype: The type of the output. `float16`, `float32`, `float64`
        :param seed: A Python integer. Used to create a default seed for the operation.
        :param name: A name for the operation (optional).
        :return: A tensor of the specified shape filled with deterministic random values.
    """
    if dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64):
        raise ValueError('Invalid dtype %r' % dtype)
    with ops.name_scope(name, 'seeded_random', [shape]):
        seeds = ops.convert_to_tensor(seeds, dtype=dtypes.int32, name='seeds')
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32, name='shape')
        offset = ops.convert_to_tensor(offset, dtype=dtypes.int32, name='offset')
        size = math_ops.reduce_prod(shape)
        graph_seed, op_seed = random_seed.get_seed(seed)
        matrix_output = SEEDED_RANDOM_SO.seeded_random(seeds, offset, size, seed=graph_seed, seed2=op_seed)
        output = gen_array_ops.reshape(matrix_output, array_ops.concat([(-1,), shape], axis=0))
        return math_ops.cast(output, dtype)
