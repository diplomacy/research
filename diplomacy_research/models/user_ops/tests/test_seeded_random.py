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
""" Tests for seeded_random op """
import numpy as np
from diplomacy_research.utils.process import run_in_separate_process

def run_seeded_random():
    """ Run tests for seeded_random """
    from diplomacy_research.utils.tensorflow import tf, load_so_by_name
    seeded_random_so = load_so_by_name('seeded_random')
    sess = tf.InteractiveSession()
    seeds = tf.placeholder(shape=[None], dtype=tf.int32)

    # Static shape - With graph seed provided
    op_1 = seeded_random_so.seeded_random(seeds=seeds, offset=1, size=100, seed=75, seed2=0)
    output_1 = sess.run(op_1, feed_dict={seeds: [12345, 0, 12345, 0]})
    output_2 = sess.run(op_1, feed_dict={seeds: [12345, 0, 12345, 0]})
    assert op_1.shape.as_list() == [None, 100]
    assert output_1.shape == (4, 100)
    assert output_2.shape == (4, 100)
    assert np.allclose(output_1[0], output_1[2])
    assert np.allclose(output_1[0], output_2[0])
    assert np.allclose(output_1[1], output_1[3])            # Since a seed was provided
    assert np.allclose(output_2[1], output_2[3])            # Since a seed was provided
    assert np.allclose(output_1[1], output_2[1])            # Since a seed was provided
    assert np.allclose(output_1[3], output_2[3])            # Since a seed was provided
    assert np.all(output_1[0] != output_1[1])

    # Dynamic shape - No seed
    shape = tf.placeholder(shape=(), dtype=tf.int32)
    op_2 = seeded_random_so.seeded_random(seeds=seeds, offset=2, size=shape, seed=0, seed2=0)
    output_1 = sess.run(op_2, feed_dict={seeds: [12345, 0, 12345, 0], shape: 200})
    output_2 = sess.run(op_2, feed_dict={seeds: [12345, 0, 12345, 0], shape: 200})
    assert op_2.shape.as_list() == [None, None]
    assert output_1.shape == (4, 200)
    assert output_2.shape == (4, 200)
    assert np.allclose(output_1[0], output_1[2])
    assert np.allclose(output_1[0], output_2[0])
    assert np.all(output_1[1] != output_1[3])
    assert np.all(output_2[1] != output_2[3])
    assert np.all(output_1[1] != output_2[1])
    assert np.all(output_1[3] != output_2[3])

def test_seeded_random():
    """ Tests for the seeded random op """
    run_in_separate_process(target=run_seeded_random, timeout=30)
