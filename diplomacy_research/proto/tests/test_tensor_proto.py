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
""" Tests the make_tensor_proto and make_ndarray functions """
import numpy as np
from diplomacy_research.utils.proto import make_tensor_proto, proto_to_bytes, make_ndarray

def test_proto_1():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto(bytes('', 'utf-8'), dtype=np.object, shape=[1])
    tensor_2 = tf.make_tensor_proto(bytes('', 'utf-8'), dtype=tf.string, shape=[1])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_2():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto([bytes('', 'utf-8')], dtype=np.object, shape=[1])
    tensor_2 = tf.make_tensor_proto([bytes('', 'utf-8')], dtype=tf.string, shape=[1])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_3():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto([], dtype=np.int32, shape=[1, 0])
    tensor_2 = tf.make_tensor_proto([], dtype=tf.int32, shape=[1, 0])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_4():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto([], dtype=np.float32, shape=[1, 0])
    tensor_2 = tf.make_tensor_proto([], dtype=tf.float32, shape=[1, 0])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_5():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto(0, dtype=np.int32, shape=[1, 50, 83])
    tensor_2 = tf.make_tensor_proto(0, dtype=tf.int32, shape=[1, 50, 83])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_6():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    tensor_1 = make_tensor_proto(0, dtype=np.float32, shape=[1, 0])
    tensor_2 = tf.make_tensor_proto(0, dtype=tf.float32, shape=[1, 0])
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype

def test_proto_7():
    """ Tests the make_tensor_proto and make_nd_array function """
    from diplomacy_research.utils.tensorflow import tf
    random_tensor = np.random.rand(15, 25)
    tensor_1 = make_tensor_proto(random_tensor, dtype=np.float32)
    tensor_2 = tf.make_tensor_proto(random_tensor, dtype=tf.float32)
    array_1 = tf.make_ndarray(tensor_1)
    array_2 = make_ndarray(tensor_2)
    assert proto_to_bytes(tensor_1) == proto_to_bytes(tensor_2)
    assert array_1.tostring() == array_2.tostring()
    assert array_1.dtype == array_2.dtype
