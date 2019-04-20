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
""" Protocol Buffer utilities """
import zlib
from google.protobuf.pyext._message import MessageMapContainer                                                          # pylint: disable=no-name-in-module
from google.protobuf.message import DecodeError
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
import numpy as np
from diplomacy_research.proto.diplomacy_proto import common_pb2, game_pb2
from diplomacy_research.proto.diplomacy_tensorflow.core.framework.tensor_pb2 import TensorProto
from diplomacy_research.proto.diplomacy_tensorflow.core.framework import types_pb2
from diplomacy_research.proto.diplomacy_tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

# Constants
TF_DTYPES = {(types_pb2.DT_FLOAT, np.float32),
             (types_pb2.DT_DOUBLE, np.float64),
             (types_pb2.DT_INT8, np.int8),
             (types_pb2.DT_INT16, np.int16),
             (types_pb2.DT_INT32, np.int32),
             (types_pb2.DT_INT64, np.int64),
             (types_pb2.DT_UINT8, np.uint8),
             (types_pb2.DT_UINT16, np.uint16),
             (types_pb2.DT_UINT32, np.uint32),
             (types_pb2.DT_UINT64, np.uint64),
             (types_pb2.DT_STRING, np.object),
             (types_pb2.DT_BOOL, np.bool)}
TF_FIELDS = {(np.float32, 'float_val'),
             (np.float64, 'double_val'),
             (np.int8, 'int_val'),
             (np.int16, 'int_val'),
             (np.int32, 'int_val'),
             (np.int64, 'int64_val'),
             (np.uint8, 'int_val'),
             (np.uint16, 'int_val'),
             (np.uint32, 'uint32_val'),
             (np.uint64, 'uint64_val'),
             (np.object, 'string_val'),
             (np.bool, 'bool_val')}
NP_TO_FIELD = {np.dtype(np_dtype).name: field_name for np_dtype, field_name in TF_FIELDS}
FIELD_TO_NP = {field_name: np_dtype  for np_dtype, field_name in TF_FIELDS}
NP_TO_DTYPE = {np.dtype(np_dtype).name: dtype for dtype, np_dtype in TF_DTYPES}
DTYPE_TO_NP = {dtype: np_dtype for dtype, np_dtype in TF_DTYPES}
INT64_FIELDS = {'time_sent'}


def proto_to_dict(src_proto, include_default=True):
    """ Converts a proto object to a dictionary
        :param src_proto: The proto object to convert
        :param include_default: Boolean. Also include default value for repeated, map, and singular fields
        :return: A dictionary representation of the proto buffer
    """
    # MessageMapContainer is MapStringList.value - Recreating a full MapStringList
    if src_proto.__class__ == MessageMapContainer:
        map_container = src_proto
        src_proto = common_pb2.MapStringList(value=src_proto)
        src_proto.value.MergeFrom(map_container)

    proto_constructor = src_proto.__class__
    if proto_constructor not in SCHEMA:
        raise ValueError('Unknown proto type %s' % proto_constructor)

    # Building dictionary
    dst_dict = {}
    proto_schema = SCHEMA[proto_constructor]
    for attr_name in proto_schema:
        if not include_default and not src_proto.IsInitialized(attr_name):
            continue
        getter, _ = proto_schema[attr_name]
        getter(src_proto, dst_dict, attr_name)

    # Removing 'value' wrapper for MapStringList
    if src_proto.__class__ == common_pb2.MapStringList:
        dst_dict = dst_dict['value']
    return dst_dict

def dict_to_proto(src_dict, proto_constructor, strict=False, dst_proto=None):
    """ Converts a python dictionary to its proto representation
        :param src_dict: The python dictionary to convert
        :param proto_constructor: The constructor to build the proto object
        :param strict: If strict, will throw an error on unknown fields.
        :param dst_proto: Optional. Proto to use for assignment, otherwise a new proto is created.
    """
    if proto_constructor not in SCHEMA:
        raise ValueError('Unknown proto type %s' % proto_constructor)

    # Creating empty proto if not provided
    if dst_proto is None:
        dst_proto = proto_constructor()
    else:
        assert dst_proto.__class__ == proto_constructor, 'The proto constructor does not match the instantiated object'

    # Returning early if src_dict is empty or None
    if not src_dict:
        return dst_proto

    # For MapStringList - Wrapping to convert properly
    if proto_constructor == common_pb2.MapStringList:
        src_dict = {'value': src_dict}

    # Setting attributes
    proto_schema = SCHEMA[proto_constructor]
    for attr_name in src_dict:
        if attr_name not in proto_schema:
            assert not strict, 'Attribute %s is not in schema.' % attr_name
            continue
        _, setter = proto_schema[attr_name]
        setter(src_dict, dst_proto, attr_name)

    # For MapStringList - Unwrapping before returning
    if proto_constructor == common_pb2.MapStringList:
        dst_proto = dst_proto.value
    return dst_proto

def proto_to_zlib(proto):
    """ Converts a proto object to a compressed byte array """
    bytes_array = proto_to_bytes(proto)
    return zlib.compress(bytes_array)

def zlib_to_proto(zlib_data, proto_constructor):
    """ Converts a compressed byte array  """
    bytes_array = zlib.decompress(zlib_data)
    return bytes_to_proto(bytes_array, proto_constructor)

def zlib_to_bytes(zlib_data):
    """ Decompress a compressed byte array """
    return zlib.decompress(zlib_data)

def bytes_to_zlib(bytes_data):
    """ Compress a bytes array """
    return zlib.compress(bytes_data)

def proto_to_bytes(proto):
    """ Converts a proto object to a bytes array """
    return proto.SerializeToString()

def bytes_to_proto(bytes_data, proto_constructor):
    """ Converts a bytes array back to a proto object """
    proto = proto_constructor()
    proto.ParseFromString(bytes_data)
    return proto

def write_proto_to_file(file_handle, proto, compressed):
    """ Writes a proto to the file handle
        :param file_handle: The file to which to write the proto
        :param proto: The Protocol Buffer object to write
        :param compressed: Boolean. If true, writes a compressed buffer, otherwise write a serialized buffer
    """
    serialized = proto_to_zlib(proto) if compressed else proto_to_bytes(proto)
    file_handle.write(_VarintBytes(len(serialized)))
    file_handle.write(serialized)

def write_bytes_to_file(file_handle, bytes_object):
    """ Writes a proto in bytes format to the file handle
        :param file_handle: The file to which to write the proto in bytes format
        :param bytes_object: The proto object in bytes format
    """
    serialized = bytes_object
    file_handle.write(_VarintBytes(len(serialized)))
    file_handle.write(serialized)

def read_next_proto(proto_constructor, file_handle, compressed):
    """ Reads a proto (and the next position) from the file at a specific position
        :param proto_constructor: The proto object constructor
        :param file_handle: The file handle created by open()
        :param compressed: Boolean. Indicates if the content is compressed
        :return: The next proto object in the file

        Note: A value of None will be returned if an error is encountered.
    """
    try:
        data = file_handle.read(4)
        size, offset = _DecodeVarint32(data, 0)
        serialized = (data + file_handle.read(size + offset - 4))[offset:]
        decoder = zlib_to_proto if compressed else bytes_to_proto
        proto = decoder(serialized, proto_constructor)
    except (DecodeError, IndexError):
        return None
    return proto

def read_next_bytes(file_handle):
    """ Reads a bytes array (and the next position) from the file at a specific position
        :param file_handle: The file handle created by open()
        :return: The next bytes array in the file

        Note: A value of None will be returned if an error is encountered.
    """
    try:
        data = file_handle.read(4)
        size, offset = _DecodeVarint32(data, 0)
        serialized = (data + file_handle.read(size + offset - 4))[offset:]
    except (DecodeError, IndexError):
        return None
    return serialized

# ----------------------------
# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_util.py
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False):
    """ Create a TensorProto
        :param values: Numpy array to put in the TensorProto
        :param dtype: Optional numpy dtype.
        :param shape: List of integers representing the dimensions of tensor.
        :param verify_shape: Boolean that enables verification of a shape of values.
        :return: A `TensorProto`
    """
    if isinstance(values, TensorProto):
        return values
    if isinstance(values, (np.ndarray, np.generic)):
        nparray = values
        if dtype is not None and nparray.dtype != dtype:
            nparray = nparray.astype(dtype)
    elif shape is not None and np.prod(shape, dtype=np.int64) == 0:
        nparray = np.empty(shape, dtype=dtype)
    else:
        nparray = np.array(values, dtype=dtype)

    # Getting the dtype of the numpy array
    numpy_dtype = dtype or nparray.dtype

    # If shape is not given, get the shape from the numpy array.
    if shape is None:
        shape = nparray.shape
        is_same_size = True
        shape_size = nparray.size
    else:
        shape = [int(dim) for dim in shape]
        shape_size = np.prod(shape, dtype=np.int64)
        is_same_size = shape_size == nparray.size

        if verify_shape:
            if nparray.shape != tuple(shape):
                raise TypeError('Expected Tensor\'s shape: %s, got %s.' % (tuple(shape), nparray.shape))

        if nparray.size > shape_size:
            raise ValueError('Too many elements provided. Wanted at most %d, got %d.' % (int(shape_size), nparray.size))

    tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=-1 if dim is None else dim) for dim in shape])
    tensor_proto = TensorProto(dtype=NP_TO_DTYPE[np.dtype(numpy_dtype).name], tensor_shape=tensor_shape)

    if is_same_size \
            and numpy_dtype in [np.float32, np.float64, np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint32,
                                np.uint64] \
            and shape_size > 1:
        if nparray.size * nparray.itemsize >= (1 << 31):
            raise ValueError('Cannot create a tensor proto whose content is larger than 2GB.')
        tensor_proto.tensor_content = nparray.tostring()
        return tensor_proto

    if numpy_dtype == np.object and isinstance(values, (bytes, list, np.ndarray)):
        if isinstance(values, bytes):
            str_values = [values]
        else:
            str_values = [x.encode('utf-8') if isinstance(x, str) else x for x in values]
        tensor_proto.string_val.extend(str_values)                                      # pylint: disable=no-member
        return tensor_proto

    if shape_size == 0:
        return tensor_proto

    # TensorFlow expects C order (a.k.a., eigen row major).
    proto_values = nparray.ravel()
    field_name = NP_TO_FIELD.get(np.dtype(numpy_dtype).name, None)
    if field_name is None:
        raise TypeError('Element type not supported in TensorProto: %s' % numpy_dtype.name)
    getattr(tensor_proto, field_name).extend([np.asscalar(x) for x in proto_values])
    return tensor_proto

def make_ndarray(tensor_proto):
    """ Converts a TensorProto to a numpy array
        :param tensor_proto: A `TensorProto`
        :return: A numpy array with the tensor contents
        :type tensor_proto: TensorProto
    """
    shape = [dim.size for dim in tensor_proto.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    np_dtype = DTYPE_TO_NP[tensor_proto.dtype]

    if tensor_proto.tensor_content:
        return np.frombuffer(tensor_proto.tensor_content, dtype=np_dtype).copy().reshape(shape)

    field_name = NP_TO_FIELD.get(np.dtype(np_dtype).name, None)
    if field_name is None:
        raise TypeError('Unsupported tensor type: %s' % np_dtype)

    if len(getattr(tensor_proto, field_name)) == 1:
        return np.repeat(np.array(getattr(tensor_proto, field_name)[0], dtype=np_dtype), num_elements).reshape(shape)
    return np.fromiter(getattr(tensor_proto, field_name), dtype=np_dtype).reshape(shape)


# ------------------------------------------------
#               Utility methods
# ------------------------------------------------
def _get_bool(src_proto, dst_dict, attr_name):
    """ Getter for a simple boolean """
    dst_dict[attr_name] = bool(getattr(src_proto, attr_name))

def _get_string(src_proto, dst_dict, attr_name):
    """ Getter for a string """
    dst_dict[attr_name] = str(getattr(src_proto, attr_name))

def _get_int(src_proto, dst_dict, attr_name):
    """ Getter for an integer """
    dst_dict[attr_name] = int(getattr(src_proto, attr_name))

def _get_float(src_proto, dst_dict, attr_name):
    """ Getter for a float """
    dst_dict[attr_name] = float(getattr(src_proto, attr_name))

def _get_proto(src_proto, dst_dict, attr_name):
    """ Getter for a proto message """
    dst_dict[attr_name] = proto_to_dict(getattr(src_proto, attr_name))

def _get_repeated_string(src_proto, dst_dict, attr_name):
    """ Getter for a list of strings """
    dst_dict[attr_name] = [str(item) for item in getattr(src_proto, attr_name)]

def _get_repeated_int(src_proto, dst_dict, attr_name):
    """ Getter for a list of integers """
    dst_dict[attr_name] = [int(item) for item in getattr(src_proto, attr_name)]

def _get_repeated_float(src_proto, dst_dict, attr_name):
    """ Getter for a list of floats """
    dst_dict[attr_name] = [float(item) for item in getattr(src_proto, attr_name)]

def _get_repeated_proto(src_proto, dst_dict, attr_name):
    """ Getter for a list of proto messages """
    dst_dict[attr_name] = [proto_to_dict(item) for item in getattr(src_proto, attr_name)]

def _get_map_int(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary of integers """
    dst_dict[attr_name] = {key: int(value) for key, value in getattr(src_proto, attr_name).items()}

def _get_map_float(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary of integers """
    dst_dict[attr_name] = {key: float(value) for key, value in getattr(src_proto, attr_name).items()}

def _get_map_proto(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary having a proto_message as value """
    dst_dict[attr_name] = {key: proto_to_dict(getattr(src_proto, attr_name)[key])
                           for key in getattr(src_proto, attr_name)}

def _get_map_proto_no_default(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary having a proto_message as value (but skip keys with blank values) """
    dst_dict[attr_name] = {key: proto_to_dict(getattr(src_proto, attr_name)[key])
                           for key in getattr(src_proto, attr_name)
                           if getattr(src_proto, attr_name)[key].ByteSize()}

def _get_map_string_list(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary having a list of string as values """
    dst_dict[attr_name] = {}
    for key in getattr(src_proto, attr_name):
        dst_dict[attr_name][key] = [str(item) for item in getattr(src_proto, attr_name)[key].value]

def _get_map_float_list(src_proto, dst_dict, attr_name):
    """ Getter for a dictionary having a list of float as values """
    dst_dict[attr_name] = {}
    for key in getattr(src_proto, attr_name):
        dst_dict[attr_name][key] = [float(item) for item in getattr(src_proto, attr_name)[key].value]


def _set_bool(src_dict, dst_proto, attr_name):
    """ Setter for a simple boolean """
    setattr(dst_proto, attr_name, bool(src_dict[attr_name]))

def _set_string(src_dict, dst_proto, attr_name):
    """ Setter for a simple string """
    setattr(dst_proto, attr_name, str(src_dict[attr_name]))

def _set_int(src_dict, dst_proto, attr_name):
    """ Setter for a simple integer """
    setattr(dst_proto, attr_name, int(src_dict[attr_name]))

def _set_float(src_dict, dst_proto, attr_name):
    """ Setter for a simple float """
    setattr(dst_proto, attr_name, float(src_dict[attr_name]))

def _set_state_proto(src_dict, dst_proto, attr_name):
    """ Setter for a proto message """
    dict_to_proto(src_dict[attr_name], game_pb2.State, dst_proto=getattr(dst_proto, attr_name))

def _set_repeated_string(src_dict, dst_proto, attr_name):
    """ Setter for a list of string """
    getattr(dst_proto, attr_name).extend([str(item) for item in src_dict[attr_name]])

def _set_repeated_int(src_dict, dst_proto, attr_name):
    """ Setter for a list of integers """
    getattr(dst_proto, attr_name).extend([int(item) for item in src_dict[attr_name]])

def _set_repeated_float(src_dict, dst_proto, attr_name):
    """ Setter for a list of floats """
    getattr(dst_proto, attr_name).extend([float(item) for item in src_dict[attr_name]])

def _set_repeated_phase_history_proto(src_dict, dst_proto, attr_name):
    """ Setter for a list of phase history protos  """
    getattr(dst_proto, attr_name).extend([dict_to_proto(item, game_pb2.PhaseHistory) for item in src_dict[attr_name]])

def _set_map_int(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary of integers """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        getattr(dst_proto, attr_name)[key] = int(value)

def _set_map_float(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary of floats """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        getattr(dst_proto, attr_name)[key] = float(value)

def _set_map_builds_proto(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary with a proto message as value """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        dict_to_proto(value, game_pb2.State.Builds, dst_proto=getattr(dst_proto, attr_name)[key])

def _set_map_keyword_args_proto(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary with a proto message as value """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        dict_to_proto(value, game_pb2.SavedGame.KeywordArgs, dst_proto=getattr(dst_proto, attr_name)[key])

def _set_map_policy_details_proto(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary with a proto message as value """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        dict_to_proto(value, game_pb2.PhaseHistory.PolicyDetails, dst_proto=getattr(dst_proto, attr_name)[key])

def _set_map_string_list(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary with a list of string as values """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        getattr(dst_proto, attr_name)[key].value.extend([str(item) for item in value])

def _set_map_float_list(src_dict, dst_proto, attr_name):
    """ Setter for a dictionary with a list of floats as value """
    if src_dict[attr_name] is None:
        return
    for key, value in src_dict[attr_name].items():
        if value is None:
            continue
        getattr(dst_proto, attr_name)[key].value.extend([float(item) for item in value])


# Syntax: getter (proto to dict), setter (dict to proto)
# getter: (src_proto, dst_dict, parent_message, attr_name)
# setter: (src_dict, dst_proto, parent_message, attr_name)
SCHEMA = {
    common_pb2.MapStringList: {'value': (_get_map_string_list, _set_map_string_list)},
    game_pb2.Message: {'sender': (_get_string, _set_string),
                       'recipient': (_get_string, _set_string),
                       'time_sent': (_get_int, _set_int),
                       'phase': (_get_string, _set_string),
                       'message': (_get_string, _set_string),
                       'tokens': (_get_repeated_int, _set_repeated_int)},
    game_pb2.State: {'game_id': (_get_string, _set_string),
                     'name': (_get_string, _set_string),
                     'map': (_get_string, _set_string),
                     'zobrist_hash': (_get_string, _set_string),
                     'rules': (_get_repeated_string, _set_repeated_string),
                     'units': (_get_map_string_list, _set_map_string_list),
                     'centers': (_get_map_string_list, _set_map_string_list),
                     'homes': (_get_map_string_list, _set_map_string_list),
                     'influence': (_get_map_string_list, _set_map_string_list),
                     'civil_disorder': (_get_map_int, _set_map_int),
                     'builds': (_get_map_proto, _set_map_builds_proto),
                     'note': (_get_string, _set_string),
                     'board_state': (_get_repeated_int, _set_repeated_int)},
    game_pb2.State.Builds: {'count': (_get_int, _set_int),
                            'homes': (_get_repeated_string, _set_repeated_string)},
    game_pb2.PhaseHistory: {'name': (_get_string, _set_string),
                            'state': (_get_proto, _set_state_proto),
                            'orders': (_get_map_string_list, _set_map_string_list),
                            'results': (_get_map_string_list, _set_map_string_list),
                            'policy': (_get_map_proto_no_default, _set_map_policy_details_proto),
                            'prev_orders_state': (_get_repeated_int, _set_repeated_int),
                            'state_value': (_get_map_float, _set_map_float),
                            'possible_orders': (_get_map_string_list, _set_map_string_list)},
    game_pb2.PhaseHistory.PolicyDetails: {'locs': (_get_repeated_string, _set_repeated_string),
                                          'tokens': (_get_repeated_int, _set_repeated_int),
                                          'log_probs': (_get_repeated_float, _set_repeated_float),
                                          'draw_action': (_get_bool, _set_bool),
                                          'draw_prob': (_get_float, _set_float)},
    game_pb2.SavedGame: {'id': (_get_string, _set_string),
                         'map': (_get_string, _set_string),
                         'rules': (_get_repeated_string, _set_repeated_string),
                         'phases': (_get_repeated_proto, _set_repeated_phase_history_proto),
                         'done_reason': (_get_string, _set_string),
                         'assigned_powers': (_get_repeated_string, _set_repeated_string),
                         'players': (_get_repeated_string, _set_repeated_string),
                         'kwargs': (_get_map_proto, _set_map_keyword_args_proto),
                         'is_partial_game': (_get_bool, _set_bool),
                         'start_phase_ix': (_get_int, _set_int),
                         'reward_fn': (_get_string, _set_string),
                         'rewards': (_get_map_float_list, _set_map_float_list),
                         'returns': (_get_map_float_list, _set_map_float_list)},
    game_pb2.SavedGame.KeywordArgs: {'player_seed': (_get_int, _set_int),
                                     'noise': (_get_float, _set_float),
                                     'temperature': (_get_float, _set_float),
                                     'dropout_rate': (_get_float, _set_float)}
}
