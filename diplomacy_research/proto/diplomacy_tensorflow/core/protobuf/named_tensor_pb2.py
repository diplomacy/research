# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/core/protobuf/named_tensor.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from diplomacy_tensorflow.core.framework import tensor_pb2 as diplomacy__tensorflow_dot_core_dot_framework_dot_tensor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/core/protobuf/named_tensor.proto',
  package='diplomacy.tensorflow',
  syntax='proto3',
  serialized_options=_b('\n\030org.tensorflow.frameworkB\021NamedTensorProtosP\001Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\370\001\001'),
  serialized_pb=_b('\n5diplomacy_tensorflow/core/protobuf/named_tensor.proto\x12\x14\x64iplomacy.tensorflow\x1a\x30\x64iplomacy_tensorflow/core/framework/tensor.proto\"S\n\x10NamedTensorProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x31\n\x06tensor\x18\x02 \x01(\x0b\x32!.diplomacy.tensorflow.TensorProtoBp\n\x18org.tensorflow.frameworkB\x11NamedTensorProtosP\x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[diplomacy__tensorflow_dot_core_dot_framework_dot_tensor__pb2.DESCRIPTOR,])




_NAMEDTENSORPROTO = _descriptor.Descriptor(
  name='NamedTensorProto',
  full_name='diplomacy.tensorflow.NamedTensorProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='diplomacy.tensorflow.NamedTensorProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='diplomacy.tensorflow.NamedTensorProto.tensor', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=129,
  serialized_end=212,
)

_NAMEDTENSORPROTO.fields_by_name['tensor'].message_type = diplomacy__tensorflow_dot_core_dot_framework_dot_tensor__pb2._TENSORPROTO
DESCRIPTOR.message_types_by_name['NamedTensorProto'] = _NAMEDTENSORPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NamedTensorProto = _reflection.GeneratedProtocolMessageType('NamedTensorProto', (_message.Message,), dict(
  DESCRIPTOR = _NAMEDTENSORPROTO,
  __module__ = 'diplomacy_tensorflow.core.protobuf.named_tensor_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.NamedTensorProto)
  ))
_sym_db.RegisterMessage(NamedTensorProto)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)