# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/python/training/checkpoint_state.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/python/training/checkpoint_state.proto',
  package='diplomacy.tensorflow',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n;diplomacy_tensorflow/python/training/checkpoint_state.proto\x12\x14\x64iplomacy.tensorflow\"\x9f\x01\n\x0f\x43heckpointState\x12\x1d\n\x15model_checkpoint_path\x18\x01 \x01(\t\x12\"\n\x1a\x61ll_model_checkpoint_paths\x18\x02 \x03(\t\x12\'\n\x1f\x61ll_model_checkpoint_timestamps\x18\x03 \x03(\x01\x12 \n\x18last_preserved_timestamp\x18\x04 \x01(\x01\x42\x03\xf8\x01\x01\x62\x06proto3')
)




_CHECKPOINTSTATE = _descriptor.Descriptor(
  name='CheckpointState',
  full_name='diplomacy.tensorflow.CheckpointState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_checkpoint_path', full_name='diplomacy.tensorflow.CheckpointState.model_checkpoint_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='all_model_checkpoint_paths', full_name='diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='all_model_checkpoint_timestamps', full_name='diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_preserved_timestamp', full_name='diplomacy.tensorflow.CheckpointState.last_preserved_timestamp', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=86,
  serialized_end=245,
)

DESCRIPTOR.message_types_by_name['CheckpointState'] = _CHECKPOINTSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CheckpointState = _reflection.GeneratedProtocolMessageType('CheckpointState', (_message.Message,), dict(
  DESCRIPTOR = _CHECKPOINTSTATE,
  __module__ = 'diplomacy_tensorflow.python.training.checkpoint_state_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.CheckpointState)
  ))
_sym_db.RegisterMessage(CheckpointState)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
