# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/core/protobuf/replay_log.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from diplomacy_tensorflow.core.framework import graph_pb2 as diplomacy__tensorflow_dot_core_dot_framework_dot_graph__pb2
from diplomacy_tensorflow.core.protobuf import cluster_pb2 as diplomacy__tensorflow_dot_core_dot_protobuf_dot_cluster__pb2
from diplomacy_tensorflow.core.protobuf import master_pb2 as diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/core/protobuf/replay_log.proto',
  package='diplomacy.tensorflow',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n3diplomacy_tensorflow/core/protobuf/replay_log.proto\x12\x14\x64iplomacy.tensorflow\x1a/diplomacy_tensorflow/core/framework/graph.proto\x1a\x30\x64iplomacy_tensorflow/core/protobuf/cluster.proto\x1a/diplomacy_tensorflow/core/protobuf/master.proto\"f\n\x10NewReplaySession\x12:\n\x07\x64\x65vices\x18\x01 \x01(\x0b\x32).diplomacy.tensorflow.ListDevicesResponse\x12\x16\n\x0esession_handle\x18\x02 \x01(\t\"\xba\x0c\n\x08ReplayOp\x12\x15\n\rstart_time_us\x18\x1f \x01(\x01\x12\x13\n\x0b\x65nd_time_us\x18  \x01(\x01\x12\x44\n\x0e\x63reate_session\x18\x01 \x01(\x0b\x32*.diplomacy.tensorflow.CreateSessionRequestH\x00\x12\x44\n\x0e\x65xtend_session\x18\x02 \x01(\x0b\x32*.diplomacy.tensorflow.ExtendSessionRequestH\x00\x12I\n\x11partial_run_setup\x18\x03 \x01(\x0b\x32,.diplomacy.tensorflow.PartialRunSetupRequestH\x00\x12\x38\n\x08run_step\x18\x04 \x01(\x0b\x32$.diplomacy.tensorflow.RunStepRequestH\x00\x12\x42\n\rclose_session\x18\x05 \x01(\x0b\x32).diplomacy.tensorflow.CloseSessionRequestH\x00\x12@\n\x0clist_devices\x18\x06 \x01(\x0b\x32(.diplomacy.tensorflow.ListDevicesRequestH\x00\x12;\n\rreset_request\x18\x07 \x01(\x0b\x32\".diplomacy.tensorflow.ResetRequestH\x00\x12\x42\n\rmake_callable\x18\x08 \x01(\x0b\x32).diplomacy.tensorflow.MakeCallableRequestH\x00\x12@\n\x0crun_callable\x18\t \x01(\x0b\x32(.diplomacy.tensorflow.RunCallableRequestH\x00\x12H\n\x10release_callable\x18\n \x01(\x0b\x32,.diplomacy.tensorflow.ReleaseCallableRequestH\x00\x12\x44\n\x12new_replay_session\x18\x0b \x01(\x0b\x32&.diplomacy.tensorflow.NewReplaySessionH\x00\x12N\n\x17\x63reate_session_response\x18\x15 \x01(\x0b\x32+.diplomacy.tensorflow.CreateSessionResponseH\x01\x12N\n\x17\x65xtend_session_response\x18\x16 \x01(\x0b\x32+.diplomacy.tensorflow.ExtendSessionResponseH\x01\x12S\n\x1apartial_run_setup_response\x18\x17 \x01(\x0b\x32-.diplomacy.tensorflow.PartialRunSetupResponseH\x01\x12\x42\n\x11run_step_response\x18\x18 \x01(\x0b\x32%.diplomacy.tensorflow.RunStepResponseH\x01\x12L\n\x16\x63lose_session_response\x18\x19 \x01(\x0b\x32*.diplomacy.tensorflow.CloseSessionResponseH\x01\x12J\n\x15list_devices_response\x18\x1a \x01(\x0b\x32).diplomacy.tensorflow.ListDevicesResponseH\x01\x12\x45\n\x16reset_request_response\x18\x1b \x01(\x0b\x32#.diplomacy.tensorflow.ResetResponseH\x01\x12L\n\x16make_callable_response\x18\x1c \x01(\x0b\x32*.diplomacy.tensorflow.MakeCallableResponseH\x01\x12J\n\x15run_callable_response\x18\x1d \x01(\x0b\x32).diplomacy.tensorflow.RunCallableResponseH\x01\x12R\n\x19release_callable_response\x18\x1e \x01(\x0b\x32-.diplomacy.tensorflow.ReleaseCallableResponseH\x01\x42\x04\n\x02opB\n\n\x08responseB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[diplomacy__tensorflow_dot_core_dot_framework_dot_graph__pb2.DESCRIPTOR,diplomacy__tensorflow_dot_core_dot_protobuf_dot_cluster__pb2.DESCRIPTOR,diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2.DESCRIPTOR,])




_NEWREPLAYSESSION = _descriptor.Descriptor(
  name='NewReplaySession',
  full_name='diplomacy.tensorflow.NewReplaySession',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='devices', full_name='diplomacy.tensorflow.NewReplaySession.devices', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='session_handle', full_name='diplomacy.tensorflow.NewReplaySession.session_handle', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=225,
  serialized_end=327,
)


_REPLAYOP = _descriptor.Descriptor(
  name='ReplayOp',
  full_name='diplomacy.tensorflow.ReplayOp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_time_us', full_name='diplomacy.tensorflow.ReplayOp.start_time_us', index=0,
      number=31, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time_us', full_name='diplomacy.tensorflow.ReplayOp.end_time_us', index=1,
      number=32, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='create_session', full_name='diplomacy.tensorflow.ReplayOp.create_session', index=2,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='extend_session', full_name='diplomacy.tensorflow.ReplayOp.extend_session', index=3,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='partial_run_setup', full_name='diplomacy.tensorflow.ReplayOp.partial_run_setup', index=4,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_step', full_name='diplomacy.tensorflow.ReplayOp.run_step', index=5,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='close_session', full_name='diplomacy.tensorflow.ReplayOp.close_session', index=6,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='list_devices', full_name='diplomacy.tensorflow.ReplayOp.list_devices', index=7,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reset_request', full_name='diplomacy.tensorflow.ReplayOp.reset_request', index=8,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='make_callable', full_name='diplomacy.tensorflow.ReplayOp.make_callable', index=9,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_callable', full_name='diplomacy.tensorflow.ReplayOp.run_callable', index=10,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='release_callable', full_name='diplomacy.tensorflow.ReplayOp.release_callable', index=11,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_replay_session', full_name='diplomacy.tensorflow.ReplayOp.new_replay_session', index=12,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='create_session_response', full_name='diplomacy.tensorflow.ReplayOp.create_session_response', index=13,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='extend_session_response', full_name='diplomacy.tensorflow.ReplayOp.extend_session_response', index=14,
      number=22, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='partial_run_setup_response', full_name='diplomacy.tensorflow.ReplayOp.partial_run_setup_response', index=15,
      number=23, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_step_response', full_name='diplomacy.tensorflow.ReplayOp.run_step_response', index=16,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='close_session_response', full_name='diplomacy.tensorflow.ReplayOp.close_session_response', index=17,
      number=25, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='list_devices_response', full_name='diplomacy.tensorflow.ReplayOp.list_devices_response', index=18,
      number=26, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reset_request_response', full_name='diplomacy.tensorflow.ReplayOp.reset_request_response', index=19,
      number=27, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='make_callable_response', full_name='diplomacy.tensorflow.ReplayOp.make_callable_response', index=20,
      number=28, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_callable_response', full_name='diplomacy.tensorflow.ReplayOp.run_callable_response', index=21,
      number=29, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='release_callable_response', full_name='diplomacy.tensorflow.ReplayOp.release_callable_response', index=22,
      number=30, type=11, cpp_type=10, label=1,
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
    _descriptor.OneofDescriptor(
      name='op', full_name='diplomacy.tensorflow.ReplayOp.op',
      index=0, containing_type=None, fields=[]),
    _descriptor.OneofDescriptor(
      name='response', full_name='diplomacy.tensorflow.ReplayOp.response',
      index=1, containing_type=None, fields=[]),
  ],
  serialized_start=330,
  serialized_end=1924,
)

_NEWREPLAYSESSION.fields_by_name['devices'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._LISTDEVICESRESPONSE
_REPLAYOP.fields_by_name['create_session'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CREATESESSIONREQUEST
_REPLAYOP.fields_by_name['extend_session'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._EXTENDSESSIONREQUEST
_REPLAYOP.fields_by_name['partial_run_setup'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._PARTIALRUNSETUPREQUEST
_REPLAYOP.fields_by_name['run_step'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNSTEPREQUEST
_REPLAYOP.fields_by_name['close_session'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CLOSESESSIONREQUEST
_REPLAYOP.fields_by_name['list_devices'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._LISTDEVICESREQUEST
_REPLAYOP.fields_by_name['reset_request'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RESETREQUEST
_REPLAYOP.fields_by_name['make_callable'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._MAKECALLABLEREQUEST
_REPLAYOP.fields_by_name['run_callable'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNCALLABLEREQUEST
_REPLAYOP.fields_by_name['release_callable'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RELEASECALLABLEREQUEST
_REPLAYOP.fields_by_name['new_replay_session'].message_type = _NEWREPLAYSESSION
_REPLAYOP.fields_by_name['create_session_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CREATESESSIONRESPONSE
_REPLAYOP.fields_by_name['extend_session_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._EXTENDSESSIONRESPONSE
_REPLAYOP.fields_by_name['partial_run_setup_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._PARTIALRUNSETUPRESPONSE
_REPLAYOP.fields_by_name['run_step_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNSTEPRESPONSE
_REPLAYOP.fields_by_name['close_session_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CLOSESESSIONRESPONSE
_REPLAYOP.fields_by_name['list_devices_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._LISTDEVICESRESPONSE
_REPLAYOP.fields_by_name['reset_request_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RESETRESPONSE
_REPLAYOP.fields_by_name['make_callable_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._MAKECALLABLERESPONSE
_REPLAYOP.fields_by_name['run_callable_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNCALLABLERESPONSE
_REPLAYOP.fields_by_name['release_callable_response'].message_type = diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RELEASECALLABLERESPONSE
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['create_session'])
_REPLAYOP.fields_by_name['create_session'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['extend_session'])
_REPLAYOP.fields_by_name['extend_session'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['partial_run_setup'])
_REPLAYOP.fields_by_name['partial_run_setup'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['run_step'])
_REPLAYOP.fields_by_name['run_step'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['close_session'])
_REPLAYOP.fields_by_name['close_session'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['list_devices'])
_REPLAYOP.fields_by_name['list_devices'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['reset_request'])
_REPLAYOP.fields_by_name['reset_request'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['make_callable'])
_REPLAYOP.fields_by_name['make_callable'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['run_callable'])
_REPLAYOP.fields_by_name['run_callable'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['release_callable'])
_REPLAYOP.fields_by_name['release_callable'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['op'].fields.append(
  _REPLAYOP.fields_by_name['new_replay_session'])
_REPLAYOP.fields_by_name['new_replay_session'].containing_oneof = _REPLAYOP.oneofs_by_name['op']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['create_session_response'])
_REPLAYOP.fields_by_name['create_session_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['extend_session_response'])
_REPLAYOP.fields_by_name['extend_session_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['partial_run_setup_response'])
_REPLAYOP.fields_by_name['partial_run_setup_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['run_step_response'])
_REPLAYOP.fields_by_name['run_step_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['close_session_response'])
_REPLAYOP.fields_by_name['close_session_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['list_devices_response'])
_REPLAYOP.fields_by_name['list_devices_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['reset_request_response'])
_REPLAYOP.fields_by_name['reset_request_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['make_callable_response'])
_REPLAYOP.fields_by_name['make_callable_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['run_callable_response'])
_REPLAYOP.fields_by_name['run_callable_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
_REPLAYOP.oneofs_by_name['response'].fields.append(
  _REPLAYOP.fields_by_name['release_callable_response'])
_REPLAYOP.fields_by_name['release_callable_response'].containing_oneof = _REPLAYOP.oneofs_by_name['response']
DESCRIPTOR.message_types_by_name['NewReplaySession'] = _NEWREPLAYSESSION
DESCRIPTOR.message_types_by_name['ReplayOp'] = _REPLAYOP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NewReplaySession = _reflection.GeneratedProtocolMessageType('NewReplaySession', (_message.Message,), dict(
  DESCRIPTOR = _NEWREPLAYSESSION,
  __module__ = 'diplomacy_tensorflow.core.protobuf.replay_log_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.NewReplaySession)
  ))
_sym_db.RegisterMessage(NewReplaySession)

ReplayOp = _reflection.GeneratedProtocolMessageType('ReplayOp', (_message.Message,), dict(
  DESCRIPTOR = _REPLAYOP,
  __module__ = 'diplomacy_tensorflow.core.protobuf.replay_log_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.ReplayOp)
  ))
_sym_db.RegisterMessage(ReplayOp)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
