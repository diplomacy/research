# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/core/protobuf/master_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from diplomacy_tensorflow.core.protobuf import master_pb2 as diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/core/protobuf/master_service.proto',
  package='diplomacy.tensorflow.grpc',
  syntax='proto3',
  serialized_options=_b('\n\032org.tensorflow.distruntimeB\023MasterServiceProtosP\001Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf'),
  serialized_pb=_b('\n7diplomacy_tensorflow/core/protobuf/master_service.proto\x12\x19\x64iplomacy.tensorflow.grpc\x1a/diplomacy_tensorflow/core/protobuf/master.proto2\x83\x08\n\rMasterService\x12h\n\rCreateSession\x12*.diplomacy.tensorflow.CreateSessionRequest\x1a+.diplomacy.tensorflow.CreateSessionResponse\x12h\n\rExtendSession\x12*.diplomacy.tensorflow.ExtendSessionRequest\x1a+.diplomacy.tensorflow.ExtendSessionResponse\x12n\n\x0fPartialRunSetup\x12,.diplomacy.tensorflow.PartialRunSetupRequest\x1a-.diplomacy.tensorflow.PartialRunSetupResponse\x12V\n\x07RunStep\x12$.diplomacy.tensorflow.RunStepRequest\x1a%.diplomacy.tensorflow.RunStepResponse\x12\x65\n\x0c\x43loseSession\x12).diplomacy.tensorflow.CloseSessionRequest\x1a*.diplomacy.tensorflow.CloseSessionResponse\x12\x62\n\x0bListDevices\x12(.diplomacy.tensorflow.ListDevicesRequest\x1a).diplomacy.tensorflow.ListDevicesResponse\x12P\n\x05Reset\x12\".diplomacy.tensorflow.ResetRequest\x1a#.diplomacy.tensorflow.ResetResponse\x12\x65\n\x0cMakeCallable\x12).diplomacy.tensorflow.MakeCallableRequest\x1a*.diplomacy.tensorflow.MakeCallableResponse\x12\x62\n\x0bRunCallable\x12(.diplomacy.tensorflow.RunCallableRequest\x1a).diplomacy.tensorflow.RunCallableResponse\x12n\n\x0fReleaseCallable\x12,.diplomacy.tensorflow.ReleaseCallableRequest\x1a-.diplomacy.tensorflow.ReleaseCallableResponseBq\n\x1aorg.tensorflow.distruntimeB\x13MasterServiceProtosP\x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobufb\x06proto3')
  ,
  dependencies=[diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_MASTERSERVICE = _descriptor.ServiceDescriptor(
  name='MasterService',
  full_name='diplomacy.tensorflow.grpc.MasterService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=136,
  serialized_end=1163,
  methods=[
  _descriptor.MethodDescriptor(
    name='CreateSession',
    full_name='diplomacy.tensorflow.grpc.MasterService.CreateSession',
    index=0,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CREATESESSIONREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CREATESESSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ExtendSession',
    full_name='diplomacy.tensorflow.grpc.MasterService.ExtendSession',
    index=1,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._EXTENDSESSIONREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._EXTENDSESSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='PartialRunSetup',
    full_name='diplomacy.tensorflow.grpc.MasterService.PartialRunSetup',
    index=2,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._PARTIALRUNSETUPREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._PARTIALRUNSETUPRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='RunStep',
    full_name='diplomacy.tensorflow.grpc.MasterService.RunStep',
    index=3,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNSTEPREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNSTEPRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='CloseSession',
    full_name='diplomacy.tensorflow.grpc.MasterService.CloseSession',
    index=4,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CLOSESESSIONREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._CLOSESESSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListDevices',
    full_name='diplomacy.tensorflow.grpc.MasterService.ListDevices',
    index=5,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._LISTDEVICESREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._LISTDEVICESRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Reset',
    full_name='diplomacy.tensorflow.grpc.MasterService.Reset',
    index=6,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RESETREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RESETRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='MakeCallable',
    full_name='diplomacy.tensorflow.grpc.MasterService.MakeCallable',
    index=7,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._MAKECALLABLEREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._MAKECALLABLERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='RunCallable',
    full_name='diplomacy.tensorflow.grpc.MasterService.RunCallable',
    index=8,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNCALLABLEREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RUNCALLABLERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReleaseCallable',
    full_name='diplomacy.tensorflow.grpc.MasterService.ReleaseCallable',
    index=9,
    containing_service=None,
    input_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RELEASECALLABLEREQUEST,
    output_type=diplomacy__tensorflow_dot_core_dot_protobuf_dot_master__pb2._RELEASECALLABLERESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MASTERSERVICE)

DESCRIPTOR.services_by_name['MasterService'] = _MASTERSERVICE

# @@protoc_insertion_point(module_scope)
