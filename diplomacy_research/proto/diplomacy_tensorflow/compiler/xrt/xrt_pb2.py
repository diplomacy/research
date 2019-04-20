# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/compiler/xrt/xrt.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from diplomacy_tensorflow.compiler.tf2xla import host_compute_metadata_pb2 as diplomacy__tensorflow_dot_compiler_dot_tf2xla_dot_host__compute__metadata__pb2
from diplomacy_tensorflow.compiler.xla import xla_pb2 as diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__pb2
from diplomacy_tensorflow.compiler.xla import xla_data_pb2 as diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2
from diplomacy_tensorflow.compiler.xla.service import hlo_pb2 as diplomacy__tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/compiler/xrt/xrt.proto',
  package='xrt',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n+diplomacy_tensorflow/compiler/xrt/xrt.proto\x12\x03xrt\x1a@diplomacy_tensorflow/compiler/tf2xla/host_compute_metadata.proto\x1a+diplomacy_tensorflow/compiler/xla/xla.proto\x1a\x30\x64iplomacy_tensorflow/compiler/xla/xla_data.proto\x1a\x33\x64iplomacy_tensorflow/compiler/xla/service/hlo.proto\"\xee\x01\n\x10\x44\x65viceAssignment\x12\x44\n\x13\x63omputation_devices\x18\x01 \x03(\x0b\x32\'.xrt.DeviceAssignment.ComputationDevice\x1a\x93\x01\n\x11\x43omputationDevice\x12V\n\x0freplica_devices\x18\x01 \x03(\x0b\x32=.xrt.DeviceAssignment.ComputationDevice.DeviceMeshCoordinates\x1a&\n\x15\x44\x65viceMeshCoordinates\x12\r\n\x05value\x18\x01 \x03(\x05\"\xdf\x02\n\x14XLAComputationConfig\x12\x14\n\x0cnum_replicas\x18\x01 \x01(\x05\x12\x1d\n\x15num_cores_per_replica\x18\x02 \x01(\x05\x12O\n\x15host_compute_metadata\x18\x03 \x01(\x0b\x32\x30.diplomacy.tensorflow.tf2xla.HostComputeMetadata\x12-\n\rprogram_shape\x18\x04 \x01(\x0b\x32\x16.xla.ProgramShapeProto\x12\x36\n\x16per_core_program_shape\x18\x05 \x03(\x0b\x32\x16.xla.ProgramShapeProto\x12\x30\n\x11\x64\x65vice_assignment\x18\x06 \x01(\x0b\x32\x15.xrt.DeviceAssignment\x12(\n\rdebug_options\x18\x07 \x01(\x0b\x32\x11.xla.DebugOptions\"c\n\x0eXLAComputation\x12)\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\x19.xrt.XLAComputationConfig\x12&\n\x0chlo_snapshot\x18\x02 \x01(\x0b\x32\x10.xla.HloSnapshot\"I\n\rXLAAllocation\x12\x16\n\x0e\x64\x65vice_ordinal\x18\x01 \x01(\x05\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.xla.LiteralProto\"d\n\x0cXLATupleNode\x12\x13\n\x0binput_index\x18\x01 \x01(\x05\x12\x1c\n\x14release_input_handle\x18\x02 \x01(\x08\x12!\n\x06tuples\x18\x03 \x03(\x0b\x32\x11.xrt.XLATupleNode\"\xdf\x01\n\x12XRTExecutionConfig\x12\x16\n\x0e\x64\x65vice_ordinal\x18\x01 \x01(\x05\x12\x1d\n\x15\x63ore_index_in_replica\x18\x02 \x01(\x05\x12\x1e\n\x16\x65xecution_instance_key\x18\x03 \x01(\t\x12\x10\n\x08rng_seed\x18\x04 \x01(\r\x12\x1d\n\x15release_input_handles\x18\x05 \x01(\x08\x12\"\n\x1arelease_compilation_handle\x18\x06 \x01(\x08\x12\x1d\n\x15return_exploded_tuple\x18\x07 \x01(\x08\x62\x06proto3')
  ,
  dependencies=[diplomacy__tensorflow_dot_compiler_dot_tf2xla_dot_host__compute__metadata__pb2.DESCRIPTOR,diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__pb2.DESCRIPTOR,diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2.DESCRIPTOR,diplomacy__tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2.DESCRIPTOR,])




_DEVICEASSIGNMENT_COMPUTATIONDEVICE_DEVICEMESHCOORDINATES = _descriptor.Descriptor(
  name='DeviceMeshCoordinates',
  full_name='xrt.DeviceAssignment.ComputationDevice.DeviceMeshCoordinates',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='xrt.DeviceAssignment.ComputationDevice.DeviceMeshCoordinates.value', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=467,
  serialized_end=505,
)

_DEVICEASSIGNMENT_COMPUTATIONDEVICE = _descriptor.Descriptor(
  name='ComputationDevice',
  full_name='xrt.DeviceAssignment.ComputationDevice',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='replica_devices', full_name='xrt.DeviceAssignment.ComputationDevice.replica_devices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DEVICEASSIGNMENT_COMPUTATIONDEVICE_DEVICEMESHCOORDINATES, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=358,
  serialized_end=505,
)

_DEVICEASSIGNMENT = _descriptor.Descriptor(
  name='DeviceAssignment',
  full_name='xrt.DeviceAssignment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='computation_devices', full_name='xrt.DeviceAssignment.computation_devices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DEVICEASSIGNMENT_COMPUTATIONDEVICE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=267,
  serialized_end=505,
)


_XLACOMPUTATIONCONFIG = _descriptor.Descriptor(
  name='XLAComputationConfig',
  full_name='xrt.XLAComputationConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_replicas', full_name='xrt.XLAComputationConfig.num_replicas', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_cores_per_replica', full_name='xrt.XLAComputationConfig.num_cores_per_replica', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='host_compute_metadata', full_name='xrt.XLAComputationConfig.host_compute_metadata', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='program_shape', full_name='xrt.XLAComputationConfig.program_shape', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='per_core_program_shape', full_name='xrt.XLAComputationConfig.per_core_program_shape', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_assignment', full_name='xrt.XLAComputationConfig.device_assignment', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='debug_options', full_name='xrt.XLAComputationConfig.debug_options', index=6,
      number=7, type=11, cpp_type=10, label=1,
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
  serialized_start=508,
  serialized_end=859,
)


_XLACOMPUTATION = _descriptor.Descriptor(
  name='XLAComputation',
  full_name='xrt.XLAComputation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='xrt.XLAComputation.config', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hlo_snapshot', full_name='xrt.XLAComputation.hlo_snapshot', index=1,
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
  serialized_start=861,
  serialized_end=960,
)


_XLAALLOCATION = _descriptor.Descriptor(
  name='XLAAllocation',
  full_name='xrt.XLAAllocation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device_ordinal', full_name='xrt.XLAAllocation.device_ordinal', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='xrt.XLAAllocation.value', index=1,
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
  serialized_start=962,
  serialized_end=1035,
)


_XLATUPLENODE = _descriptor.Descriptor(
  name='XLATupleNode',
  full_name='xrt.XLATupleNode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_index', full_name='xrt.XLATupleNode.input_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='release_input_handle', full_name='xrt.XLATupleNode.release_input_handle', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tuples', full_name='xrt.XLATupleNode.tuples', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1037,
  serialized_end=1137,
)


_XRTEXECUTIONCONFIG = _descriptor.Descriptor(
  name='XRTExecutionConfig',
  full_name='xrt.XRTExecutionConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device_ordinal', full_name='xrt.XRTExecutionConfig.device_ordinal', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='core_index_in_replica', full_name='xrt.XRTExecutionConfig.core_index_in_replica', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='execution_instance_key', full_name='xrt.XRTExecutionConfig.execution_instance_key', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rng_seed', full_name='xrt.XRTExecutionConfig.rng_seed', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='release_input_handles', full_name='xrt.XRTExecutionConfig.release_input_handles', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='release_compilation_handle', full_name='xrt.XRTExecutionConfig.release_compilation_handle', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='return_exploded_tuple', full_name='xrt.XRTExecutionConfig.return_exploded_tuple', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=1140,
  serialized_end=1363,
)

_DEVICEASSIGNMENT_COMPUTATIONDEVICE_DEVICEMESHCOORDINATES.containing_type = _DEVICEASSIGNMENT_COMPUTATIONDEVICE
_DEVICEASSIGNMENT_COMPUTATIONDEVICE.fields_by_name['replica_devices'].message_type = _DEVICEASSIGNMENT_COMPUTATIONDEVICE_DEVICEMESHCOORDINATES
_DEVICEASSIGNMENT_COMPUTATIONDEVICE.containing_type = _DEVICEASSIGNMENT
_DEVICEASSIGNMENT.fields_by_name['computation_devices'].message_type = _DEVICEASSIGNMENT_COMPUTATIONDEVICE
_XLACOMPUTATIONCONFIG.fields_by_name['host_compute_metadata'].message_type = diplomacy__tensorflow_dot_compiler_dot_tf2xla_dot_host__compute__metadata__pb2._HOSTCOMPUTEMETADATA
_XLACOMPUTATIONCONFIG.fields_by_name['program_shape'].message_type = diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2._PROGRAMSHAPEPROTO
_XLACOMPUTATIONCONFIG.fields_by_name['per_core_program_shape'].message_type = diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2._PROGRAMSHAPEPROTO
_XLACOMPUTATIONCONFIG.fields_by_name['device_assignment'].message_type = _DEVICEASSIGNMENT
_XLACOMPUTATIONCONFIG.fields_by_name['debug_options'].message_type = diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__pb2._DEBUGOPTIONS
_XLACOMPUTATION.fields_by_name['config'].message_type = _XLACOMPUTATIONCONFIG
_XLACOMPUTATION.fields_by_name['hlo_snapshot'].message_type = diplomacy__tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2._HLOSNAPSHOT
_XLAALLOCATION.fields_by_name['value'].message_type = diplomacy__tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2._LITERALPROTO
_XLATUPLENODE.fields_by_name['tuples'].message_type = _XLATUPLENODE
DESCRIPTOR.message_types_by_name['DeviceAssignment'] = _DEVICEASSIGNMENT
DESCRIPTOR.message_types_by_name['XLAComputationConfig'] = _XLACOMPUTATIONCONFIG
DESCRIPTOR.message_types_by_name['XLAComputation'] = _XLACOMPUTATION
DESCRIPTOR.message_types_by_name['XLAAllocation'] = _XLAALLOCATION
DESCRIPTOR.message_types_by_name['XLATupleNode'] = _XLATUPLENODE
DESCRIPTOR.message_types_by_name['XRTExecutionConfig'] = _XRTEXECUTIONCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DeviceAssignment = _reflection.GeneratedProtocolMessageType('DeviceAssignment', (_message.Message,), dict(

  ComputationDevice = _reflection.GeneratedProtocolMessageType('ComputationDevice', (_message.Message,), dict(

    DeviceMeshCoordinates = _reflection.GeneratedProtocolMessageType('DeviceMeshCoordinates', (_message.Message,), dict(
      DESCRIPTOR = _DEVICEASSIGNMENT_COMPUTATIONDEVICE_DEVICEMESHCOORDINATES,
      __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
      # @@protoc_insertion_point(class_scope:xrt.DeviceAssignment.ComputationDevice.DeviceMeshCoordinates)
      ))
    ,
    DESCRIPTOR = _DEVICEASSIGNMENT_COMPUTATIONDEVICE,
    __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
    # @@protoc_insertion_point(class_scope:xrt.DeviceAssignment.ComputationDevice)
    ))
  ,
  DESCRIPTOR = _DEVICEASSIGNMENT,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.DeviceAssignment)
  ))
_sym_db.RegisterMessage(DeviceAssignment)
_sym_db.RegisterMessage(DeviceAssignment.ComputationDevice)
_sym_db.RegisterMessage(DeviceAssignment.ComputationDevice.DeviceMeshCoordinates)

XLAComputationConfig = _reflection.GeneratedProtocolMessageType('XLAComputationConfig', (_message.Message,), dict(
  DESCRIPTOR = _XLACOMPUTATIONCONFIG,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.XLAComputationConfig)
  ))
_sym_db.RegisterMessage(XLAComputationConfig)

XLAComputation = _reflection.GeneratedProtocolMessageType('XLAComputation', (_message.Message,), dict(
  DESCRIPTOR = _XLACOMPUTATION,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.XLAComputation)
  ))
_sym_db.RegisterMessage(XLAComputation)

XLAAllocation = _reflection.GeneratedProtocolMessageType('XLAAllocation', (_message.Message,), dict(
  DESCRIPTOR = _XLAALLOCATION,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.XLAAllocation)
  ))
_sym_db.RegisterMessage(XLAAllocation)

XLATupleNode = _reflection.GeneratedProtocolMessageType('XLATupleNode', (_message.Message,), dict(
  DESCRIPTOR = _XLATUPLENODE,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.XLATupleNode)
  ))
_sym_db.RegisterMessage(XLATupleNode)

XRTExecutionConfig = _reflection.GeneratedProtocolMessageType('XRTExecutionConfig', (_message.Message,), dict(
  DESCRIPTOR = _XRTEXECUTIONCONFIG,
  __module__ = 'diplomacy_tensorflow.compiler.xrt.xrt_pb2'
  # @@protoc_insertion_point(class_scope:xrt.XRTExecutionConfig)
  ))
_sym_db.RegisterMessage(XRTExecutionConfig)


# @@protoc_insertion_point(module_scope)
