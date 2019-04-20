# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: diplomacy_tensorflow/core/profiler/profile.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='diplomacy_tensorflow/core/profiler/profile.proto',
  package='diplomacy.tensorflow.tfprof.pprof',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n0diplomacy_tensorflow/core/profiler/profile.proto\x12!diplomacy.tensorflow.tfprof.pprof\"\xaf\x04\n\x07Profile\x12\x41\n\x0bsample_type\x18\x01 \x03(\x0b\x32,.diplomacy.tensorflow.tfprof.pprof.ValueType\x12\x39\n\x06sample\x18\x02 \x03(\x0b\x32).diplomacy.tensorflow.tfprof.pprof.Sample\x12;\n\x07mapping\x18\x03 \x03(\x0b\x32*.diplomacy.tensorflow.tfprof.pprof.Mapping\x12=\n\x08location\x18\x04 \x03(\x0b\x32+.diplomacy.tensorflow.tfprof.pprof.Location\x12=\n\x08\x66unction\x18\x05 \x03(\x0b\x32+.diplomacy.tensorflow.tfprof.pprof.Function\x12\x14\n\x0cstring_table\x18\x06 \x03(\t\x12\x13\n\x0b\x64rop_frames\x18\x07 \x01(\x03\x12\x13\n\x0bkeep_frames\x18\x08 \x01(\x03\x12\x12\n\ntime_nanos\x18\t \x01(\x03\x12\x16\n\x0e\x64uration_nanos\x18\n \x01(\x03\x12\x41\n\x0bperiod_type\x18\x0b \x01(\x0b\x32,.diplomacy.tensorflow.tfprof.pprof.ValueType\x12\x0e\n\x06period\x18\x0c \x01(\x03\x12\x0f\n\x07\x63omment\x18\r \x03(\x03\x12\x1b\n\x13\x64\x65\x66\x61ult_sample_type\x18\x0e \x01(\x03\"\'\n\tValueType\x12\x0c\n\x04type\x18\x01 \x01(\x03\x12\x0c\n\x04unit\x18\x02 \x01(\x03\"e\n\x06Sample\x12\x13\n\x0blocation_id\x18\x01 \x03(\x04\x12\r\n\x05value\x18\x02 \x03(\x03\x12\x37\n\x05label\x18\x03 \x03(\x0b\x32(.diplomacy.tensorflow.tfprof.pprof.Label\".\n\x05Label\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\x0b\n\x03str\x18\x02 \x01(\x03\x12\x0b\n\x03num\x18\x03 \x01(\x03\"\xdd\x01\n\x07Mapping\x12\n\n\x02id\x18\x01 \x01(\x04\x12\x14\n\x0cmemory_start\x18\x02 \x01(\x04\x12\x14\n\x0cmemory_limit\x18\x03 \x01(\x04\x12\x13\n\x0b\x66ile_offset\x18\x04 \x01(\x04\x12\x10\n\x08\x66ilename\x18\x05 \x01(\x03\x12\x10\n\x08\x62uild_id\x18\x06 \x01(\x03\x12\x15\n\rhas_functions\x18\x07 \x01(\x08\x12\x15\n\rhas_filenames\x18\x08 \x01(\x08\x12\x18\n\x10has_line_numbers\x18\t \x01(\x08\x12\x19\n\x11has_inline_frames\x18\n \x01(\x08\"r\n\x08Location\x12\n\n\x02id\x18\x01 \x01(\x04\x12\x12\n\nmapping_id\x18\x02 \x01(\x04\x12\x0f\n\x07\x61\x64\x64ress\x18\x03 \x01(\x04\x12\x35\n\x04line\x18\x04 \x03(\x0b\x32\'.diplomacy.tensorflow.tfprof.pprof.Line\")\n\x04Line\x12\x13\n\x0b\x66unction_id\x18\x01 \x01(\x04\x12\x0c\n\x04line\x18\x02 \x01(\x03\"_\n\x08\x46unction\x12\n\n\x02id\x18\x01 \x01(\x04\x12\x0c\n\x04name\x18\x02 \x01(\x03\x12\x13\n\x0bsystem_name\x18\x03 \x01(\x03\x12\x10\n\x08\x66ilename\x18\x04 \x01(\x03\x12\x12\n\nstart_line\x18\x05 \x01(\x03\x62\x06proto3')
)




_PROFILE = _descriptor.Descriptor(
  name='Profile',
  full_name='diplomacy.tensorflow.tfprof.pprof.Profile',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sample_type', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.sample_type', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.sample', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mapping', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.mapping', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.location', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='function', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.function', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='string_table', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.string_table', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='drop_frames', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.drop_frames', index=6,
      number=7, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keep_frames', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.keep_frames', index=7,
      number=8, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_nanos', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.time_nanos', index=8,
      number=9, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='duration_nanos', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.duration_nanos', index=9,
      number=10, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='period_type', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.period_type', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='period', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.period', index=11,
      number=12, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='comment', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.comment', index=12,
      number=13, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='default_sample_type', full_name='diplomacy.tensorflow.tfprof.pprof.Profile.default_sample_type', index=13,
      number=14, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=88,
  serialized_end=647,
)


_VALUETYPE = _descriptor.Descriptor(
  name='ValueType',
  full_name='diplomacy.tensorflow.tfprof.pprof.ValueType',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='diplomacy.tensorflow.tfprof.pprof.ValueType.type', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit', full_name='diplomacy.tensorflow.tfprof.pprof.ValueType.unit', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=649,
  serialized_end=688,
)


_SAMPLE = _descriptor.Descriptor(
  name='Sample',
  full_name='diplomacy.tensorflow.tfprof.pprof.Sample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='location_id', full_name='diplomacy.tensorflow.tfprof.pprof.Sample.location_id', index=0,
      number=1, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='diplomacy.tensorflow.tfprof.pprof.Sample.value', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='diplomacy.tensorflow.tfprof.pprof.Sample.label', index=2,
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
  serialized_start=690,
  serialized_end=791,
)


_LABEL = _descriptor.Descriptor(
  name='Label',
  full_name='diplomacy.tensorflow.tfprof.pprof.Label',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='diplomacy.tensorflow.tfprof.pprof.Label.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='str', full_name='diplomacy.tensorflow.tfprof.pprof.Label.str', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num', full_name='diplomacy.tensorflow.tfprof.pprof.Label.num', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=793,
  serialized_end=839,
)


_MAPPING = _descriptor.Descriptor(
  name='Mapping',
  full_name='diplomacy.tensorflow.tfprof.pprof.Mapping',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='memory_start', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.memory_start', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='memory_limit', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.memory_limit', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='file_offset', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.file_offset', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filename', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.filename', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='build_id', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.build_id', index=5,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='has_functions', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.has_functions', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='has_filenames', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.has_filenames', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='has_line_numbers', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.has_line_numbers', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='has_inline_frames', full_name='diplomacy.tensorflow.tfprof.pprof.Mapping.has_inline_frames', index=9,
      number=10, type=8, cpp_type=7, label=1,
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
  serialized_start=842,
  serialized_end=1063,
)


_LOCATION = _descriptor.Descriptor(
  name='Location',
  full_name='diplomacy.tensorflow.tfprof.pprof.Location',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='diplomacy.tensorflow.tfprof.pprof.Location.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mapping_id', full_name='diplomacy.tensorflow.tfprof.pprof.Location.mapping_id', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='diplomacy.tensorflow.tfprof.pprof.Location.address', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='line', full_name='diplomacy.tensorflow.tfprof.pprof.Location.line', index=3,
      number=4, type=11, cpp_type=10, label=3,
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
  serialized_start=1065,
  serialized_end=1179,
)


_LINE = _descriptor.Descriptor(
  name='Line',
  full_name='diplomacy.tensorflow.tfprof.pprof.Line',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_id', full_name='diplomacy.tensorflow.tfprof.pprof.Line.function_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='line', full_name='diplomacy.tensorflow.tfprof.pprof.Line.line', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=1181,
  serialized_end=1222,
)


_FUNCTION = _descriptor.Descriptor(
  name='Function',
  full_name='diplomacy.tensorflow.tfprof.pprof.Function',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='diplomacy.tensorflow.tfprof.pprof.Function.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='diplomacy.tensorflow.tfprof.pprof.Function.name', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='system_name', full_name='diplomacy.tensorflow.tfprof.pprof.Function.system_name', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filename', full_name='diplomacy.tensorflow.tfprof.pprof.Function.filename', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_line', full_name='diplomacy.tensorflow.tfprof.pprof.Function.start_line', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=1224,
  serialized_end=1319,
)

_PROFILE.fields_by_name['sample_type'].message_type = _VALUETYPE
_PROFILE.fields_by_name['sample'].message_type = _SAMPLE
_PROFILE.fields_by_name['mapping'].message_type = _MAPPING
_PROFILE.fields_by_name['location'].message_type = _LOCATION
_PROFILE.fields_by_name['function'].message_type = _FUNCTION
_PROFILE.fields_by_name['period_type'].message_type = _VALUETYPE
_SAMPLE.fields_by_name['label'].message_type = _LABEL
_LOCATION.fields_by_name['line'].message_type = _LINE
DESCRIPTOR.message_types_by_name['Profile'] = _PROFILE
DESCRIPTOR.message_types_by_name['ValueType'] = _VALUETYPE
DESCRIPTOR.message_types_by_name['Sample'] = _SAMPLE
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['Mapping'] = _MAPPING
DESCRIPTOR.message_types_by_name['Location'] = _LOCATION
DESCRIPTOR.message_types_by_name['Line'] = _LINE
DESCRIPTOR.message_types_by_name['Function'] = _FUNCTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Profile = _reflection.GeneratedProtocolMessageType('Profile', (_message.Message,), dict(
  DESCRIPTOR = _PROFILE,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Profile)
  ))
_sym_db.RegisterMessage(Profile)

ValueType = _reflection.GeneratedProtocolMessageType('ValueType', (_message.Message,), dict(
  DESCRIPTOR = _VALUETYPE,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.ValueType)
  ))
_sym_db.RegisterMessage(ValueType)

Sample = _reflection.GeneratedProtocolMessageType('Sample', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLE,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Sample)
  ))
_sym_db.RegisterMessage(Sample)

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), dict(
  DESCRIPTOR = _LABEL,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Label)
  ))
_sym_db.RegisterMessage(Label)

Mapping = _reflection.GeneratedProtocolMessageType('Mapping', (_message.Message,), dict(
  DESCRIPTOR = _MAPPING,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Mapping)
  ))
_sym_db.RegisterMessage(Mapping)

Location = _reflection.GeneratedProtocolMessageType('Location', (_message.Message,), dict(
  DESCRIPTOR = _LOCATION,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Location)
  ))
_sym_db.RegisterMessage(Location)

Line = _reflection.GeneratedProtocolMessageType('Line', (_message.Message,), dict(
  DESCRIPTOR = _LINE,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Line)
  ))
_sym_db.RegisterMessage(Line)

Function = _reflection.GeneratedProtocolMessageType('Function', (_message.Message,), dict(
  DESCRIPTOR = _FUNCTION,
  __module__ = 'diplomacy_tensorflow.core.profiler.profile_pb2'
  # @@protoc_insertion_point(class_scope:diplomacy.tensorflow.tfprof.pprof.Function)
  ))
_sym_db.RegisterMessage(Function)


# @@protoc_insertion_point(module_scope)
