// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/lib/core/error_codes.proto

#include "diplomacy_tensorflow/core/lib/core/error_codes.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace diplomacy {
namespace tensorflow {
namespace error {
}  // namespace error
}  // namespace tensorflow
}  // namespace diplomacy
namespace protobuf_diplomacy_5ftensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto {
void InitDefaults() {
}

const ::google::protobuf::EnumDescriptor* file_level_enum_descriptors[1];
const ::google::protobuf::uint32 TableStruct::offsets[1] = {};
static const ::google::protobuf::internal::MigrationSchema* schemas = NULL;
static const ::google::protobuf::Message* const* file_default_instances = NULL;

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/core/lib/core/error_codes.proto", schemas, file_default_instances, TableStruct::offsets,
      NULL, file_level_enum_descriptors, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n4diplomacy_tensorflow/core/lib/core/err"
      "or_codes.proto\022\032diplomacy.tensorflow.err"
      "or*\204\003\n\004Code\022\006\n\002OK\020\000\022\r\n\tCANCELLED\020\001\022\013\n\007UN"
      "KNOWN\020\002\022\024\n\020INVALID_ARGUMENT\020\003\022\025\n\021DEADLIN"
      "E_EXCEEDED\020\004\022\r\n\tNOT_FOUND\020\005\022\022\n\016ALREADY_E"
      "XISTS\020\006\022\025\n\021PERMISSION_DENIED\020\007\022\023\n\017UNAUTH"
      "ENTICATED\020\020\022\026\n\022RESOURCE_EXHAUSTED\020\010\022\027\n\023F"
      "AILED_PRECONDITION\020\t\022\013\n\007ABORTED\020\n\022\020\n\014OUT"
      "_OF_RANGE\020\013\022\021\n\rUNIMPLEMENTED\020\014\022\014\n\010INTERN"
      "AL\020\r\022\017\n\013UNAVAILABLE\020\016\022\r\n\tDATA_LOSS\020\017\022K\nG"
      "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION"
      "_USE_DEFAULT_IN_SWITCH_INSTEAD_\020\024Bo\n\030org"
      ".tensorflow.frameworkB\020ErrorCodesProtosP"
      "\001Z<github.com/tensorflow/tensorflow/tens"
      "orflow/go/core/lib/core\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 594);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/core/lib/core/error_codes.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto
namespace diplomacy {
namespace tensorflow {
namespace error {
const ::google::protobuf::EnumDescriptor* Code_descriptor() {
  protobuf_diplomacy_5ftensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_diplomacy_5ftensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto::file_level_enum_descriptors[0];
}
bool Code_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 20:
      return true;
    default:
      return false;
  }
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace error
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)