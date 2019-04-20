// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/compiler/xla/rpc/xla_service.proto

#include "diplomacy_tensorflow/compiler/xla/rpc/xla_service.pb.h"

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

namespace xla {
}  // namespace xla
namespace protobuf_diplomacy_5ftensorflow_2fcompiler_2fxla_2frpc_2fxla_5fservice_2eproto {
void InitDefaults() {
}

const ::google::protobuf::uint32 TableStruct::offsets[1] = {};
static const ::google::protobuf::internal::MigrationSchema* schemas = NULL;
static const ::google::protobuf::Message* const* file_default_instances = NULL;

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/compiler/xla/rpc/xla_service.proto", schemas, file_default_instances, TableStruct::offsets,
      NULL, NULL, NULL);
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
      "\n7diplomacy_tensorflow/compiler/xla/rpc/"
      "xla_service.proto\022\003xla\032+diplomacy_tensor"
      "flow/compiler/xla/xla.proto2\352\n\n\nXlaServi"
      "ce\022\?\n\nUnregister\022\026.xla.UnregisterRequest"
      "\032\027.xla.UnregisterResponse\"\000\022Q\n\020Deconstru"
      "ctTuple\022\034.xla.DeconstructTupleRequest\032\035."
      "xla.DeconstructTupleResponse\"\000\0223\n\006Unpack"
      "\022\022.xla.UnpackRequest\032\023.xla.UnpackRespons"
      "e\"\000\0229\n\010GetShape\022\024.xla.GetShapeRequest\032\025."
      "xla.GetShapeResponse\"\000\022^\n\030GetComputation"
      "GraphStats\022!.xla.ComputationGraphStatsRe"
      "quest\032\035.xla.ComputationStatsResponse\"\000\0229"
      "\n\010LoadData\022\024.xla.LoadDataRequest\032\025.xla.L"
      "oadDataResponse\"\000\022Q\n\020TransferToClient\022\034."
      "xla.TransferToClientRequest\032\035.xla.Transf"
      "erToClientResponse\"\000\022Q\n\020TransferToServer"
      "\022\034.xla.TransferToServerRequest\032\035.xla.Tra"
      "nsferToServerResponse\"\000\022Q\n\020TransferToInf"
      "eed\022\034.xla.TransferToInfeedRequest\032\035.xla."
      "TransferToInfeedResponse\"\000\022Z\n\023TransferFr"
      "omOutfeed\022\037.xla.TransferFromOutfeedReque"
      "st\032 .xla.TransferFromOutfeedResponse\"\000\022B"
      "\n\013ResetDevice\022\027.xla.ResetDeviceRequest\032\030"
      ".xla.ResetDeviceResponse\"\000\022X\n\024ComputeCon"
      "stantGraph\022 .xla.ComputeConstantGraphReq"
      "uest\032\034.xla.ComputeConstantResponse\"\000\022Q\n\020"
      "GetDeviceHandles\022\034.xla.GetDeviceHandlesR"
      "equest\032\035.xla.GetDeviceHandlesResponse\"\000\022"
      "Z\n\023CreateChannelHandle\022\037.xla.CreateChann"
      "elHandleRequest\032 .xla.CreateChannelHandl"
      "eResponse\"\000\0226\n\007Compile\022\023.xla.CompileRequ"
      "est\032\024.xla.CompileResponse\"\000\0226\n\007Execute\022\023"
      ".xla.ExecuteRequest\032\024.xla.ExecuteRespons"
      "e\"\000\022X\n\024ExecuteGraphParallel\022 .xla.Execut"
      "eGraphParallelRequest\032\034.xla.ExecuteParal"
      "lelResponse\"\000\022Q\n\020WaitForExecution\022\034.xla."
      "WaitForExecutionRequest\032\035.xla.WaitForExe"
      "cutionResponse\"\000b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 1504);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/compiler/xla/rpc/xla_service.proto", &protobuf_RegisterTypes);
  ::protobuf_diplomacy_5ftensorflow_2fcompiler_2fxla_2fxla_2eproto::AddDescriptors();
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcompiler_2fxla_2frpc_2fxla_5fservice_2eproto
namespace xla {

// @@protoc_insertion_point(namespace_scope)
}  // namespace xla
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
