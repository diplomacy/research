// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: tensorflow_serving/apis/model_service.proto

#include "tensorflow_serving/apis/model_service.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"

#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_unary_call.h>
#include <grpcpp/impl/codegen/method_handler_impl.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/sync_stream.h>
namespace tensorflow {
namespace serving {

static const char* ModelService_method_names[] = {
  "/tensorflow.serving.ModelService/GetModelStatus",
  "/tensorflow.serving.ModelService/HandleReloadConfigRequest",
};

std::unique_ptr< ModelService::Stub> ModelService::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< ModelService::Stub> stub(new ModelService::Stub(channel));
  return stub;
}

ModelService::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_GetModelStatus_(ModelService_method_names[0], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_HandleReloadConfigRequest_(ModelService_method_names[1], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status ModelService::Stub::GetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::tensorflow::serving::GetModelStatusResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_GetModelStatus_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>* ModelService::Stub::AsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory< ::tensorflow::serving::GetModelStatusResponse>::Create(channel_.get(), cq, rpcmethod_GetModelStatus_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>* ModelService::Stub::PrepareAsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory< ::tensorflow::serving::GetModelStatusResponse>::Create(channel_.get(), cq, rpcmethod_GetModelStatus_, context, request, false);
}

::grpc::Status ModelService::Stub::HandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::tensorflow::serving::ReloadConfigResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_HandleReloadConfigRequest_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>* ModelService::Stub::AsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory< ::tensorflow::serving::ReloadConfigResponse>::Create(channel_.get(), cq, rpcmethod_HandleReloadConfigRequest_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>* ModelService::Stub::PrepareAsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory< ::tensorflow::serving::ReloadConfigResponse>::Create(channel_.get(), cq, rpcmethod_HandleReloadConfigRequest_, context, request, false);
}

ModelService::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      ModelService_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< ModelService::Service, ::tensorflow::serving::GetModelStatusRequest, ::tensorflow::serving::GetModelStatusResponse>(
          std::mem_fn(&ModelService::Service::GetModelStatus), this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      ModelService_method_names[1],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< ModelService::Service, ::tensorflow::serving::ReloadConfigRequest, ::tensorflow::serving::ReloadConfigResponse>(
          std::mem_fn(&ModelService::Service::HandleReloadConfigRequest), this)));
}

ModelService::Service::~Service() {
}

::grpc::Status ModelService::Service::GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status ModelService::Service::HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace tensorflow
}  // namespace serving

