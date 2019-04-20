// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: tensorflow_serving/apis/model_service.proto
#ifndef GRPC_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto__INCLUDED
#define GRPC_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto__INCLUDED

#include "tensorflow_serving/apis/model_service.pb.h"

#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/method_handler_impl.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace tensorflow {
namespace serving {

// ModelService provides methods to query and update the state of the server,
// e.g. which models/versions are being served.
class ModelService final {
 public:
  static constexpr char const* service_full_name() {
    return "tensorflow.serving.ModelService";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Gets status of model. If the ModelSpec in the request does not specify
    // version, information about all versions of the model will be returned. If
    // the ModelSpec in the request does specify a version, the status of only
    // that version will be returned.
    virtual ::grpc::Status GetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::tensorflow::serving::GetModelStatusResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>> AsyncGetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>>(AsyncGetModelStatusRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>> PrepareAsyncGetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>>(PrepareAsyncGetModelStatusRaw(context, request, cq));
    }
    // Reloads the set of served models. The new config supersedes the old one,
    // so if a model is omitted from the new config it will be unloaded and no
    // longer served.
    virtual ::grpc::Status HandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::tensorflow::serving::ReloadConfigResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>> AsyncHandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>>(AsyncHandleReloadConfigRequestRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>> PrepareAsyncHandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>>(PrepareAsyncHandleReloadConfigRequestRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>* AsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelStatusResponse>* PrepareAsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>* AsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ReloadConfigResponse>* PrepareAsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::tensorflow::serving::GetModelStatusResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>> AsyncGetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>>(AsyncGetModelStatusRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>> PrepareAsyncGetModelStatus(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>>(PrepareAsyncGetModelStatusRaw(context, request, cq));
    }
    ::grpc::Status HandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::tensorflow::serving::ReloadConfigResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>> AsyncHandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>>(AsyncHandleReloadConfigRequestRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>> PrepareAsyncHandleReloadConfigRequest(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>>(PrepareAsyncHandleReloadConfigRequestRaw(context, request, cq));
    }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>* AsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::GetModelStatusResponse>* PrepareAsyncGetModelStatusRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelStatusRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>* AsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::ReloadConfigResponse>* PrepareAsyncHandleReloadConfigRequestRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ReloadConfigRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_GetModelStatus_;
    const ::grpc::internal::RpcMethod rpcmethod_HandleReloadConfigRequest_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Gets status of model. If the ModelSpec in the request does not specify
    // version, information about all versions of the model will be returned. If
    // the ModelSpec in the request does specify a version, the status of only
    // that version will be returned.
    virtual ::grpc::Status GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response);
    // Reloads the set of served models. The new config supersedes the old one,
    // so if a model is omitted from the new config it will be unloaded and no
    // longer served.
    virtual ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_GetModelStatus : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_GetModelStatus() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_GetModelStatus() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetModelStatus(::grpc::ServerContext* context, ::tensorflow::serving::GetModelStatusRequest* request, ::grpc::ServerAsyncResponseWriter< ::tensorflow::serving::GetModelStatusResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_HandleReloadConfigRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_HandleReloadConfigRequest() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_HandleReloadConfigRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestHandleReloadConfigRequest(::grpc::ServerContext* context, ::tensorflow::serving::ReloadConfigRequest* request, ::grpc::ServerAsyncResponseWriter< ::tensorflow::serving::ReloadConfigResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_GetModelStatus<WithAsyncMethod_HandleReloadConfigRequest<Service > > AsyncService;
  template <class BaseClass>
  class WithGenericMethod_GetModelStatus : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_GetModelStatus() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_GetModelStatus() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_HandleReloadConfigRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_HandleReloadConfigRequest() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_HandleReloadConfigRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_GetModelStatus : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_GetModelStatus() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_GetModelStatus() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetModelStatus(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_HandleReloadConfigRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_HandleReloadConfigRequest() {
      ::grpc::Service::MarkMethodRaw(1);
    }
    ~WithRawMethod_HandleReloadConfigRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestHandleReloadConfigRequest(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetModelStatus : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_GetModelStatus() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::tensorflow::serving::GetModelStatusRequest, ::tensorflow::serving::GetModelStatusResponse>(std::bind(&WithStreamedUnaryMethod_GetModelStatus<BaseClass>::StreamedGetModelStatus, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetModelStatus() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetModelStatus(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelStatusRequest* request, ::tensorflow::serving::GetModelStatusResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetModelStatus(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::tensorflow::serving::GetModelStatusRequest,::tensorflow::serving::GetModelStatusResponse>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_HandleReloadConfigRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_HandleReloadConfigRequest() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< ::tensorflow::serving::ReloadConfigRequest, ::tensorflow::serving::ReloadConfigResponse>(std::bind(&WithStreamedUnaryMethod_HandleReloadConfigRequest<BaseClass>::StreamedHandleReloadConfigRequest, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_HandleReloadConfigRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context, const ::tensorflow::serving::ReloadConfigRequest* request, ::tensorflow::serving::ReloadConfigResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedHandleReloadConfigRequest(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::tensorflow::serving::ReloadConfigRequest,::tensorflow::serving::ReloadConfigResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_GetModelStatus<WithStreamedUnaryMethod_HandleReloadConfigRequest<Service > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_GetModelStatus<WithStreamedUnaryMethod_HandleReloadConfigRequest<Service > > StreamedService;
};

}  // namespace serving
}  // namespace tensorflow


#endif  // GRPC_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto__INCLUDED
