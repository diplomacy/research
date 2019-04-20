// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.proto

#ifndef PROTOBUF_INCLUDED_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto
#define PROTOBUF_INCLUDED_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto 

namespace protobuf_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto
namespace tensorflow {
namespace serving {
namespace test_util {
class StoragePathErrorInjectingSourceAdapterConfig;
class StoragePathErrorInjectingSourceAdapterConfigDefaultTypeInternal;
extern StoragePathErrorInjectingSourceAdapterConfigDefaultTypeInternal _StoragePathErrorInjectingSourceAdapterConfig_default_instance_;
}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
namespace google {
namespace protobuf {
template<> ::tensorflow::serving::test_util::StoragePathErrorInjectingSourceAdapterConfig* Arena::CreateMaybeMessage<::tensorflow::serving::test_util::StoragePathErrorInjectingSourceAdapterConfig>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace tensorflow {
namespace serving {
namespace test_util {

// ===================================================================

class StoragePathErrorInjectingSourceAdapterConfig : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig) */ {
 public:
  StoragePathErrorInjectingSourceAdapterConfig();
  virtual ~StoragePathErrorInjectingSourceAdapterConfig();

  StoragePathErrorInjectingSourceAdapterConfig(const StoragePathErrorInjectingSourceAdapterConfig& from);

  inline StoragePathErrorInjectingSourceAdapterConfig& operator=(const StoragePathErrorInjectingSourceAdapterConfig& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  StoragePathErrorInjectingSourceAdapterConfig(StoragePathErrorInjectingSourceAdapterConfig&& from) noexcept
    : StoragePathErrorInjectingSourceAdapterConfig() {
    *this = ::std::move(from);
  }

  inline StoragePathErrorInjectingSourceAdapterConfig& operator=(StoragePathErrorInjectingSourceAdapterConfig&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const StoragePathErrorInjectingSourceAdapterConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const StoragePathErrorInjectingSourceAdapterConfig* internal_default_instance() {
    return reinterpret_cast<const StoragePathErrorInjectingSourceAdapterConfig*>(
               &_StoragePathErrorInjectingSourceAdapterConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(StoragePathErrorInjectingSourceAdapterConfig* other);
  friend void swap(StoragePathErrorInjectingSourceAdapterConfig& a, StoragePathErrorInjectingSourceAdapterConfig& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline StoragePathErrorInjectingSourceAdapterConfig* New() const final {
    return CreateMaybeMessage<StoragePathErrorInjectingSourceAdapterConfig>(NULL);
  }

  StoragePathErrorInjectingSourceAdapterConfig* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<StoragePathErrorInjectingSourceAdapterConfig>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const StoragePathErrorInjectingSourceAdapterConfig& from);
  void MergeFrom(const StoragePathErrorInjectingSourceAdapterConfig& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(StoragePathErrorInjectingSourceAdapterConfig* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string error_message = 1;
  void clear_error_message();
  static const int kErrorMessageFieldNumber = 1;
  const ::std::string& error_message() const;
  void set_error_message(const ::std::string& value);
  #if LANG_CXX11
  void set_error_message(::std::string&& value);
  #endif
  void set_error_message(const char* value);
  void set_error_message(const char* value, size_t size);
  ::std::string* mutable_error_message();
  ::std::string* release_error_message();
  void set_allocated_error_message(::std::string* error_message);

  // @@protoc_insertion_point(class_scope:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr error_message_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// StoragePathErrorInjectingSourceAdapterConfig

// string error_message = 1;
inline void StoragePathErrorInjectingSourceAdapterConfig::clear_error_message() {
  error_message_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& StoragePathErrorInjectingSourceAdapterConfig::error_message() const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
  return error_message_.GetNoArena();
}
inline void StoragePathErrorInjectingSourceAdapterConfig::set_error_message(const ::std::string& value) {
  
  error_message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
}
#if LANG_CXX11
inline void StoragePathErrorInjectingSourceAdapterConfig::set_error_message(::std::string&& value) {
  
  error_message_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
}
#endif
inline void StoragePathErrorInjectingSourceAdapterConfig::set_error_message(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  error_message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
}
inline void StoragePathErrorInjectingSourceAdapterConfig::set_error_message(const char* value, size_t size) {
  
  error_message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
}
inline ::std::string* StoragePathErrorInjectingSourceAdapterConfig::mutable_error_message() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
  return error_message_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* StoragePathErrorInjectingSourceAdapterConfig::release_error_message() {
  // @@protoc_insertion_point(field_release:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
  
  return error_message_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void StoragePathErrorInjectingSourceAdapterConfig::set_allocated_error_message(::std::string* error_message) {
  if (error_message != NULL) {
    
  } else {
    
  }
  error_message_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), error_message);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.test_util.StoragePathErrorInjectingSourceAdapterConfig.error_message)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_tensorflow_5fserving_2fmodel_5fservers_2ftest_5futil_2fstorage_5fpath_5ferror_5finjecting_5fsource_5fadapter_2eproto
