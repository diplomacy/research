// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/sources/storage_path/static_storage_path_source.proto

#ifndef PROTOBUF_INCLUDED_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto
#define PROTOBUF_INCLUDED_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto 

namespace protobuf_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto {
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
}  // namespace protobuf_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto
namespace tensorflow {
namespace serving {
class StaticStoragePathSourceConfig;
class StaticStoragePathSourceConfigDefaultTypeInternal;
extern StaticStoragePathSourceConfigDefaultTypeInternal _StaticStoragePathSourceConfig_default_instance_;
}  // namespace serving
}  // namespace tensorflow
namespace google {
namespace protobuf {
template<> ::tensorflow::serving::StaticStoragePathSourceConfig* Arena::CreateMaybeMessage<::tensorflow::serving::StaticStoragePathSourceConfig>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace tensorflow {
namespace serving {

// ===================================================================

class StaticStoragePathSourceConfig : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:tensorflow.serving.StaticStoragePathSourceConfig) */ {
 public:
  StaticStoragePathSourceConfig();
  virtual ~StaticStoragePathSourceConfig();

  StaticStoragePathSourceConfig(const StaticStoragePathSourceConfig& from);

  inline StaticStoragePathSourceConfig& operator=(const StaticStoragePathSourceConfig& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  StaticStoragePathSourceConfig(StaticStoragePathSourceConfig&& from) noexcept
    : StaticStoragePathSourceConfig() {
    *this = ::std::move(from);
  }

  inline StaticStoragePathSourceConfig& operator=(StaticStoragePathSourceConfig&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const StaticStoragePathSourceConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const StaticStoragePathSourceConfig* internal_default_instance() {
    return reinterpret_cast<const StaticStoragePathSourceConfig*>(
               &_StaticStoragePathSourceConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(StaticStoragePathSourceConfig* other);
  friend void swap(StaticStoragePathSourceConfig& a, StaticStoragePathSourceConfig& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline StaticStoragePathSourceConfig* New() const final {
    return CreateMaybeMessage<StaticStoragePathSourceConfig>(NULL);
  }

  StaticStoragePathSourceConfig* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<StaticStoragePathSourceConfig>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const StaticStoragePathSourceConfig& from);
  void MergeFrom(const StaticStoragePathSourceConfig& from);
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
  void InternalSwap(StaticStoragePathSourceConfig* other);
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

  // string servable_name = 1;
  void clear_servable_name();
  static const int kServableNameFieldNumber = 1;
  const ::std::string& servable_name() const;
  void set_servable_name(const ::std::string& value);
  #if LANG_CXX11
  void set_servable_name(::std::string&& value);
  #endif
  void set_servable_name(const char* value);
  void set_servable_name(const char* value, size_t size);
  ::std::string* mutable_servable_name();
  ::std::string* release_servable_name();
  void set_allocated_servable_name(::std::string* servable_name);

  // string version_path = 3;
  void clear_version_path();
  static const int kVersionPathFieldNumber = 3;
  const ::std::string& version_path() const;
  void set_version_path(const ::std::string& value);
  #if LANG_CXX11
  void set_version_path(::std::string&& value);
  #endif
  void set_version_path(const char* value);
  void set_version_path(const char* value, size_t size);
  ::std::string* mutable_version_path();
  ::std::string* release_version_path();
  void set_allocated_version_path(::std::string* version_path);

  // int64 version_num = 2;
  void clear_version_num();
  static const int kVersionNumFieldNumber = 2;
  ::google::protobuf::int64 version_num() const;
  void set_version_num(::google::protobuf::int64 value);

  // @@protoc_insertion_point(class_scope:tensorflow.serving.StaticStoragePathSourceConfig)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr servable_name_;
  ::google::protobuf::internal::ArenaStringPtr version_path_;
  ::google::protobuf::int64 version_num_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// StaticStoragePathSourceConfig

// string servable_name = 1;
inline void StaticStoragePathSourceConfig::clear_servable_name() {
  servable_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& StaticStoragePathSourceConfig::servable_name() const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
  return servable_name_.GetNoArena();
}
inline void StaticStoragePathSourceConfig::set_servable_name(const ::std::string& value) {
  
  servable_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
}
#if LANG_CXX11
inline void StaticStoragePathSourceConfig::set_servable_name(::std::string&& value) {
  
  servable_name_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
}
#endif
inline void StaticStoragePathSourceConfig::set_servable_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  servable_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
}
inline void StaticStoragePathSourceConfig::set_servable_name(const char* value, size_t size) {
  
  servable_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
}
inline ::std::string* StaticStoragePathSourceConfig::mutable_servable_name() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
  return servable_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* StaticStoragePathSourceConfig::release_servable_name() {
  // @@protoc_insertion_point(field_release:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
  
  return servable_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void StaticStoragePathSourceConfig::set_allocated_servable_name(::std::string* servable_name) {
  if (servable_name != NULL) {
    
  } else {
    
  }
  servable_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), servable_name);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.StaticStoragePathSourceConfig.servable_name)
}

// int64 version_num = 2;
inline void StaticStoragePathSourceConfig::clear_version_num() {
  version_num_ = GOOGLE_LONGLONG(0);
}
inline ::google::protobuf::int64 StaticStoragePathSourceConfig::version_num() const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.StaticStoragePathSourceConfig.version_num)
  return version_num_;
}
inline void StaticStoragePathSourceConfig::set_version_num(::google::protobuf::int64 value) {
  
  version_num_ = value;
  // @@protoc_insertion_point(field_set:tensorflow.serving.StaticStoragePathSourceConfig.version_num)
}

// string version_path = 3;
inline void StaticStoragePathSourceConfig::clear_version_path() {
  version_path_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& StaticStoragePathSourceConfig::version_path() const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
  return version_path_.GetNoArena();
}
inline void StaticStoragePathSourceConfig::set_version_path(const ::std::string& value) {
  
  version_path_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
}
#if LANG_CXX11
inline void StaticStoragePathSourceConfig::set_version_path(::std::string&& value) {
  
  version_path_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
}
#endif
inline void StaticStoragePathSourceConfig::set_version_path(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  version_path_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
}
inline void StaticStoragePathSourceConfig::set_version_path(const char* value, size_t size) {
  
  version_path_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
}
inline ::std::string* StaticStoragePathSourceConfig::mutable_version_path() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
  return version_path_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* StaticStoragePathSourceConfig::release_version_path() {
  // @@protoc_insertion_point(field_release:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
  
  return version_path_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void StaticStoragePathSourceConfig::set_allocated_version_path(::std::string* version_path) {
  if (version_path != NULL) {
    
  } else {
    
  }
  version_path_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), version_path);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.StaticStoragePathSourceConfig.version_path)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace serving
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_tensorflow_5fserving_2fsources_2fstorage_5fpath_2fstatic_5fstorage_5fpath_5fsource_2eproto
