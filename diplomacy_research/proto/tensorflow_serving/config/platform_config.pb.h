// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/config/platform_config.proto

#ifndef PROTOBUF_INCLUDED_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto
#define PROTOBUF_INCLUDED_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto

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
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/any.pb.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto 

namespace protobuf_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[3];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto
namespace tensorflow {
namespace serving {
class PlatformConfig;
class PlatformConfigDefaultTypeInternal;
extern PlatformConfigDefaultTypeInternal _PlatformConfig_default_instance_;
class PlatformConfigMap;
class PlatformConfigMapDefaultTypeInternal;
extern PlatformConfigMapDefaultTypeInternal _PlatformConfigMap_default_instance_;
class PlatformConfigMap_PlatformConfigsEntry_DoNotUse;
class PlatformConfigMap_PlatformConfigsEntry_DoNotUseDefaultTypeInternal;
extern PlatformConfigMap_PlatformConfigsEntry_DoNotUseDefaultTypeInternal _PlatformConfigMap_PlatformConfigsEntry_DoNotUse_default_instance_;
}  // namespace serving
}  // namespace tensorflow
namespace google {
namespace protobuf {
template<> ::tensorflow::serving::PlatformConfig* Arena::CreateMaybeMessage<::tensorflow::serving::PlatformConfig>(Arena*);
template<> ::tensorflow::serving::PlatformConfigMap* Arena::CreateMaybeMessage<::tensorflow::serving::PlatformConfigMap>(Arena*);
template<> ::tensorflow::serving::PlatformConfigMap_PlatformConfigsEntry_DoNotUse* Arena::CreateMaybeMessage<::tensorflow::serving::PlatformConfigMap_PlatformConfigsEntry_DoNotUse>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace tensorflow {
namespace serving {

// ===================================================================

class PlatformConfig : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:tensorflow.serving.PlatformConfig) */ {
 public:
  PlatformConfig();
  virtual ~PlatformConfig();

  PlatformConfig(const PlatformConfig& from);

  inline PlatformConfig& operator=(const PlatformConfig& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  PlatformConfig(PlatformConfig&& from) noexcept
    : PlatformConfig() {
    *this = ::std::move(from);
  }

  inline PlatformConfig& operator=(PlatformConfig&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline ::google::protobuf::Arena* GetArena() const final {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const final {
    return MaybeArenaPtr();
  }
  static const ::google::protobuf::Descriptor* descriptor();
  static const PlatformConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PlatformConfig* internal_default_instance() {
    return reinterpret_cast<const PlatformConfig*>(
               &_PlatformConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(PlatformConfig* other);
  void Swap(PlatformConfig* other);
  friend void swap(PlatformConfig& a, PlatformConfig& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline PlatformConfig* New() const final {
    return CreateMaybeMessage<PlatformConfig>(NULL);
  }

  PlatformConfig* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<PlatformConfig>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const PlatformConfig& from);
  void MergeFrom(const PlatformConfig& from);
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
  void InternalSwap(PlatformConfig* other);
  protected:
  explicit PlatformConfig(::google::protobuf::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .google.protobuf.Any source_adapter_config = 1;
  bool has_source_adapter_config() const;
  void clear_source_adapter_config();
  static const int kSourceAdapterConfigFieldNumber = 1;
  private:
  const ::google::protobuf::Any& _internal_source_adapter_config() const;
  public:
  const ::google::protobuf::Any& source_adapter_config() const;
  ::google::protobuf::Any* release_source_adapter_config();
  ::google::protobuf::Any* mutable_source_adapter_config();
  void set_allocated_source_adapter_config(::google::protobuf::Any* source_adapter_config);
  void unsafe_arena_set_allocated_source_adapter_config(
      ::google::protobuf::Any* source_adapter_config);
  ::google::protobuf::Any* unsafe_arena_release_source_adapter_config();

  // @@protoc_insertion_point(class_scope:tensorflow.serving.PlatformConfig)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::Any* source_adapter_config_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class PlatformConfigMap_PlatformConfigsEntry_DoNotUse : public ::google::protobuf::internal::MapEntry<PlatformConfigMap_PlatformConfigsEntry_DoNotUse, 
    ::std::string, ::tensorflow::serving::PlatformConfig,
    ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
    ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
    0 > {
public:
  typedef ::google::protobuf::internal::MapEntry<PlatformConfigMap_PlatformConfigsEntry_DoNotUse, 
    ::std::string, ::tensorflow::serving::PlatformConfig,
    ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
    ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
    0 > SuperType;
  PlatformConfigMap_PlatformConfigsEntry_DoNotUse();
  PlatformConfigMap_PlatformConfigsEntry_DoNotUse(::google::protobuf::Arena* arena);
  void MergeFrom(const PlatformConfigMap_PlatformConfigsEntry_DoNotUse& other);
  static const PlatformConfigMap_PlatformConfigsEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const PlatformConfigMap_PlatformConfigsEntry_DoNotUse*>(&_PlatformConfigMap_PlatformConfigsEntry_DoNotUse_default_instance_); }
  void MergeFrom(const ::google::protobuf::Message& other) final;
  ::google::protobuf::Metadata GetMetadata() const;
};

// -------------------------------------------------------------------

class PlatformConfigMap : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:tensorflow.serving.PlatformConfigMap) */ {
 public:
  PlatformConfigMap();
  virtual ~PlatformConfigMap();

  PlatformConfigMap(const PlatformConfigMap& from);

  inline PlatformConfigMap& operator=(const PlatformConfigMap& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  PlatformConfigMap(PlatformConfigMap&& from) noexcept
    : PlatformConfigMap() {
    *this = ::std::move(from);
  }

  inline PlatformConfigMap& operator=(PlatformConfigMap&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline ::google::protobuf::Arena* GetArena() const final {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const final {
    return MaybeArenaPtr();
  }
  static const ::google::protobuf::Descriptor* descriptor();
  static const PlatformConfigMap& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PlatformConfigMap* internal_default_instance() {
    return reinterpret_cast<const PlatformConfigMap*>(
               &_PlatformConfigMap_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void UnsafeArenaSwap(PlatformConfigMap* other);
  void Swap(PlatformConfigMap* other);
  friend void swap(PlatformConfigMap& a, PlatformConfigMap& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline PlatformConfigMap* New() const final {
    return CreateMaybeMessage<PlatformConfigMap>(NULL);
  }

  PlatformConfigMap* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<PlatformConfigMap>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const PlatformConfigMap& from);
  void MergeFrom(const PlatformConfigMap& from);
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
  void InternalSwap(PlatformConfigMap* other);
  protected:
  explicit PlatformConfigMap(::google::protobuf::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  // map<string, .tensorflow.serving.PlatformConfig> platform_configs = 1;
  int platform_configs_size() const;
  void clear_platform_configs();
  static const int kPlatformConfigsFieldNumber = 1;
  const ::google::protobuf::Map< ::std::string, ::tensorflow::serving::PlatformConfig >&
      platform_configs() const;
  ::google::protobuf::Map< ::std::string, ::tensorflow::serving::PlatformConfig >*
      mutable_platform_configs();

  // @@protoc_insertion_point(class_scope:tensorflow.serving.PlatformConfigMap)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::MapField<
      PlatformConfigMap_PlatformConfigsEntry_DoNotUse,
      ::std::string, ::tensorflow::serving::PlatformConfig,
      ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
      ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
      0 > platform_configs_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// PlatformConfig

// .google.protobuf.Any source_adapter_config = 1;
inline bool PlatformConfig::has_source_adapter_config() const {
  return this != internal_default_instance() && source_adapter_config_ != NULL;
}
inline const ::google::protobuf::Any& PlatformConfig::_internal_source_adapter_config() const {
  return *source_adapter_config_;
}
inline const ::google::protobuf::Any& PlatformConfig::source_adapter_config() const {
  const ::google::protobuf::Any* p = source_adapter_config_;
  // @@protoc_insertion_point(field_get:tensorflow.serving.PlatformConfig.source_adapter_config)
  return p != NULL ? *p : *reinterpret_cast<const ::google::protobuf::Any*>(
      &::google::protobuf::_Any_default_instance_);
}
inline ::google::protobuf::Any* PlatformConfig::release_source_adapter_config() {
  // @@protoc_insertion_point(field_release:tensorflow.serving.PlatformConfig.source_adapter_config)
  
  ::google::protobuf::Any* temp = source_adapter_config_;
  if (GetArenaNoVirtual() != NULL) {
    temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
  }
  source_adapter_config_ = NULL;
  return temp;
}
inline ::google::protobuf::Any* PlatformConfig::unsafe_arena_release_source_adapter_config() {
  // @@protoc_insertion_point(field_unsafe_arena_release:tensorflow.serving.PlatformConfig.source_adapter_config)
  
  ::google::protobuf::Any* temp = source_adapter_config_;
  source_adapter_config_ = NULL;
  return temp;
}
inline ::google::protobuf::Any* PlatformConfig::mutable_source_adapter_config() {
  
  if (source_adapter_config_ == NULL) {
    auto* p = CreateMaybeMessage<::google::protobuf::Any>(GetArenaNoVirtual());
    source_adapter_config_ = p;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.serving.PlatformConfig.source_adapter_config)
  return source_adapter_config_;
}
inline void PlatformConfig::set_allocated_source_adapter_config(::google::protobuf::Any* source_adapter_config) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(source_adapter_config_);
  }
  if (source_adapter_config) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      source_adapter_config = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, source_adapter_config, submessage_arena);
    }
    
  } else {
    
  }
  source_adapter_config_ = source_adapter_config;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.PlatformConfig.source_adapter_config)
}

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// PlatformConfigMap

// map<string, .tensorflow.serving.PlatformConfig> platform_configs = 1;
inline int PlatformConfigMap::platform_configs_size() const {
  return platform_configs_.size();
}
inline void PlatformConfigMap::clear_platform_configs() {
  platform_configs_.Clear();
}
inline const ::google::protobuf::Map< ::std::string, ::tensorflow::serving::PlatformConfig >&
PlatformConfigMap::platform_configs() const {
  // @@protoc_insertion_point(field_map:tensorflow.serving.PlatformConfigMap.platform_configs)
  return platform_configs_.GetMap();
}
inline ::google::protobuf::Map< ::std::string, ::tensorflow::serving::PlatformConfig >*
PlatformConfigMap::mutable_platform_configs() {
  // @@protoc_insertion_point(field_mutable_map:tensorflow.serving.PlatformConfigMap.platform_configs)
  return platform_configs_.MutableMap();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace serving
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_tensorflow_5fserving_2fconfig_2fplatform_5fconfig_2eproto
