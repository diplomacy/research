// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/protobuf/critical_section.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[2];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto
namespace diplomacy {
namespace tensorflow {
class CriticalSectionDef;
class CriticalSectionDefDefaultTypeInternal;
extern CriticalSectionDefDefaultTypeInternal _CriticalSectionDef_default_instance_;
class CriticalSectionExecutionDef;
class CriticalSectionExecutionDefDefaultTypeInternal;
extern CriticalSectionExecutionDefDefaultTypeInternal _CriticalSectionExecutionDef_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> ::diplomacy::tensorflow::CriticalSectionDef* Arena::CreateMaybeMessage<::diplomacy::tensorflow::CriticalSectionDef>(Arena*);
template<> ::diplomacy::tensorflow::CriticalSectionExecutionDef* Arena::CreateMaybeMessage<::diplomacy::tensorflow::CriticalSectionExecutionDef>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace diplomacy {
namespace tensorflow {

// ===================================================================

class CriticalSectionDef : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.CriticalSectionDef) */ {
 public:
  CriticalSectionDef();
  virtual ~CriticalSectionDef();

  CriticalSectionDef(const CriticalSectionDef& from);

  inline CriticalSectionDef& operator=(const CriticalSectionDef& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CriticalSectionDef(CriticalSectionDef&& from) noexcept
    : CriticalSectionDef() {
    *this = ::std::move(from);
  }

  inline CriticalSectionDef& operator=(CriticalSectionDef&& from) noexcept {
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
  static const CriticalSectionDef& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CriticalSectionDef* internal_default_instance() {
    return reinterpret_cast<const CriticalSectionDef*>(
               &_CriticalSectionDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(CriticalSectionDef* other);
  void Swap(CriticalSectionDef* other);
  friend void swap(CriticalSectionDef& a, CriticalSectionDef& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CriticalSectionDef* New() const final {
    return CreateMaybeMessage<CriticalSectionDef>(NULL);
  }

  CriticalSectionDef* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CriticalSectionDef>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CriticalSectionDef& from);
  void MergeFrom(const CriticalSectionDef& from);
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
  void InternalSwap(CriticalSectionDef* other);
  protected:
  explicit CriticalSectionDef(::google::protobuf::Arena* arena);
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

  // string critical_section_name = 1;
  void clear_critical_section_name();
  static const int kCriticalSectionNameFieldNumber = 1;
  const ::std::string& critical_section_name() const;
  void set_critical_section_name(const ::std::string& value);
  #if LANG_CXX11
  void set_critical_section_name(::std::string&& value);
  #endif
  void set_critical_section_name(const char* value);
  void set_critical_section_name(const char* value, size_t size);
  ::std::string* mutable_critical_section_name();
  ::std::string* release_critical_section_name();
  void set_allocated_critical_section_name(::std::string* critical_section_name);
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  ::std::string* unsafe_arena_release_critical_section_name();
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_critical_section_name(
      ::std::string* critical_section_name);

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.CriticalSectionDef)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::ArenaStringPtr critical_section_name_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class CriticalSectionExecutionDef : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.CriticalSectionExecutionDef) */ {
 public:
  CriticalSectionExecutionDef();
  virtual ~CriticalSectionExecutionDef();

  CriticalSectionExecutionDef(const CriticalSectionExecutionDef& from);

  inline CriticalSectionExecutionDef& operator=(const CriticalSectionExecutionDef& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CriticalSectionExecutionDef(CriticalSectionExecutionDef&& from) noexcept
    : CriticalSectionExecutionDef() {
    *this = ::std::move(from);
  }

  inline CriticalSectionExecutionDef& operator=(CriticalSectionExecutionDef&& from) noexcept {
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
  static const CriticalSectionExecutionDef& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CriticalSectionExecutionDef* internal_default_instance() {
    return reinterpret_cast<const CriticalSectionExecutionDef*>(
               &_CriticalSectionExecutionDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void UnsafeArenaSwap(CriticalSectionExecutionDef* other);
  void Swap(CriticalSectionExecutionDef* other);
  friend void swap(CriticalSectionExecutionDef& a, CriticalSectionExecutionDef& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CriticalSectionExecutionDef* New() const final {
    return CreateMaybeMessage<CriticalSectionExecutionDef>(NULL);
  }

  CriticalSectionExecutionDef* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CriticalSectionExecutionDef>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CriticalSectionExecutionDef& from);
  void MergeFrom(const CriticalSectionExecutionDef& from);
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
  void InternalSwap(CriticalSectionExecutionDef* other);
  protected:
  explicit CriticalSectionExecutionDef(::google::protobuf::Arena* arena);
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

  // string execute_in_critical_section_name = 1;
  void clear_execute_in_critical_section_name();
  static const int kExecuteInCriticalSectionNameFieldNumber = 1;
  const ::std::string& execute_in_critical_section_name() const;
  void set_execute_in_critical_section_name(const ::std::string& value);
  #if LANG_CXX11
  void set_execute_in_critical_section_name(::std::string&& value);
  #endif
  void set_execute_in_critical_section_name(const char* value);
  void set_execute_in_critical_section_name(const char* value, size_t size);
  ::std::string* mutable_execute_in_critical_section_name();
  ::std::string* release_execute_in_critical_section_name();
  void set_allocated_execute_in_critical_section_name(::std::string* execute_in_critical_section_name);
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  ::std::string* unsafe_arena_release_execute_in_critical_section_name();
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_execute_in_critical_section_name(
      ::std::string* execute_in_critical_section_name);

  // bool exclusive_resource_access = 2;
  void clear_exclusive_resource_access();
  static const int kExclusiveResourceAccessFieldNumber = 2;
  bool exclusive_resource_access() const;
  void set_exclusive_resource_access(bool value);

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.CriticalSectionExecutionDef)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::ArenaStringPtr execute_in_critical_section_name_;
  bool exclusive_resource_access_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CriticalSectionDef

// string critical_section_name = 1;
inline void CriticalSectionDef::clear_critical_section_name() {
  critical_section_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& CriticalSectionDef::critical_section_name() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
  return critical_section_name_.Get();
}
inline void CriticalSectionDef::set_critical_section_name(const ::std::string& value) {
  
  critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}
#if LANG_CXX11
inline void CriticalSectionDef::set_critical_section_name(::std::string&& value) {
  
  critical_section_name_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}
#endif
inline void CriticalSectionDef::set_critical_section_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}
inline void CriticalSectionDef::set_critical_section_name(const char* value,
    size_t size) {
  
  critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}
inline ::std::string* CriticalSectionDef::mutable_critical_section_name() {
  
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
  return critical_section_name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* CriticalSectionDef::release_critical_section_name() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
  
  return critical_section_name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void CriticalSectionDef::set_allocated_critical_section_name(::std::string* critical_section_name) {
  if (critical_section_name != NULL) {
    
  } else {
    
  }
  critical_section_name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), critical_section_name,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}
inline ::std::string* CriticalSectionDef::unsafe_arena_release_critical_section_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return critical_section_name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void CriticalSectionDef::unsafe_arena_set_allocated_critical_section_name(
    ::std::string* critical_section_name) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (critical_section_name != NULL) {
    
  } else {
    
  }
  critical_section_name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      critical_section_name, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:diplomacy.tensorflow.CriticalSectionDef.critical_section_name)
}

// -------------------------------------------------------------------

// CriticalSectionExecutionDef

// string execute_in_critical_section_name = 1;
inline void CriticalSectionExecutionDef::clear_execute_in_critical_section_name() {
  execute_in_critical_section_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& CriticalSectionExecutionDef::execute_in_critical_section_name() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
  return execute_in_critical_section_name_.Get();
}
inline void CriticalSectionExecutionDef::set_execute_in_critical_section_name(const ::std::string& value) {
  
  execute_in_critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}
#if LANG_CXX11
inline void CriticalSectionExecutionDef::set_execute_in_critical_section_name(::std::string&& value) {
  
  execute_in_critical_section_name_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}
#endif
inline void CriticalSectionExecutionDef::set_execute_in_critical_section_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  execute_in_critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}
inline void CriticalSectionExecutionDef::set_execute_in_critical_section_name(const char* value,
    size_t size) {
  
  execute_in_critical_section_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}
inline ::std::string* CriticalSectionExecutionDef::mutable_execute_in_critical_section_name() {
  
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
  return execute_in_critical_section_name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* CriticalSectionExecutionDef::release_execute_in_critical_section_name() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
  
  return execute_in_critical_section_name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void CriticalSectionExecutionDef::set_allocated_execute_in_critical_section_name(::std::string* execute_in_critical_section_name) {
  if (execute_in_critical_section_name != NULL) {
    
  } else {
    
  }
  execute_in_critical_section_name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), execute_in_critical_section_name,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}
inline ::std::string* CriticalSectionExecutionDef::unsafe_arena_release_execute_in_critical_section_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return execute_in_critical_section_name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void CriticalSectionExecutionDef::unsafe_arena_set_allocated_execute_in_critical_section_name(
    ::std::string* execute_in_critical_section_name) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (execute_in_critical_section_name != NULL) {
    
  } else {
    
  }
  execute_in_critical_section_name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      execute_in_critical_section_name, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:diplomacy.tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name)
}

// bool exclusive_resource_access = 2;
inline void CriticalSectionExecutionDef::clear_exclusive_resource_access() {
  exclusive_resource_access_ = false;
}
inline bool CriticalSectionExecutionDef::exclusive_resource_access() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CriticalSectionExecutionDef.exclusive_resource_access)
  return exclusive_resource_access_;
}
inline void CriticalSectionExecutionDef::set_exclusive_resource_access(bool value) {
  
  exclusive_resource_access_ = value;
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CriticalSectionExecutionDef.exclusive_resource_access)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow
}  // namespace diplomacy

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcritical_5fsection_2eproto
