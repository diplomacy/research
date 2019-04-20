// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/protobuf/named_tensor.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto

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
#include "diplomacy_tensorflow/core/framework/tensor.pb.h"
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto {
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto
namespace diplomacy {
namespace tensorflow {
class NamedTensorProto;
class NamedTensorProtoDefaultTypeInternal;
extern NamedTensorProtoDefaultTypeInternal _NamedTensorProto_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> ::diplomacy::tensorflow::NamedTensorProto* Arena::CreateMaybeMessage<::diplomacy::tensorflow::NamedTensorProto>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace diplomacy {
namespace tensorflow {

// ===================================================================

class NamedTensorProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.NamedTensorProto) */ {
 public:
  NamedTensorProto();
  virtual ~NamedTensorProto();

  NamedTensorProto(const NamedTensorProto& from);

  inline NamedTensorProto& operator=(const NamedTensorProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  NamedTensorProto(NamedTensorProto&& from) noexcept
    : NamedTensorProto() {
    *this = ::std::move(from);
  }

  inline NamedTensorProto& operator=(NamedTensorProto&& from) noexcept {
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
  static const NamedTensorProto& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const NamedTensorProto* internal_default_instance() {
    return reinterpret_cast<const NamedTensorProto*>(
               &_NamedTensorProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(NamedTensorProto* other);
  void Swap(NamedTensorProto* other);
  friend void swap(NamedTensorProto& a, NamedTensorProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline NamedTensorProto* New() const final {
    return CreateMaybeMessage<NamedTensorProto>(NULL);
  }

  NamedTensorProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<NamedTensorProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const NamedTensorProto& from);
  void MergeFrom(const NamedTensorProto& from);
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
  void InternalSwap(NamedTensorProto* other);
  protected:
  explicit NamedTensorProto(::google::protobuf::Arena* arena);
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

  // string name = 1;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  #if LANG_CXX11
  void set_name(::std::string&& value);
  #endif
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  ::std::string* unsafe_arena_release_name();
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_name(
      ::std::string* name);

  // .diplomacy.tensorflow.TensorProto tensor = 2;
  bool has_tensor() const;
  void clear_tensor();
  static const int kTensorFieldNumber = 2;
  private:
  const ::diplomacy::tensorflow::TensorProto& _internal_tensor() const;
  public:
  const ::diplomacy::tensorflow::TensorProto& tensor() const;
  ::diplomacy::tensorflow::TensorProto* release_tensor();
  ::diplomacy::tensorflow::TensorProto* mutable_tensor();
  void set_allocated_tensor(::diplomacy::tensorflow::TensorProto* tensor);
  void unsafe_arena_set_allocated_tensor(
      ::diplomacy::tensorflow::TensorProto* tensor);
  ::diplomacy::tensorflow::TensorProto* unsafe_arena_release_tensor();

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.NamedTensorProto)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::diplomacy::tensorflow::TensorProto* tensor_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// NamedTensorProto

// string name = 1;
inline void NamedTensorProto::clear_name() {
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& NamedTensorProto::name() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.NamedTensorProto.name)
  return name_.Get();
}
inline void NamedTensorProto::set_name(const ::std::string& value) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.NamedTensorProto.name)
}
#if LANG_CXX11
inline void NamedTensorProto::set_name(::std::string&& value) {
  
  name_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:diplomacy.tensorflow.NamedTensorProto.name)
}
#endif
inline void NamedTensorProto::set_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.NamedTensorProto.name)
}
inline void NamedTensorProto::set_name(const char* value,
    size_t size) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.NamedTensorProto.name)
}
inline ::std::string* NamedTensorProto::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.NamedTensorProto.name)
  return name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* NamedTensorProto::release_name() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.NamedTensorProto.name)
  
  return name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void NamedTensorProto::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.NamedTensorProto.name)
}
inline ::std::string* NamedTensorProto::unsafe_arena_release_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.NamedTensorProto.name)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void NamedTensorProto::unsafe_arena_set_allocated_name(
    ::std::string* name) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (name != NULL) {
    
  } else {
    
  }
  name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      name, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:diplomacy.tensorflow.NamedTensorProto.name)
}

// .diplomacy.tensorflow.TensorProto tensor = 2;
inline bool NamedTensorProto::has_tensor() const {
  return this != internal_default_instance() && tensor_ != NULL;
}
inline const ::diplomacy::tensorflow::TensorProto& NamedTensorProto::_internal_tensor() const {
  return *tensor_;
}
inline const ::diplomacy::tensorflow::TensorProto& NamedTensorProto::tensor() const {
  const ::diplomacy::tensorflow::TensorProto* p = tensor_;
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.NamedTensorProto.tensor)
  return p != NULL ? *p : *reinterpret_cast<const ::diplomacy::tensorflow::TensorProto*>(
      &::diplomacy::tensorflow::_TensorProto_default_instance_);
}
inline ::diplomacy::tensorflow::TensorProto* NamedTensorProto::release_tensor() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.NamedTensorProto.tensor)
  
  ::diplomacy::tensorflow::TensorProto* temp = tensor_;
  if (GetArenaNoVirtual() != NULL) {
    temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
  }
  tensor_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::TensorProto* NamedTensorProto::unsafe_arena_release_tensor() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.NamedTensorProto.tensor)
  
  ::diplomacy::tensorflow::TensorProto* temp = tensor_;
  tensor_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::TensorProto* NamedTensorProto::mutable_tensor() {
  
  if (tensor_ == NULL) {
    auto* p = CreateMaybeMessage<::diplomacy::tensorflow::TensorProto>(GetArenaNoVirtual());
    tensor_ = p;
  }
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.NamedTensorProto.tensor)
  return tensor_;
}
inline void NamedTensorProto::set_allocated_tensor(::diplomacy::tensorflow::TensorProto* tensor) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(tensor_);
  }
  if (tensor) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(tensor)->GetArena();
    if (message_arena != submessage_arena) {
      tensor = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, tensor, submessage_arena);
    }
    
  } else {
    
  }
  tensor_ = tensor;
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.NamedTensorProto.tensor)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow
}  // namespace diplomacy

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto
