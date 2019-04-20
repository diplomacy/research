// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/example/example.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto

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
#include "diplomacy_tensorflow/core/example/feature.pb.h"
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto {
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto
namespace diplomacy {
namespace tensorflow {
class Example;
class ExampleDefaultTypeInternal;
extern ExampleDefaultTypeInternal _Example_default_instance_;
class SequenceExample;
class SequenceExampleDefaultTypeInternal;
extern SequenceExampleDefaultTypeInternal _SequenceExample_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> ::diplomacy::tensorflow::Example* Arena::CreateMaybeMessage<::diplomacy::tensorflow::Example>(Arena*);
template<> ::diplomacy::tensorflow::SequenceExample* Arena::CreateMaybeMessage<::diplomacy::tensorflow::SequenceExample>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace diplomacy {
namespace tensorflow {

// ===================================================================

class Example : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.Example) */ {
 public:
  Example();
  virtual ~Example();

  Example(const Example& from);

  inline Example& operator=(const Example& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Example(Example&& from) noexcept
    : Example() {
    *this = ::std::move(from);
  }

  inline Example& operator=(Example&& from) noexcept {
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
  static const Example& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Example* internal_default_instance() {
    return reinterpret_cast<const Example*>(
               &_Example_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(Example* other);
  void Swap(Example* other);
  friend void swap(Example& a, Example& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Example* New() const final {
    return CreateMaybeMessage<Example>(NULL);
  }

  Example* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<Example>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const Example& from);
  void MergeFrom(const Example& from);
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
  void InternalSwap(Example* other);
  protected:
  explicit Example(::google::protobuf::Arena* arena);
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

  // .diplomacy.tensorflow.Features features = 1;
  bool has_features() const;
  void clear_features();
  static const int kFeaturesFieldNumber = 1;
  private:
  const ::diplomacy::tensorflow::Features& _internal_features() const;
  public:
  const ::diplomacy::tensorflow::Features& features() const;
  ::diplomacy::tensorflow::Features* release_features();
  ::diplomacy::tensorflow::Features* mutable_features();
  void set_allocated_features(::diplomacy::tensorflow::Features* features);
  void unsafe_arena_set_allocated_features(
      ::diplomacy::tensorflow::Features* features);
  ::diplomacy::tensorflow::Features* unsafe_arena_release_features();

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.Example)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::diplomacy::tensorflow::Features* features_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class SequenceExample : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.SequenceExample) */ {
 public:
  SequenceExample();
  virtual ~SequenceExample();

  SequenceExample(const SequenceExample& from);

  inline SequenceExample& operator=(const SequenceExample& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SequenceExample(SequenceExample&& from) noexcept
    : SequenceExample() {
    *this = ::std::move(from);
  }

  inline SequenceExample& operator=(SequenceExample&& from) noexcept {
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
  static const SequenceExample& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SequenceExample* internal_default_instance() {
    return reinterpret_cast<const SequenceExample*>(
               &_SequenceExample_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void UnsafeArenaSwap(SequenceExample* other);
  void Swap(SequenceExample* other);
  friend void swap(SequenceExample& a, SequenceExample& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SequenceExample* New() const final {
    return CreateMaybeMessage<SequenceExample>(NULL);
  }

  SequenceExample* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<SequenceExample>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const SequenceExample& from);
  void MergeFrom(const SequenceExample& from);
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
  void InternalSwap(SequenceExample* other);
  protected:
  explicit SequenceExample(::google::protobuf::Arena* arena);
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

  // .diplomacy.tensorflow.Features context = 1;
  bool has_context() const;
  void clear_context();
  static const int kContextFieldNumber = 1;
  private:
  const ::diplomacy::tensorflow::Features& _internal_context() const;
  public:
  const ::diplomacy::tensorflow::Features& context() const;
  ::diplomacy::tensorflow::Features* release_context();
  ::diplomacy::tensorflow::Features* mutable_context();
  void set_allocated_context(::diplomacy::tensorflow::Features* context);
  void unsafe_arena_set_allocated_context(
      ::diplomacy::tensorflow::Features* context);
  ::diplomacy::tensorflow::Features* unsafe_arena_release_context();

  // .diplomacy.tensorflow.FeatureLists feature_lists = 2;
  bool has_feature_lists() const;
  void clear_feature_lists();
  static const int kFeatureListsFieldNumber = 2;
  private:
  const ::diplomacy::tensorflow::FeatureLists& _internal_feature_lists() const;
  public:
  const ::diplomacy::tensorflow::FeatureLists& feature_lists() const;
  ::diplomacy::tensorflow::FeatureLists* release_feature_lists();
  ::diplomacy::tensorflow::FeatureLists* mutable_feature_lists();
  void set_allocated_feature_lists(::diplomacy::tensorflow::FeatureLists* feature_lists);
  void unsafe_arena_set_allocated_feature_lists(
      ::diplomacy::tensorflow::FeatureLists* feature_lists);
  ::diplomacy::tensorflow::FeatureLists* unsafe_arena_release_feature_lists();

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.SequenceExample)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::diplomacy::tensorflow::Features* context_;
  ::diplomacy::tensorflow::FeatureLists* feature_lists_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Example

// .diplomacy.tensorflow.Features features = 1;
inline bool Example::has_features() const {
  return this != internal_default_instance() && features_ != NULL;
}
inline const ::diplomacy::tensorflow::Features& Example::_internal_features() const {
  return *features_;
}
inline const ::diplomacy::tensorflow::Features& Example::features() const {
  const ::diplomacy::tensorflow::Features* p = features_;
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.Example.features)
  return p != NULL ? *p : *reinterpret_cast<const ::diplomacy::tensorflow::Features*>(
      &::diplomacy::tensorflow::_Features_default_instance_);
}
inline ::diplomacy::tensorflow::Features* Example::release_features() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.Example.features)
  
  ::diplomacy::tensorflow::Features* temp = features_;
  if (GetArenaNoVirtual() != NULL) {
    temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
  }
  features_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::Features* Example::unsafe_arena_release_features() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.Example.features)
  
  ::diplomacy::tensorflow::Features* temp = features_;
  features_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::Features* Example::mutable_features() {
  
  if (features_ == NULL) {
    auto* p = CreateMaybeMessage<::diplomacy::tensorflow::Features>(GetArenaNoVirtual());
    features_ = p;
  }
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.Example.features)
  return features_;
}
inline void Example::set_allocated_features(::diplomacy::tensorflow::Features* features) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(features_);
  }
  if (features) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(features)->GetArena();
    if (message_arena != submessage_arena) {
      features = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, features, submessage_arena);
    }
    
  } else {
    
  }
  features_ = features;
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.Example.features)
}

// -------------------------------------------------------------------

// SequenceExample

// .diplomacy.tensorflow.Features context = 1;
inline bool SequenceExample::has_context() const {
  return this != internal_default_instance() && context_ != NULL;
}
inline const ::diplomacy::tensorflow::Features& SequenceExample::_internal_context() const {
  return *context_;
}
inline const ::diplomacy::tensorflow::Features& SequenceExample::context() const {
  const ::diplomacy::tensorflow::Features* p = context_;
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.SequenceExample.context)
  return p != NULL ? *p : *reinterpret_cast<const ::diplomacy::tensorflow::Features*>(
      &::diplomacy::tensorflow::_Features_default_instance_);
}
inline ::diplomacy::tensorflow::Features* SequenceExample::release_context() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.SequenceExample.context)
  
  ::diplomacy::tensorflow::Features* temp = context_;
  if (GetArenaNoVirtual() != NULL) {
    temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
  }
  context_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::Features* SequenceExample::unsafe_arena_release_context() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.SequenceExample.context)
  
  ::diplomacy::tensorflow::Features* temp = context_;
  context_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::Features* SequenceExample::mutable_context() {
  
  if (context_ == NULL) {
    auto* p = CreateMaybeMessage<::diplomacy::tensorflow::Features>(GetArenaNoVirtual());
    context_ = p;
  }
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.SequenceExample.context)
  return context_;
}
inline void SequenceExample::set_allocated_context(::diplomacy::tensorflow::Features* context) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(context_);
  }
  if (context) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(context)->GetArena();
    if (message_arena != submessage_arena) {
      context = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, context, submessage_arena);
    }
    
  } else {
    
  }
  context_ = context;
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.SequenceExample.context)
}

// .diplomacy.tensorflow.FeatureLists feature_lists = 2;
inline bool SequenceExample::has_feature_lists() const {
  return this != internal_default_instance() && feature_lists_ != NULL;
}
inline const ::diplomacy::tensorflow::FeatureLists& SequenceExample::_internal_feature_lists() const {
  return *feature_lists_;
}
inline const ::diplomacy::tensorflow::FeatureLists& SequenceExample::feature_lists() const {
  const ::diplomacy::tensorflow::FeatureLists* p = feature_lists_;
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.SequenceExample.feature_lists)
  return p != NULL ? *p : *reinterpret_cast<const ::diplomacy::tensorflow::FeatureLists*>(
      &::diplomacy::tensorflow::_FeatureLists_default_instance_);
}
inline ::diplomacy::tensorflow::FeatureLists* SequenceExample::release_feature_lists() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.SequenceExample.feature_lists)
  
  ::diplomacy::tensorflow::FeatureLists* temp = feature_lists_;
  if (GetArenaNoVirtual() != NULL) {
    temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
  }
  feature_lists_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::FeatureLists* SequenceExample::unsafe_arena_release_feature_lists() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.SequenceExample.feature_lists)
  
  ::diplomacy::tensorflow::FeatureLists* temp = feature_lists_;
  feature_lists_ = NULL;
  return temp;
}
inline ::diplomacy::tensorflow::FeatureLists* SequenceExample::mutable_feature_lists() {
  
  if (feature_lists_ == NULL) {
    auto* p = CreateMaybeMessage<::diplomacy::tensorflow::FeatureLists>(GetArenaNoVirtual());
    feature_lists_ = p;
  }
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.SequenceExample.feature_lists)
  return feature_lists_;
}
inline void SequenceExample::set_allocated_feature_lists(::diplomacy::tensorflow::FeatureLists* feature_lists) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(feature_lists_);
  }
  if (feature_lists) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(feature_lists)->GetArena();
    if (message_arena != submessage_arena) {
      feature_lists = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, feature_lists, submessage_arena);
    }
    
  } else {
    
  }
  feature_lists_ = feature_lists;
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.SequenceExample.feature_lists)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow
}  // namespace diplomacy

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2fexample_2fexample_2eproto
