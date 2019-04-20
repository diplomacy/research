// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/python/training/checkpoint_state.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto {
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto
namespace diplomacy {
namespace tensorflow {
class CheckpointState;
class CheckpointStateDefaultTypeInternal;
extern CheckpointStateDefaultTypeInternal _CheckpointState_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> ::diplomacy::tensorflow::CheckpointState* Arena::CreateMaybeMessage<::diplomacy::tensorflow::CheckpointState>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace diplomacy {
namespace tensorflow {

// ===================================================================

class CheckpointState : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.CheckpointState) */ {
 public:
  CheckpointState();
  virtual ~CheckpointState();

  CheckpointState(const CheckpointState& from);

  inline CheckpointState& operator=(const CheckpointState& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CheckpointState(CheckpointState&& from) noexcept
    : CheckpointState() {
    *this = ::std::move(from);
  }

  inline CheckpointState& operator=(CheckpointState&& from) noexcept {
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
  static const CheckpointState& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CheckpointState* internal_default_instance() {
    return reinterpret_cast<const CheckpointState*>(
               &_CheckpointState_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(CheckpointState* other);
  void Swap(CheckpointState* other);
  friend void swap(CheckpointState& a, CheckpointState& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CheckpointState* New() const final {
    return CreateMaybeMessage<CheckpointState>(NULL);
  }

  CheckpointState* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CheckpointState>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CheckpointState& from);
  void MergeFrom(const CheckpointState& from);
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
  void InternalSwap(CheckpointState* other);
  protected:
  explicit CheckpointState(::google::protobuf::Arena* arena);
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

  // repeated string all_model_checkpoint_paths = 2;
  int all_model_checkpoint_paths_size() const;
  void clear_all_model_checkpoint_paths();
  static const int kAllModelCheckpointPathsFieldNumber = 2;
  const ::std::string& all_model_checkpoint_paths(int index) const;
  ::std::string* mutable_all_model_checkpoint_paths(int index);
  void set_all_model_checkpoint_paths(int index, const ::std::string& value);
  #if LANG_CXX11
  void set_all_model_checkpoint_paths(int index, ::std::string&& value);
  #endif
  void set_all_model_checkpoint_paths(int index, const char* value);
  void set_all_model_checkpoint_paths(int index, const char* value, size_t size);
  ::std::string* add_all_model_checkpoint_paths();
  void add_all_model_checkpoint_paths(const ::std::string& value);
  #if LANG_CXX11
  void add_all_model_checkpoint_paths(::std::string&& value);
  #endif
  void add_all_model_checkpoint_paths(const char* value);
  void add_all_model_checkpoint_paths(const char* value, size_t size);
  const ::google::protobuf::RepeatedPtrField< ::std::string>& all_model_checkpoint_paths() const;
  ::google::protobuf::RepeatedPtrField< ::std::string>* mutable_all_model_checkpoint_paths();

  // repeated double all_model_checkpoint_timestamps = 3;
  int all_model_checkpoint_timestamps_size() const;
  void clear_all_model_checkpoint_timestamps();
  static const int kAllModelCheckpointTimestampsFieldNumber = 3;
  double all_model_checkpoint_timestamps(int index) const;
  void set_all_model_checkpoint_timestamps(int index, double value);
  void add_all_model_checkpoint_timestamps(double value);
  const ::google::protobuf::RepeatedField< double >&
      all_model_checkpoint_timestamps() const;
  ::google::protobuf::RepeatedField< double >*
      mutable_all_model_checkpoint_timestamps();

  // string model_checkpoint_path = 1;
  void clear_model_checkpoint_path();
  static const int kModelCheckpointPathFieldNumber = 1;
  const ::std::string& model_checkpoint_path() const;
  void set_model_checkpoint_path(const ::std::string& value);
  #if LANG_CXX11
  void set_model_checkpoint_path(::std::string&& value);
  #endif
  void set_model_checkpoint_path(const char* value);
  void set_model_checkpoint_path(const char* value, size_t size);
  ::std::string* mutable_model_checkpoint_path();
  ::std::string* release_model_checkpoint_path();
  void set_allocated_model_checkpoint_path(::std::string* model_checkpoint_path);
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  ::std::string* unsafe_arena_release_model_checkpoint_path();
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_model_checkpoint_path(
      ::std::string* model_checkpoint_path);

  // double last_preserved_timestamp = 4;
  void clear_last_preserved_timestamp();
  static const int kLastPreservedTimestampFieldNumber = 4;
  double last_preserved_timestamp() const;
  void set_last_preserved_timestamp(double value);

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.CheckpointState)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::std::string> all_model_checkpoint_paths_;
  ::google::protobuf::RepeatedField< double > all_model_checkpoint_timestamps_;
  mutable int _all_model_checkpoint_timestamps_cached_byte_size_;
  ::google::protobuf::internal::ArenaStringPtr model_checkpoint_path_;
  double last_preserved_timestamp_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CheckpointState

// string model_checkpoint_path = 1;
inline void CheckpointState::clear_model_checkpoint_path() {
  model_checkpoint_path_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& CheckpointState::model_checkpoint_path() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
  return model_checkpoint_path_.Get();
}
inline void CheckpointState::set_model_checkpoint_path(const ::std::string& value) {
  
  model_checkpoint_path_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}
#if LANG_CXX11
inline void CheckpointState::set_model_checkpoint_path(::std::string&& value) {
  
  model_checkpoint_path_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}
#endif
inline void CheckpointState::set_model_checkpoint_path(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  model_checkpoint_path_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}
inline void CheckpointState::set_model_checkpoint_path(const char* value,
    size_t size) {
  
  model_checkpoint_path_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}
inline ::std::string* CheckpointState::mutable_model_checkpoint_path() {
  
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
  return model_checkpoint_path_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* CheckpointState::release_model_checkpoint_path() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
  
  return model_checkpoint_path_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void CheckpointState::set_allocated_model_checkpoint_path(::std::string* model_checkpoint_path) {
  if (model_checkpoint_path != NULL) {
    
  } else {
    
  }
  model_checkpoint_path_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), model_checkpoint_path,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}
inline ::std::string* CheckpointState::unsafe_arena_release_model_checkpoint_path() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return model_checkpoint_path_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void CheckpointState::unsafe_arena_set_allocated_model_checkpoint_path(
    ::std::string* model_checkpoint_path) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (model_checkpoint_path != NULL) {
    
  } else {
    
  }
  model_checkpoint_path_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      model_checkpoint_path, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:diplomacy.tensorflow.CheckpointState.model_checkpoint_path)
}

// repeated string all_model_checkpoint_paths = 2;
inline int CheckpointState::all_model_checkpoint_paths_size() const {
  return all_model_checkpoint_paths_.size();
}
inline void CheckpointState::clear_all_model_checkpoint_paths() {
  all_model_checkpoint_paths_.Clear();
}
inline const ::std::string& CheckpointState::all_model_checkpoint_paths(int index) const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  return all_model_checkpoint_paths_.Get(index);
}
inline ::std::string* CheckpointState::mutable_all_model_checkpoint_paths(int index) {
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  return all_model_checkpoint_paths_.Mutable(index);
}
inline void CheckpointState::set_all_model_checkpoint_paths(int index, const ::std::string& value) {
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  all_model_checkpoint_paths_.Mutable(index)->assign(value);
}
#if LANG_CXX11
inline void CheckpointState::set_all_model_checkpoint_paths(int index, ::std::string&& value) {
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  all_model_checkpoint_paths_.Mutable(index)->assign(std::move(value));
}
#endif
inline void CheckpointState::set_all_model_checkpoint_paths(int index, const char* value) {
  GOOGLE_DCHECK(value != NULL);
  all_model_checkpoint_paths_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
inline void CheckpointState::set_all_model_checkpoint_paths(int index, const char* value, size_t size) {
  all_model_checkpoint_paths_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
inline ::std::string* CheckpointState::add_all_model_checkpoint_paths() {
  // @@protoc_insertion_point(field_add_mutable:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  return all_model_checkpoint_paths_.Add();
}
inline void CheckpointState::add_all_model_checkpoint_paths(const ::std::string& value) {
  all_model_checkpoint_paths_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
#if LANG_CXX11
inline void CheckpointState::add_all_model_checkpoint_paths(::std::string&& value) {
  all_model_checkpoint_paths_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
#endif
inline void CheckpointState::add_all_model_checkpoint_paths(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  all_model_checkpoint_paths_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
inline void CheckpointState::add_all_model_checkpoint_paths(const char* value, size_t size) {
  all_model_checkpoint_paths_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
}
inline const ::google::protobuf::RepeatedPtrField< ::std::string>&
CheckpointState::all_model_checkpoint_paths() const {
  // @@protoc_insertion_point(field_list:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  return all_model_checkpoint_paths_;
}
inline ::google::protobuf::RepeatedPtrField< ::std::string>*
CheckpointState::mutable_all_model_checkpoint_paths() {
  // @@protoc_insertion_point(field_mutable_list:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_paths)
  return &all_model_checkpoint_paths_;
}

// repeated double all_model_checkpoint_timestamps = 3;
inline int CheckpointState::all_model_checkpoint_timestamps_size() const {
  return all_model_checkpoint_timestamps_.size();
}
inline void CheckpointState::clear_all_model_checkpoint_timestamps() {
  all_model_checkpoint_timestamps_.Clear();
}
inline double CheckpointState::all_model_checkpoint_timestamps(int index) const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps)
  return all_model_checkpoint_timestamps_.Get(index);
}
inline void CheckpointState::set_all_model_checkpoint_timestamps(int index, double value) {
  all_model_checkpoint_timestamps_.Set(index, value);
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps)
}
inline void CheckpointState::add_all_model_checkpoint_timestamps(double value) {
  all_model_checkpoint_timestamps_.Add(value);
  // @@protoc_insertion_point(field_add:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps)
}
inline const ::google::protobuf::RepeatedField< double >&
CheckpointState::all_model_checkpoint_timestamps() const {
  // @@protoc_insertion_point(field_list:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps)
  return all_model_checkpoint_timestamps_;
}
inline ::google::protobuf::RepeatedField< double >*
CheckpointState::mutable_all_model_checkpoint_timestamps() {
  // @@protoc_insertion_point(field_mutable_list:diplomacy.tensorflow.CheckpointState.all_model_checkpoint_timestamps)
  return &all_model_checkpoint_timestamps_;
}

// double last_preserved_timestamp = 4;
inline void CheckpointState::clear_last_preserved_timestamp() {
  last_preserved_timestamp_ = 0;
}
inline double CheckpointState::last_preserved_timestamp() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.CheckpointState.last_preserved_timestamp)
  return last_preserved_timestamp_;
}
inline void CheckpointState::set_last_preserved_timestamp(double value) {
  
  last_preserved_timestamp_ = value;
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.CheckpointState.last_preserved_timestamp)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow
}  // namespace diplomacy

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fpython_2ftraining_2fcheckpoint_5fstate_2eproto
