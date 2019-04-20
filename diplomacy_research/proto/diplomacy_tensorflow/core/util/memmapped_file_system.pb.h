// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/util/memmapped_file_system.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto {
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto
namespace diplomacy {
namespace tensorflow {
class MemmappedFileSystemDirectory;
class MemmappedFileSystemDirectoryDefaultTypeInternal;
extern MemmappedFileSystemDirectoryDefaultTypeInternal _MemmappedFileSystemDirectory_default_instance_;
class MemmappedFileSystemDirectoryElement;
class MemmappedFileSystemDirectoryElementDefaultTypeInternal;
extern MemmappedFileSystemDirectoryElementDefaultTypeInternal _MemmappedFileSystemDirectoryElement_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> ::diplomacy::tensorflow::MemmappedFileSystemDirectory* Arena::CreateMaybeMessage<::diplomacy::tensorflow::MemmappedFileSystemDirectory>(Arena*);
template<> ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement* Arena::CreateMaybeMessage<::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace diplomacy {
namespace tensorflow {

// ===================================================================

class MemmappedFileSystemDirectoryElement : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement) */ {
 public:
  MemmappedFileSystemDirectoryElement();
  virtual ~MemmappedFileSystemDirectoryElement();

  MemmappedFileSystemDirectoryElement(const MemmappedFileSystemDirectoryElement& from);

  inline MemmappedFileSystemDirectoryElement& operator=(const MemmappedFileSystemDirectoryElement& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  MemmappedFileSystemDirectoryElement(MemmappedFileSystemDirectoryElement&& from) noexcept
    : MemmappedFileSystemDirectoryElement() {
    *this = ::std::move(from);
  }

  inline MemmappedFileSystemDirectoryElement& operator=(MemmappedFileSystemDirectoryElement&& from) noexcept {
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
  static const MemmappedFileSystemDirectoryElement& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const MemmappedFileSystemDirectoryElement* internal_default_instance() {
    return reinterpret_cast<const MemmappedFileSystemDirectoryElement*>(
               &_MemmappedFileSystemDirectoryElement_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(MemmappedFileSystemDirectoryElement* other);
  void Swap(MemmappedFileSystemDirectoryElement* other);
  friend void swap(MemmappedFileSystemDirectoryElement& a, MemmappedFileSystemDirectoryElement& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline MemmappedFileSystemDirectoryElement* New() const final {
    return CreateMaybeMessage<MemmappedFileSystemDirectoryElement>(NULL);
  }

  MemmappedFileSystemDirectoryElement* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<MemmappedFileSystemDirectoryElement>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const MemmappedFileSystemDirectoryElement& from);
  void MergeFrom(const MemmappedFileSystemDirectoryElement& from);
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
  void InternalSwap(MemmappedFileSystemDirectoryElement* other);
  protected:
  explicit MemmappedFileSystemDirectoryElement(::google::protobuf::Arena* arena);
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

  // string name = 2;
  void clear_name();
  static const int kNameFieldNumber = 2;
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

  // uint64 offset = 1;
  void clear_offset();
  static const int kOffsetFieldNumber = 1;
  ::google::protobuf::uint64 offset() const;
  void set_offset(::google::protobuf::uint64 value);

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::uint64 offset_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class MemmappedFileSystemDirectory : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:diplomacy.tensorflow.MemmappedFileSystemDirectory) */ {
 public:
  MemmappedFileSystemDirectory();
  virtual ~MemmappedFileSystemDirectory();

  MemmappedFileSystemDirectory(const MemmappedFileSystemDirectory& from);

  inline MemmappedFileSystemDirectory& operator=(const MemmappedFileSystemDirectory& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  MemmappedFileSystemDirectory(MemmappedFileSystemDirectory&& from) noexcept
    : MemmappedFileSystemDirectory() {
    *this = ::std::move(from);
  }

  inline MemmappedFileSystemDirectory& operator=(MemmappedFileSystemDirectory&& from) noexcept {
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
  static const MemmappedFileSystemDirectory& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const MemmappedFileSystemDirectory* internal_default_instance() {
    return reinterpret_cast<const MemmappedFileSystemDirectory*>(
               &_MemmappedFileSystemDirectory_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void UnsafeArenaSwap(MemmappedFileSystemDirectory* other);
  void Swap(MemmappedFileSystemDirectory* other);
  friend void swap(MemmappedFileSystemDirectory& a, MemmappedFileSystemDirectory& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline MemmappedFileSystemDirectory* New() const final {
    return CreateMaybeMessage<MemmappedFileSystemDirectory>(NULL);
  }

  MemmappedFileSystemDirectory* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<MemmappedFileSystemDirectory>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const MemmappedFileSystemDirectory& from);
  void MergeFrom(const MemmappedFileSystemDirectory& from);
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
  void InternalSwap(MemmappedFileSystemDirectory* other);
  protected:
  explicit MemmappedFileSystemDirectory(::google::protobuf::Arena* arena);
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

  // repeated .diplomacy.tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  int element_size() const;
  void clear_element();
  static const int kElementFieldNumber = 1;
  ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement* mutable_element(int index);
  ::google::protobuf::RepeatedPtrField< ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement >*
      mutable_element();
  const ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement& element(int index) const;
  ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement* add_element();
  const ::google::protobuf::RepeatedPtrField< ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement >&
      element() const;

  // @@protoc_insertion_point(class_scope:diplomacy.tensorflow.MemmappedFileSystemDirectory)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement > element_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// MemmappedFileSystemDirectoryElement

// uint64 offset = 1;
inline void MemmappedFileSystemDirectoryElement::clear_offset() {
  offset_ = GOOGLE_ULONGLONG(0);
}
inline ::google::protobuf::uint64 MemmappedFileSystemDirectoryElement::offset() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.offset)
  return offset_;
}
inline void MemmappedFileSystemDirectoryElement::set_offset(::google::protobuf::uint64 value) {
  
  offset_ = value;
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.offset)
}

// string name = 2;
inline void MemmappedFileSystemDirectoryElement::clear_name() {
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& MemmappedFileSystemDirectoryElement::name() const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
  return name_.Get();
}
inline void MemmappedFileSystemDirectoryElement::set_name(const ::std::string& value) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}
#if LANG_CXX11
inline void MemmappedFileSystemDirectoryElement::set_name(::std::string&& value) {
  
  name_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}
#endif
inline void MemmappedFileSystemDirectoryElement::set_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}
inline void MemmappedFileSystemDirectoryElement::set_name(const char* value,
    size_t size) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}
inline ::std::string* MemmappedFileSystemDirectoryElement::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
  return name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* MemmappedFileSystemDirectoryElement::release_name() {
  // @@protoc_insertion_point(field_release:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
  
  return name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void MemmappedFileSystemDirectoryElement::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}
inline ::std::string* MemmappedFileSystemDirectoryElement::unsafe_arena_release_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void MemmappedFileSystemDirectoryElement::unsafe_arena_set_allocated_name(
    ::std::string* name) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (name != NULL) {
    
  } else {
    
  }
  name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      name, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:diplomacy.tensorflow.MemmappedFileSystemDirectoryElement.name)
}

// -------------------------------------------------------------------

// MemmappedFileSystemDirectory

// repeated .diplomacy.tensorflow.MemmappedFileSystemDirectoryElement element = 1;
inline int MemmappedFileSystemDirectory::element_size() const {
  return element_.size();
}
inline void MemmappedFileSystemDirectory::clear_element() {
  element_.Clear();
}
inline ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement* MemmappedFileSystemDirectory::mutable_element(int index) {
  // @@protoc_insertion_point(field_mutable:diplomacy.tensorflow.MemmappedFileSystemDirectory.element)
  return element_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement >*
MemmappedFileSystemDirectory::mutable_element() {
  // @@protoc_insertion_point(field_mutable_list:diplomacy.tensorflow.MemmappedFileSystemDirectory.element)
  return &element_;
}
inline const ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement& MemmappedFileSystemDirectory::element(int index) const {
  // @@protoc_insertion_point(field_get:diplomacy.tensorflow.MemmappedFileSystemDirectory.element)
  return element_.Get(index);
}
inline ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement* MemmappedFileSystemDirectory::add_element() {
  // @@protoc_insertion_point(field_add:diplomacy.tensorflow.MemmappedFileSystemDirectory.element)
  return element_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::diplomacy::tensorflow::MemmappedFileSystemDirectoryElement >&
MemmappedFileSystemDirectory::element() const {
  // @@protoc_insertion_point(field_list:diplomacy.tensorflow.MemmappedFileSystemDirectory.element)
  return element_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow
}  // namespace diplomacy

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto
