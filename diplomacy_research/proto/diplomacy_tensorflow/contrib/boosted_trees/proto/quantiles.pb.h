// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/contrib/boosted_trees/proto/quantiles.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[4];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto
namespace boosted_trees {
class QuantileConfig;
class QuantileConfigDefaultTypeInternal;
extern QuantileConfigDefaultTypeInternal _QuantileConfig_default_instance_;
class QuantileEntry;
class QuantileEntryDefaultTypeInternal;
extern QuantileEntryDefaultTypeInternal _QuantileEntry_default_instance_;
class QuantileStreamState;
class QuantileStreamStateDefaultTypeInternal;
extern QuantileStreamStateDefaultTypeInternal _QuantileStreamState_default_instance_;
class QuantileSummaryState;
class QuantileSummaryStateDefaultTypeInternal;
extern QuantileSummaryStateDefaultTypeInternal _QuantileSummaryState_default_instance_;
}  // namespace boosted_trees
namespace google {
namespace protobuf {
template<> ::boosted_trees::QuantileConfig* Arena::CreateMaybeMessage<::boosted_trees::QuantileConfig>(Arena*);
template<> ::boosted_trees::QuantileEntry* Arena::CreateMaybeMessage<::boosted_trees::QuantileEntry>(Arena*);
template<> ::boosted_trees::QuantileStreamState* Arena::CreateMaybeMessage<::boosted_trees::QuantileStreamState>(Arena*);
template<> ::boosted_trees::QuantileSummaryState* Arena::CreateMaybeMessage<::boosted_trees::QuantileSummaryState>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace boosted_trees {

// ===================================================================

class QuantileConfig : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:boosted_trees.QuantileConfig) */ {
 public:
  QuantileConfig();
  virtual ~QuantileConfig();

  QuantileConfig(const QuantileConfig& from);

  inline QuantileConfig& operator=(const QuantileConfig& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  QuantileConfig(QuantileConfig&& from) noexcept
    : QuantileConfig() {
    *this = ::std::move(from);
  }

  inline QuantileConfig& operator=(QuantileConfig&& from) noexcept {
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
  static const QuantileConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const QuantileConfig* internal_default_instance() {
    return reinterpret_cast<const QuantileConfig*>(
               &_QuantileConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(QuantileConfig* other);
  void Swap(QuantileConfig* other);
  friend void swap(QuantileConfig& a, QuantileConfig& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline QuantileConfig* New() const final {
    return CreateMaybeMessage<QuantileConfig>(NULL);
  }

  QuantileConfig* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<QuantileConfig>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const QuantileConfig& from);
  void MergeFrom(const QuantileConfig& from);
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
  void InternalSwap(QuantileConfig* other);
  protected:
  explicit QuantileConfig(::google::protobuf::Arena* arena);
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

  // double eps = 1;
  void clear_eps();
  static const int kEpsFieldNumber = 1;
  double eps() const;
  void set_eps(double value);

  // int64 num_quantiles = 2;
  void clear_num_quantiles();
  static const int kNumQuantilesFieldNumber = 2;
  ::google::protobuf::int64 num_quantiles() const;
  void set_num_quantiles(::google::protobuf::int64 value);

  // @@protoc_insertion_point(class_scope:boosted_trees.QuantileConfig)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  double eps_;
  ::google::protobuf::int64 num_quantiles_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class QuantileEntry : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:boosted_trees.QuantileEntry) */ {
 public:
  QuantileEntry();
  virtual ~QuantileEntry();

  QuantileEntry(const QuantileEntry& from);

  inline QuantileEntry& operator=(const QuantileEntry& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  QuantileEntry(QuantileEntry&& from) noexcept
    : QuantileEntry() {
    *this = ::std::move(from);
  }

  inline QuantileEntry& operator=(QuantileEntry&& from) noexcept {
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
  static const QuantileEntry& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const QuantileEntry* internal_default_instance() {
    return reinterpret_cast<const QuantileEntry*>(
               &_QuantileEntry_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void UnsafeArenaSwap(QuantileEntry* other);
  void Swap(QuantileEntry* other);
  friend void swap(QuantileEntry& a, QuantileEntry& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline QuantileEntry* New() const final {
    return CreateMaybeMessage<QuantileEntry>(NULL);
  }

  QuantileEntry* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<QuantileEntry>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const QuantileEntry& from);
  void MergeFrom(const QuantileEntry& from);
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
  void InternalSwap(QuantileEntry* other);
  protected:
  explicit QuantileEntry(::google::protobuf::Arena* arena);
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

  // float value = 1;
  void clear_value();
  static const int kValueFieldNumber = 1;
  float value() const;
  void set_value(float value);

  // float weight = 2;
  void clear_weight();
  static const int kWeightFieldNumber = 2;
  float weight() const;
  void set_weight(float value);

  // float min_rank = 3;
  void clear_min_rank();
  static const int kMinRankFieldNumber = 3;
  float min_rank() const;
  void set_min_rank(float value);

  // float max_rank = 4;
  void clear_max_rank();
  static const int kMaxRankFieldNumber = 4;
  float max_rank() const;
  void set_max_rank(float value);

  // @@protoc_insertion_point(class_scope:boosted_trees.QuantileEntry)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  float value_;
  float weight_;
  float min_rank_;
  float max_rank_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class QuantileSummaryState : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:boosted_trees.QuantileSummaryState) */ {
 public:
  QuantileSummaryState();
  virtual ~QuantileSummaryState();

  QuantileSummaryState(const QuantileSummaryState& from);

  inline QuantileSummaryState& operator=(const QuantileSummaryState& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  QuantileSummaryState(QuantileSummaryState&& from) noexcept
    : QuantileSummaryState() {
    *this = ::std::move(from);
  }

  inline QuantileSummaryState& operator=(QuantileSummaryState&& from) noexcept {
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
  static const QuantileSummaryState& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const QuantileSummaryState* internal_default_instance() {
    return reinterpret_cast<const QuantileSummaryState*>(
               &_QuantileSummaryState_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void UnsafeArenaSwap(QuantileSummaryState* other);
  void Swap(QuantileSummaryState* other);
  friend void swap(QuantileSummaryState& a, QuantileSummaryState& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline QuantileSummaryState* New() const final {
    return CreateMaybeMessage<QuantileSummaryState>(NULL);
  }

  QuantileSummaryState* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<QuantileSummaryState>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const QuantileSummaryState& from);
  void MergeFrom(const QuantileSummaryState& from);
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
  void InternalSwap(QuantileSummaryState* other);
  protected:
  explicit QuantileSummaryState(::google::protobuf::Arena* arena);
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

  // repeated .boosted_trees.QuantileEntry entries = 1;
  int entries_size() const;
  void clear_entries();
  static const int kEntriesFieldNumber = 1;
  ::boosted_trees::QuantileEntry* mutable_entries(int index);
  ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileEntry >*
      mutable_entries();
  const ::boosted_trees::QuantileEntry& entries(int index) const;
  ::boosted_trees::QuantileEntry* add_entries();
  const ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileEntry >&
      entries() const;

  // @@protoc_insertion_point(class_scope:boosted_trees.QuantileSummaryState)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileEntry > entries_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class QuantileStreamState : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:boosted_trees.QuantileStreamState) */ {
 public:
  QuantileStreamState();
  virtual ~QuantileStreamState();

  QuantileStreamState(const QuantileStreamState& from);

  inline QuantileStreamState& operator=(const QuantileStreamState& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  QuantileStreamState(QuantileStreamState&& from) noexcept
    : QuantileStreamState() {
    *this = ::std::move(from);
  }

  inline QuantileStreamState& operator=(QuantileStreamState&& from) noexcept {
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
  static const QuantileStreamState& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const QuantileStreamState* internal_default_instance() {
    return reinterpret_cast<const QuantileStreamState*>(
               &_QuantileStreamState_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  void UnsafeArenaSwap(QuantileStreamState* other);
  void Swap(QuantileStreamState* other);
  friend void swap(QuantileStreamState& a, QuantileStreamState& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline QuantileStreamState* New() const final {
    return CreateMaybeMessage<QuantileStreamState>(NULL);
  }

  QuantileStreamState* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<QuantileStreamState>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const QuantileStreamState& from);
  void MergeFrom(const QuantileStreamState& from);
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
  void InternalSwap(QuantileStreamState* other);
  protected:
  explicit QuantileStreamState(::google::protobuf::Arena* arena);
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

  // repeated .boosted_trees.QuantileSummaryState summaries = 1;
  int summaries_size() const;
  void clear_summaries();
  static const int kSummariesFieldNumber = 1;
  ::boosted_trees::QuantileSummaryState* mutable_summaries(int index);
  ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileSummaryState >*
      mutable_summaries();
  const ::boosted_trees::QuantileSummaryState& summaries(int index) const;
  ::boosted_trees::QuantileSummaryState* add_summaries();
  const ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileSummaryState >&
      summaries() const;

  // @@protoc_insertion_point(class_scope:boosted_trees.QuantileStreamState)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileSummaryState > summaries_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// QuantileConfig

// double eps = 1;
inline void QuantileConfig::clear_eps() {
  eps_ = 0;
}
inline double QuantileConfig::eps() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileConfig.eps)
  return eps_;
}
inline void QuantileConfig::set_eps(double value) {
  
  eps_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileConfig.eps)
}

// int64 num_quantiles = 2;
inline void QuantileConfig::clear_num_quantiles() {
  num_quantiles_ = GOOGLE_LONGLONG(0);
}
inline ::google::protobuf::int64 QuantileConfig::num_quantiles() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileConfig.num_quantiles)
  return num_quantiles_;
}
inline void QuantileConfig::set_num_quantiles(::google::protobuf::int64 value) {
  
  num_quantiles_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileConfig.num_quantiles)
}

// -------------------------------------------------------------------

// QuantileEntry

// float value = 1;
inline void QuantileEntry::clear_value() {
  value_ = 0;
}
inline float QuantileEntry::value() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileEntry.value)
  return value_;
}
inline void QuantileEntry::set_value(float value) {
  
  value_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileEntry.value)
}

// float weight = 2;
inline void QuantileEntry::clear_weight() {
  weight_ = 0;
}
inline float QuantileEntry::weight() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileEntry.weight)
  return weight_;
}
inline void QuantileEntry::set_weight(float value) {
  
  weight_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileEntry.weight)
}

// float min_rank = 3;
inline void QuantileEntry::clear_min_rank() {
  min_rank_ = 0;
}
inline float QuantileEntry::min_rank() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileEntry.min_rank)
  return min_rank_;
}
inline void QuantileEntry::set_min_rank(float value) {
  
  min_rank_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileEntry.min_rank)
}

// float max_rank = 4;
inline void QuantileEntry::clear_max_rank() {
  max_rank_ = 0;
}
inline float QuantileEntry::max_rank() const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileEntry.max_rank)
  return max_rank_;
}
inline void QuantileEntry::set_max_rank(float value) {
  
  max_rank_ = value;
  // @@protoc_insertion_point(field_set:boosted_trees.QuantileEntry.max_rank)
}

// -------------------------------------------------------------------

// QuantileSummaryState

// repeated .boosted_trees.QuantileEntry entries = 1;
inline int QuantileSummaryState::entries_size() const {
  return entries_.size();
}
inline void QuantileSummaryState::clear_entries() {
  entries_.Clear();
}
inline ::boosted_trees::QuantileEntry* QuantileSummaryState::mutable_entries(int index) {
  // @@protoc_insertion_point(field_mutable:boosted_trees.QuantileSummaryState.entries)
  return entries_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileEntry >*
QuantileSummaryState::mutable_entries() {
  // @@protoc_insertion_point(field_mutable_list:boosted_trees.QuantileSummaryState.entries)
  return &entries_;
}
inline const ::boosted_trees::QuantileEntry& QuantileSummaryState::entries(int index) const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileSummaryState.entries)
  return entries_.Get(index);
}
inline ::boosted_trees::QuantileEntry* QuantileSummaryState::add_entries() {
  // @@protoc_insertion_point(field_add:boosted_trees.QuantileSummaryState.entries)
  return entries_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileEntry >&
QuantileSummaryState::entries() const {
  // @@protoc_insertion_point(field_list:boosted_trees.QuantileSummaryState.entries)
  return entries_;
}

// -------------------------------------------------------------------

// QuantileStreamState

// repeated .boosted_trees.QuantileSummaryState summaries = 1;
inline int QuantileStreamState::summaries_size() const {
  return summaries_.size();
}
inline void QuantileStreamState::clear_summaries() {
  summaries_.Clear();
}
inline ::boosted_trees::QuantileSummaryState* QuantileStreamState::mutable_summaries(int index) {
  // @@protoc_insertion_point(field_mutable:boosted_trees.QuantileStreamState.summaries)
  return summaries_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileSummaryState >*
QuantileStreamState::mutable_summaries() {
  // @@protoc_insertion_point(field_mutable_list:boosted_trees.QuantileStreamState.summaries)
  return &summaries_;
}
inline const ::boosted_trees::QuantileSummaryState& QuantileStreamState::summaries(int index) const {
  // @@protoc_insertion_point(field_get:boosted_trees.QuantileStreamState.summaries)
  return summaries_.Get(index);
}
inline ::boosted_trees::QuantileSummaryState* QuantileStreamState::add_summaries() {
  // @@protoc_insertion_point(field_add:boosted_trees.QuantileStreamState.summaries)
  return summaries_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::boosted_trees::QuantileSummaryState >&
QuantileStreamState::summaries() const {
  // @@protoc_insertion_point(field_list:boosted_trees.QuantileStreamState.summaries)
  return summaries_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace boosted_trees

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fcontrib_2fboosted_5ftrees_2fproto_2fquantiles_2eproto
