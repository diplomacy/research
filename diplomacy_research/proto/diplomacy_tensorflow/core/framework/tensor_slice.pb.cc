// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/framework/tensor_slice.proto

#include "diplomacy_tensorflow/core/framework/tensor_slice.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto ::google::protobuf::internal::SCCInfo<0> scc_info_TensorSliceProto_Extent;
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto
namespace diplomacy {
namespace tensorflow {
class TensorSliceProto_ExtentDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<TensorSliceProto_Extent>
      _instance;
  ::google::protobuf::int64 length_;
} _TensorSliceProto_Extent_default_instance_;
class TensorSliceProtoDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<TensorSliceProto>
      _instance;
} _TensorSliceProto_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto {
static void InitDefaultsTensorSliceProto_Extent() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_TensorSliceProto_Extent_default_instance_;
    new (ptr) ::diplomacy::tensorflow::TensorSliceProto_Extent();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::TensorSliceProto_Extent::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_TensorSliceProto_Extent =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsTensorSliceProto_Extent}, {}};

static void InitDefaultsTensorSliceProto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_TensorSliceProto_default_instance_;
    new (ptr) ::diplomacy::tensorflow::TensorSliceProto();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::TensorSliceProto::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_TensorSliceProto =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsTensorSliceProto}, {
      &protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto_Extent.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_TensorSliceProto_Extent.base);
  ::google::protobuf::internal::InitSCC(&scc_info_TensorSliceProto.base);
}

::google::protobuf::Metadata file_level_metadata[2];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto_Extent, _internal_metadata_),
  ~0u,  // no _extensions_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto_Extent, _oneof_case_[0]),
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto_Extent, start_),
  offsetof(::diplomacy::tensorflow::TensorSliceProto_ExtentDefaultTypeInternal, length_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto_Extent, has_length_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::TensorSliceProto, extent_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::diplomacy::tensorflow::TensorSliceProto_Extent)},
  { 8, -1, sizeof(::diplomacy::tensorflow::TensorSliceProto)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_TensorSliceProto_Extent_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_TensorSliceProto_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/core/framework/tensor_slice.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n6diplomacy_tensorflow/core/framework/te"
      "nsor_slice.proto\022\024diplomacy.tensorflow\"\212"
      "\001\n\020TensorSliceProto\022=\n\006extent\030\001 \003(\0132-.di"
      "plomacy.tensorflow.TensorSliceProto.Exte"
      "nt\0327\n\006Extent\022\r\n\005start\030\001 \001(\003\022\020\n\006length\030\002 "
      "\001(\003H\000B\014\n\nhas_lengthBq\n\030org.tensorflow.fr"
      "ameworkB\021TensorSliceProtosP\001Z=github.com"
      "/tensorflow/tensorflow/tensorflow/go/cor"
      "e/framework\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 342);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/core/framework/tensor_slice.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto
namespace diplomacy {
namespace tensorflow {

// ===================================================================

void TensorSliceProto_Extent::InitAsDefaultInstance() {
  ::diplomacy::tensorflow::_TensorSliceProto_Extent_default_instance_.length_ = GOOGLE_LONGLONG(0);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorSliceProto_Extent::kStartFieldNumber;
const int TensorSliceProto_Extent::kLengthFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

TensorSliceProto_Extent::TensorSliceProto_Extent()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto_Extent.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.TensorSliceProto.Extent)
}
TensorSliceProto_Extent::TensorSliceProto_Extent(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto_Extent.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.TensorSliceProto.Extent)
}
TensorSliceProto_Extent::TensorSliceProto_Extent(const TensorSliceProto_Extent& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  start_ = from.start_;
  clear_has_has_length();
  switch (from.has_length_case()) {
    case kLength: {
      set_length(from.length());
      break;
    }
    case HAS_LENGTH_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.TensorSliceProto.Extent)
}

void TensorSliceProto_Extent::SharedCtor() {
  start_ = GOOGLE_LONGLONG(0);
  clear_has_has_length();
}

TensorSliceProto_Extent::~TensorSliceProto_Extent() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.TensorSliceProto.Extent)
  SharedDtor();
}

void TensorSliceProto_Extent::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  if (has_has_length()) {
    clear_has_length();
  }
}

void TensorSliceProto_Extent::ArenaDtor(void* object) {
  TensorSliceProto_Extent* _this = reinterpret_cast< TensorSliceProto_Extent* >(object);
  (void)_this;
}
void TensorSliceProto_Extent::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void TensorSliceProto_Extent::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* TensorSliceProto_Extent::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const TensorSliceProto_Extent& TensorSliceProto_Extent::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto_Extent.base);
  return *internal_default_instance();
}


void TensorSliceProto_Extent::clear_has_length() {
// @@protoc_insertion_point(one_of_clear_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  switch (has_length_case()) {
    case kLength: {
      // No need to clear
      break;
    }
    case HAS_LENGTH_NOT_SET: {
      break;
    }
  }
  _oneof_case_[0] = HAS_LENGTH_NOT_SET;
}


void TensorSliceProto_Extent::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  start_ = GOOGLE_LONGLONG(0);
  clear_has_length();
  _internal_metadata_.Clear();
}

bool TensorSliceProto_Extent::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // int64 start = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u /* 8 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &start_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int64 length = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(16u /* 16 & 0xFF */)) {
          clear_has_length();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &has_length_.length_)));
          set_has_length();
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.TensorSliceProto.Extent)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.TensorSliceProto.Extent)
  return false;
#undef DO_
}

void TensorSliceProto_Extent::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 start = 1;
  if (this->start() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(1, this->start(), output);
  }

  // int64 length = 2;
  if (has_length()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(2, this->length(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.TensorSliceProto.Extent)
}

::google::protobuf::uint8* TensorSliceProto_Extent::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 start = 1;
  if (this->start() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(1, this->start(), target);
  }

  // int64 length = 2;
  if (has_length()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(2, this->length(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.TensorSliceProto.Extent)
  return target;
}

size_t TensorSliceProto_Extent::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // int64 start = 1;
  if (this->start() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int64Size(
        this->start());
  }

  switch (has_length_case()) {
    // int64 length = 2;
    case kLength: {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int64Size(
          this->length());
      break;
    }
    case HAS_LENGTH_NOT_SET: {
      break;
    }
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TensorSliceProto_Extent::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorSliceProto_Extent* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const TensorSliceProto_Extent>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.TensorSliceProto.Extent)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.TensorSliceProto.Extent)
    MergeFrom(*source);
  }
}

void TensorSliceProto_Extent::MergeFrom(const TensorSliceProto_Extent& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.start() != 0) {
    set_start(from.start());
  }
  switch (from.has_length_case()) {
    case kLength: {
      set_length(from.length());
      break;
    }
    case HAS_LENGTH_NOT_SET: {
      break;
    }
  }
}

void TensorSliceProto_Extent::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorSliceProto_Extent::CopyFrom(const TensorSliceProto_Extent& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.TensorSliceProto.Extent)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorSliceProto_Extent::IsInitialized() const {
  return true;
}

void TensorSliceProto_Extent::Swap(TensorSliceProto_Extent* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    TensorSliceProto_Extent* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void TensorSliceProto_Extent::UnsafeArenaSwap(TensorSliceProto_Extent* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void TensorSliceProto_Extent::InternalSwap(TensorSliceProto_Extent* other) {
  using std::swap;
  swap(start_, other->start_);
  swap(has_length_, other->has_length_);
  swap(_oneof_case_[0], other->_oneof_case_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata TensorSliceProto_Extent::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::file_level_metadata[kIndexInFileMessages];
}


// ===================================================================

void TensorSliceProto::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorSliceProto::kExtentFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

TensorSliceProto::TensorSliceProto()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.TensorSliceProto)
}
TensorSliceProto::TensorSliceProto(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  extent_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.TensorSliceProto)
}
TensorSliceProto::TensorSliceProto(const TensorSliceProto& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      extent_(from.extent_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.TensorSliceProto)
}

void TensorSliceProto::SharedCtor() {
}

TensorSliceProto::~TensorSliceProto() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.TensorSliceProto)
  SharedDtor();
}

void TensorSliceProto::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
}

void TensorSliceProto::ArenaDtor(void* object) {
  TensorSliceProto* _this = reinterpret_cast< TensorSliceProto* >(object);
  (void)_this;
}
void TensorSliceProto::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void TensorSliceProto::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* TensorSliceProto::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const TensorSliceProto& TensorSliceProto::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::scc_info_TensorSliceProto.base);
  return *internal_default_instance();
}


void TensorSliceProto::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.TensorSliceProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  extent_.Clear();
  _internal_metadata_.Clear();
}

bool TensorSliceProto::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.TensorSliceProto)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .diplomacy.tensorflow.TensorSliceProto.Extent extent = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_extent()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.TensorSliceProto)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.TensorSliceProto)
  return false;
#undef DO_
}

void TensorSliceProto::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.TensorSliceProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .diplomacy.tensorflow.TensorSliceProto.Extent extent = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->extent_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->extent(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.TensorSliceProto)
}

::google::protobuf::uint8* TensorSliceProto::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.TensorSliceProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .diplomacy.tensorflow.TensorSliceProto.Extent extent = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->extent_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->extent(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.TensorSliceProto)
  return target;
}

size_t TensorSliceProto::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.TensorSliceProto)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .diplomacy.tensorflow.TensorSliceProto.Extent extent = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->extent_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->extent(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TensorSliceProto::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.TensorSliceProto)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorSliceProto* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const TensorSliceProto>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.TensorSliceProto)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.TensorSliceProto)
    MergeFrom(*source);
  }
}

void TensorSliceProto::MergeFrom(const TensorSliceProto& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.TensorSliceProto)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  extent_.MergeFrom(from.extent_);
}

void TensorSliceProto::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.TensorSliceProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorSliceProto::CopyFrom(const TensorSliceProto& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.TensorSliceProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorSliceProto::IsInitialized() const {
  return true;
}

void TensorSliceProto::Swap(TensorSliceProto* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    TensorSliceProto* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void TensorSliceProto::UnsafeArenaSwap(TensorSliceProto* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void TensorSliceProto::InternalSwap(TensorSliceProto* other) {
  using std::swap;
  CastToBase(&extent_)->InternalSwap(CastToBase(&other->extent_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata TensorSliceProto::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::TensorSliceProto_Extent* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::TensorSliceProto_Extent >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::TensorSliceProto_Extent >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::TensorSliceProto* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::TensorSliceProto >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::TensorSliceProto >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
