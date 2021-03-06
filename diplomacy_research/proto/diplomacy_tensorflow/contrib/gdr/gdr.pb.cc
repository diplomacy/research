// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/contrib/gdr/gdr.proto

#include "diplomacy_tensorflow/contrib/gdr/gdr.pb.h"

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

namespace diplomacy {
namespace tensorflow {
class RemoteMemoryRegionDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<RemoteMemoryRegion>
      _instance;
} _RemoteMemoryRegion_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto {
static void InitDefaultsRemoteMemoryRegion() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_RemoteMemoryRegion_default_instance_;
    new (ptr) ::diplomacy::tensorflow::RemoteMemoryRegion();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::RemoteMemoryRegion::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_RemoteMemoryRegion =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsRemoteMemoryRegion}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_RemoteMemoryRegion.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, host_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, port_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, addr_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, rkey_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::RemoteMemoryRegion, tensor_key_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::diplomacy::tensorflow::RemoteMemoryRegion)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_RemoteMemoryRegion_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/contrib/gdr/gdr.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n*diplomacy_tensorflow/contrib/gdr/gdr.p"
      "roto\022\024diplomacy.tensorflow\"`\n\022RemoteMemo"
      "ryRegion\022\014\n\004host\030\001 \001(\t\022\014\n\004port\030\002 \001(\t\022\014\n\004"
      "addr\030\003 \001(\004\022\014\n\004rkey\030\004 \001(\r\022\022\n\ntensor_key\030\005"
      " \001(\rB\003\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 177);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/contrib/gdr/gdr.proto", &protobuf_RegisterTypes);
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto
namespace diplomacy {
namespace tensorflow {

// ===================================================================

void RemoteMemoryRegion::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int RemoteMemoryRegion::kHostFieldNumber;
const int RemoteMemoryRegion::kPortFieldNumber;
const int RemoteMemoryRegion::kAddrFieldNumber;
const int RemoteMemoryRegion::kRkeyFieldNumber;
const int RemoteMemoryRegion::kTensorKeyFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

RemoteMemoryRegion::RemoteMemoryRegion()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::scc_info_RemoteMemoryRegion.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.RemoteMemoryRegion)
}
RemoteMemoryRegion::RemoteMemoryRegion(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::scc_info_RemoteMemoryRegion.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.RemoteMemoryRegion)
}
RemoteMemoryRegion::RemoteMemoryRegion(const RemoteMemoryRegion& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  host_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.host().size() > 0) {
    host_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.host(),
      GetArenaNoVirtual());
  }
  port_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.port().size() > 0) {
    port_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.port(),
      GetArenaNoVirtual());
  }
  ::memcpy(&addr_, &from.addr_,
    static_cast<size_t>(reinterpret_cast<char*>(&tensor_key_) -
    reinterpret_cast<char*>(&addr_)) + sizeof(tensor_key_));
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.RemoteMemoryRegion)
}

void RemoteMemoryRegion::SharedCtor() {
  host_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  port_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&addr_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&tensor_key_) -
      reinterpret_cast<char*>(&addr_)) + sizeof(tensor_key_));
}

RemoteMemoryRegion::~RemoteMemoryRegion() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.RemoteMemoryRegion)
  SharedDtor();
}

void RemoteMemoryRegion::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  host_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  port_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void RemoteMemoryRegion::ArenaDtor(void* object) {
  RemoteMemoryRegion* _this = reinterpret_cast< RemoteMemoryRegion* >(object);
  (void)_this;
}
void RemoteMemoryRegion::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void RemoteMemoryRegion::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* RemoteMemoryRegion::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const RemoteMemoryRegion& RemoteMemoryRegion::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::scc_info_RemoteMemoryRegion.base);
  return *internal_default_instance();
}


void RemoteMemoryRegion::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.RemoteMemoryRegion)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  host_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  port_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  ::memset(&addr_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&tensor_key_) -
      reinterpret_cast<char*>(&addr_)) + sizeof(tensor_key_));
  _internal_metadata_.Clear();
}

bool RemoteMemoryRegion::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.RemoteMemoryRegion)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string host = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_host()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->host().data(), static_cast<int>(this->host().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.RemoteMemoryRegion.host"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string port = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_port()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->port().data(), static_cast<int>(this->port().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.RemoteMemoryRegion.port"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint64 addr = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &addr_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint32 rkey = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(32u /* 32 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &rkey_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint32 tensor_key = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(40u /* 40 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &tensor_key_)));
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
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.RemoteMemoryRegion)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.RemoteMemoryRegion)
  return false;
#undef DO_
}

void RemoteMemoryRegion::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.RemoteMemoryRegion)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string host = 1;
  if (this->host().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->host().data(), static_cast<int>(this->host().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.RemoteMemoryRegion.host");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->host(), output);
  }

  // string port = 2;
  if (this->port().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->port().data(), static_cast<int>(this->port().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.RemoteMemoryRegion.port");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->port(), output);
  }

  // uint64 addr = 3;
  if (this->addr() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(3, this->addr(), output);
  }

  // uint32 rkey = 4;
  if (this->rkey() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(4, this->rkey(), output);
  }

  // uint32 tensor_key = 5;
  if (this->tensor_key() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(5, this->tensor_key(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.RemoteMemoryRegion)
}

::google::protobuf::uint8* RemoteMemoryRegion::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.RemoteMemoryRegion)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string host = 1;
  if (this->host().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->host().data(), static_cast<int>(this->host().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.RemoteMemoryRegion.host");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->host(), target);
  }

  // string port = 2;
  if (this->port().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->port().data(), static_cast<int>(this->port().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.RemoteMemoryRegion.port");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->port(), target);
  }

  // uint64 addr = 3;
  if (this->addr() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(3, this->addr(), target);
  }

  // uint32 rkey = 4;
  if (this->rkey() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(4, this->rkey(), target);
  }

  // uint32 tensor_key = 5;
  if (this->tensor_key() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(5, this->tensor_key(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.RemoteMemoryRegion)
  return target;
}

size_t RemoteMemoryRegion::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.RemoteMemoryRegion)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string host = 1;
  if (this->host().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->host());
  }

  // string port = 2;
  if (this->port().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->port());
  }

  // uint64 addr = 3;
  if (this->addr() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt64Size(
        this->addr());
  }

  // uint32 rkey = 4;
  if (this->rkey() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->rkey());
  }

  // uint32 tensor_key = 5;
  if (this->tensor_key() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->tensor_key());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void RemoteMemoryRegion::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.RemoteMemoryRegion)
  GOOGLE_DCHECK_NE(&from, this);
  const RemoteMemoryRegion* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const RemoteMemoryRegion>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.RemoteMemoryRegion)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.RemoteMemoryRegion)
    MergeFrom(*source);
  }
}

void RemoteMemoryRegion::MergeFrom(const RemoteMemoryRegion& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.RemoteMemoryRegion)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.host().size() > 0) {
    set_host(from.host());
  }
  if (from.port().size() > 0) {
    set_port(from.port());
  }
  if (from.addr() != 0) {
    set_addr(from.addr());
  }
  if (from.rkey() != 0) {
    set_rkey(from.rkey());
  }
  if (from.tensor_key() != 0) {
    set_tensor_key(from.tensor_key());
  }
}

void RemoteMemoryRegion::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.RemoteMemoryRegion)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void RemoteMemoryRegion::CopyFrom(const RemoteMemoryRegion& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.RemoteMemoryRegion)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool RemoteMemoryRegion::IsInitialized() const {
  return true;
}

void RemoteMemoryRegion::Swap(RemoteMemoryRegion* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    RemoteMemoryRegion* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void RemoteMemoryRegion::UnsafeArenaSwap(RemoteMemoryRegion* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void RemoteMemoryRegion::InternalSwap(RemoteMemoryRegion* other) {
  using std::swap;
  host_.Swap(&other->host_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  port_.Swap(&other->port_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(addr_, other->addr_);
  swap(rkey_, other->rkey_);
  swap(tensor_key_, other->tensor_key_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata RemoteMemoryRegion::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcontrib_2fgdr_2fgdr_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::RemoteMemoryRegion* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::RemoteMemoryRegion >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::RemoteMemoryRegion >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
