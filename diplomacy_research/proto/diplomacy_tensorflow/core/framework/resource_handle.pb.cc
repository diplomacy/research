// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/framework/resource_handle.proto

#include "diplomacy_tensorflow/core/framework/resource_handle.pb.h"

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
class ResourceHandleProtoDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<ResourceHandleProto>
      _instance;
} _ResourceHandleProto_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto {
static void InitDefaultsResourceHandleProto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_ResourceHandleProto_default_instance_;
    new (ptr) ::diplomacy::tensorflow::ResourceHandleProto();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::ResourceHandleProto::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_ResourceHandleProto =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsResourceHandleProto}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_ResourceHandleProto.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, device_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, container_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, hash_code_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ResourceHandleProto, maybe_type_name_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::diplomacy::tensorflow::ResourceHandleProto)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_ResourceHandleProto_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/core/framework/resource_handle.proto", schemas, file_default_instances, TableStruct::offsets,
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
      "\n9diplomacy_tensorflow/core/framework/re"
      "source_handle.proto\022\024diplomacy.tensorflo"
      "w\"r\n\023ResourceHandleProto\022\016\n\006device\030\001 \001(\t"
      "\022\021\n\tcontainer\030\002 \001(\t\022\014\n\004name\030\003 \001(\t\022\021\n\thas"
      "h_code\030\004 \001(\004\022\027\n\017maybe_type_name\030\005 \001(\tBn\n"
      "\030org.tensorflow.frameworkB\016ResourceHandl"
      "eP\001Z=github.com/tensorflow/tensorflow/te"
      "nsorflow/go/core/framework\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 317);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/core/framework/resource_handle.proto", &protobuf_RegisterTypes);
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto
namespace diplomacy {
namespace tensorflow {

// ===================================================================

void ResourceHandleProto::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ResourceHandleProto::kDeviceFieldNumber;
const int ResourceHandleProto::kContainerFieldNumber;
const int ResourceHandleProto::kNameFieldNumber;
const int ResourceHandleProto::kHashCodeFieldNumber;
const int ResourceHandleProto::kMaybeTypeNameFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

ResourceHandleProto::ResourceHandleProto()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::scc_info_ResourceHandleProto.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.ResourceHandleProto)
}
ResourceHandleProto::ResourceHandleProto(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::scc_info_ResourceHandleProto.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.ResourceHandleProto)
}
ResourceHandleProto::ResourceHandleProto(const ResourceHandleProto& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  device_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.device().size() > 0) {
    device_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.device(),
      GetArenaNoVirtual());
  }
  container_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.container().size() > 0) {
    container_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.container(),
      GetArenaNoVirtual());
  }
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  maybe_type_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.maybe_type_name().size() > 0) {
    maybe_type_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.maybe_type_name(),
      GetArenaNoVirtual());
  }
  hash_code_ = from.hash_code_;
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.ResourceHandleProto)
}

void ResourceHandleProto::SharedCtor() {
  device_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  container_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  maybe_type_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  hash_code_ = GOOGLE_ULONGLONG(0);
}

ResourceHandleProto::~ResourceHandleProto() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.ResourceHandleProto)
  SharedDtor();
}

void ResourceHandleProto::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  device_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  container_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  maybe_type_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void ResourceHandleProto::ArenaDtor(void* object) {
  ResourceHandleProto* _this = reinterpret_cast< ResourceHandleProto* >(object);
  (void)_this;
}
void ResourceHandleProto::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void ResourceHandleProto::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* ResourceHandleProto::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const ResourceHandleProto& ResourceHandleProto::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::scc_info_ResourceHandleProto.base);
  return *internal_default_instance();
}


void ResourceHandleProto::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.ResourceHandleProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  device_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  container_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  maybe_type_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  hash_code_ = GOOGLE_ULONGLONG(0);
  _internal_metadata_.Clear();
}

bool ResourceHandleProto::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.ResourceHandleProto)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string device = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_device()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->device().data(), static_cast<int>(this->device().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.ResourceHandleProto.device"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string container = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_container()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->container().data(), static_cast<int>(this->container().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.ResourceHandleProto.container"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string name = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.ResourceHandleProto.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint64 hash_code = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(32u /* 32 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &hash_code_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string maybe_type_name = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(42u /* 42 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_maybe_type_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->maybe_type_name().data(), static_cast<int>(this->maybe_type_name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.ResourceHandleProto.maybe_type_name"));
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
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.ResourceHandleProto)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.ResourceHandleProto)
  return false;
#undef DO_
}

void ResourceHandleProto::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.ResourceHandleProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string device = 1;
  if (this->device().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->device().data(), static_cast<int>(this->device().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.device");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->device(), output);
  }

  // string container = 2;
  if (this->container().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->container().data(), static_cast<int>(this->container().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.container");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->container(), output);
  }

  // string name = 3;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      3, this->name(), output);
  }

  // uint64 hash_code = 4;
  if (this->hash_code() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(4, this->hash_code(), output);
  }

  // string maybe_type_name = 5;
  if (this->maybe_type_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->maybe_type_name().data(), static_cast<int>(this->maybe_type_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.maybe_type_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      5, this->maybe_type_name(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.ResourceHandleProto)
}

::google::protobuf::uint8* ResourceHandleProto::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.ResourceHandleProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string device = 1;
  if (this->device().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->device().data(), static_cast<int>(this->device().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.device");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->device(), target);
  }

  // string container = 2;
  if (this->container().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->container().data(), static_cast<int>(this->container().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.container");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->container(), target);
  }

  // string name = 3;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        3, this->name(), target);
  }

  // uint64 hash_code = 4;
  if (this->hash_code() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(4, this->hash_code(), target);
  }

  // string maybe_type_name = 5;
  if (this->maybe_type_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->maybe_type_name().data(), static_cast<int>(this->maybe_type_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.ResourceHandleProto.maybe_type_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        5, this->maybe_type_name(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.ResourceHandleProto)
  return target;
}

size_t ResourceHandleProto::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.ResourceHandleProto)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string device = 1;
  if (this->device().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->device());
  }

  // string container = 2;
  if (this->container().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->container());
  }

  // string name = 3;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  // string maybe_type_name = 5;
  if (this->maybe_type_name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->maybe_type_name());
  }

  // uint64 hash_code = 4;
  if (this->hash_code() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt64Size(
        this->hash_code());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ResourceHandleProto::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.ResourceHandleProto)
  GOOGLE_DCHECK_NE(&from, this);
  const ResourceHandleProto* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const ResourceHandleProto>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.ResourceHandleProto)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.ResourceHandleProto)
    MergeFrom(*source);
  }
}

void ResourceHandleProto::MergeFrom(const ResourceHandleProto& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.ResourceHandleProto)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.device().size() > 0) {
    set_device(from.device());
  }
  if (from.container().size() > 0) {
    set_container(from.container());
  }
  if (from.name().size() > 0) {
    set_name(from.name());
  }
  if (from.maybe_type_name().size() > 0) {
    set_maybe_type_name(from.maybe_type_name());
  }
  if (from.hash_code() != 0) {
    set_hash_code(from.hash_code());
  }
}

void ResourceHandleProto::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.ResourceHandleProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ResourceHandleProto::CopyFrom(const ResourceHandleProto& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.ResourceHandleProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ResourceHandleProto::IsInitialized() const {
  return true;
}

void ResourceHandleProto::Swap(ResourceHandleProto* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    ResourceHandleProto* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void ResourceHandleProto::UnsafeArenaSwap(ResourceHandleProto* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void ResourceHandleProto::InternalSwap(ResourceHandleProto* other) {
  using std::swap;
  device_.Swap(&other->device_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  container_.Swap(&other->container_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  name_.Swap(&other->name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  maybe_type_name_.Swap(&other->maybe_type_name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(hash_code_, other->hash_code_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata ResourceHandleProto::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::ResourceHandleProto* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::ResourceHandleProto >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::ResourceHandleProto >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
