// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/core/protobuf/cluster.proto

#include "diplomacy_tensorflow/core/protobuf/cluster.pb.h"

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

namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto ::google::protobuf::internal::SCCInfo<0> scc_info_JobDef_TasksEntry_DoNotUse;
extern PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto ::google::protobuf::internal::SCCInfo<1> scc_info_JobDef;
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto
namespace diplomacy {
namespace tensorflow {
class JobDef_TasksEntry_DoNotUseDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<JobDef_TasksEntry_DoNotUse>
      _instance;
} _JobDef_TasksEntry_DoNotUse_default_instance_;
class JobDefDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<JobDef>
      _instance;
} _JobDef_default_instance_;
class ClusterDefDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<ClusterDef>
      _instance;
} _ClusterDef_default_instance_;
}  // namespace tensorflow
}  // namespace diplomacy
namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto {
static void InitDefaultsJobDef_TasksEntry_DoNotUse() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_JobDef_TasksEntry_DoNotUse_default_instance_;
    new (ptr) ::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse();
  }
  ::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_JobDef_TasksEntry_DoNotUse =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsJobDef_TasksEntry_DoNotUse}, {}};

static void InitDefaultsJobDef() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_JobDef_default_instance_;
    new (ptr) ::diplomacy::tensorflow::JobDef();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::JobDef::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_JobDef =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsJobDef}, {
      &protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_JobDef_TasksEntry_DoNotUse.base,}};

static void InitDefaultsClusterDef() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::diplomacy::tensorflow::_ClusterDef_default_instance_;
    new (ptr) ::diplomacy::tensorflow::ClusterDef();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::diplomacy::tensorflow::ClusterDef::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_ClusterDef =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsClusterDef}, {
      &protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_JobDef.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_JobDef_TasksEntry_DoNotUse.base);
  ::google::protobuf::internal::InitSCC(&scc_info_JobDef.base);
  ::google::protobuf::internal::InitSCC(&scc_info_ClusterDef.base);
}

::google::protobuf::Metadata file_level_metadata[3];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse, key_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse, value_),
  0,
  1,
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef, name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::JobDef, tasks_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ClusterDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::diplomacy::tensorflow::ClusterDef, job_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse)},
  { 9, -1, sizeof(::diplomacy::tensorflow::JobDef)},
  { 16, -1, sizeof(::diplomacy::tensorflow::ClusterDef)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_JobDef_TasksEntry_DoNotUse_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_JobDef_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::diplomacy::tensorflow::_ClusterDef_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "diplomacy_tensorflow/core/protobuf/cluster.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 3);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n0diplomacy_tensorflow/core/protobuf/clu"
      "ster.proto\022\024diplomacy.tensorflow\"|\n\006JobD"
      "ef\022\014\n\004name\030\001 \001(\t\0226\n\005tasks\030\002 \003(\0132\'.diplom"
      "acy.tensorflow.JobDef.TasksEntry\032,\n\nTask"
      "sEntry\022\013\n\003key\030\001 \001(\005\022\r\n\005value\030\002 \001(\t:\0028\001\"7"
      "\n\nClusterDef\022)\n\003job\030\001 \003(\0132\034.diplomacy.te"
      "nsorflow.JobDefBn\n\032org.tensorflow.distru"
      "ntimeB\rClusterProtosP\001Z<github.com/tenso"
      "rflow/tensorflow/tensorflow/go/core/prot"
      "obuf\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 375);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "diplomacy_tensorflow/core/protobuf/cluster.proto", &protobuf_RegisterTypes);
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
}  // namespace protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto
namespace diplomacy {
namespace tensorflow {

// ===================================================================

JobDef_TasksEntry_DoNotUse::JobDef_TasksEntry_DoNotUse() {}
JobDef_TasksEntry_DoNotUse::JobDef_TasksEntry_DoNotUse(::google::protobuf::Arena* arena) : SuperType(arena) {}
void JobDef_TasksEntry_DoNotUse::MergeFrom(const JobDef_TasksEntry_DoNotUse& other) {
  MergeFromInternal(other);
}
::google::protobuf::Metadata JobDef_TasksEntry_DoNotUse::GetMetadata() const {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::file_level_metadata[0];
}
void JobDef_TasksEntry_DoNotUse::MergeFrom(
    const ::google::protobuf::Message& other) {
  ::google::protobuf::Message::MergeFrom(other);
}


// ===================================================================

void JobDef::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int JobDef::kNameFieldNumber;
const int JobDef::kTasksFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

JobDef::JobDef()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_JobDef.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.JobDef)
}
JobDef::JobDef(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  tasks_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_JobDef.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.JobDef)
}
JobDef::JobDef(const JobDef& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  tasks_.MergeFrom(from.tasks_);
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.JobDef)
}

void JobDef::SharedCtor() {
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

JobDef::~JobDef() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.JobDef)
  SharedDtor();
}

void JobDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void JobDef::ArenaDtor(void* object) {
  JobDef* _this = reinterpret_cast< JobDef* >(object);
  (void)_this;
}
void JobDef::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void JobDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* JobDef::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const JobDef& JobDef::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_JobDef.base);
  return *internal_default_instance();
}


void JobDef::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.JobDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  tasks_.Clear();
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  _internal_metadata_.Clear();
}

bool JobDef::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.JobDef)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string name = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.JobDef.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // map<int32, string> tasks = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          JobDef_TasksEntry_DoNotUse::Parser< ::google::protobuf::internal::MapField<
              JobDef_TasksEntry_DoNotUse,
              ::google::protobuf::int32, ::std::string,
              ::google::protobuf::internal::WireFormatLite::TYPE_INT32,
              ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
              0 >,
            ::google::protobuf::Map< ::google::protobuf::int32, ::std::string > > parser(&tasks_);
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
              input, &parser));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            parser.value().data(), static_cast<int>(parser.value().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "diplomacy.tensorflow.JobDef.TasksEntry.value"));
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
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.JobDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.JobDef)
  return false;
#undef DO_
}

void JobDef::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.JobDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.JobDef.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->name(), output);
  }

  // map<int32, string> tasks = 2;
  if (!this->tasks().empty()) {
    typedef ::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_pointer
        ConstPtr;
    typedef ::google::protobuf::internal::SortItem< ::google::protobuf::int32, ConstPtr > SortItem;
    typedef ::google::protobuf::internal::CompareByFirstField<SortItem> Less;
    struct Utf8Check {
      static void Check(ConstPtr p) {
        ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
          p->second.data(), static_cast<int>(p->second.length()),
          ::google::protobuf::internal::WireFormatLite::SERIALIZE,
          "diplomacy.tensorflow.JobDef.TasksEntry.value");
      }
    };

    if (output->IsSerializationDeterministic() &&
        this->tasks().size() > 1) {
      ::std::unique_ptr<SortItem[]> items(
          new SortItem[this->tasks().size()]);
      typedef ::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::size_type size_type;
      size_type n = 0;
      for (::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_iterator
          it = this->tasks().begin();
          it != this->tasks().end(); ++it, ++n) {
        items[static_cast<ptrdiff_t>(n)] = SortItem(&*it);
      }
      ::std::sort(&items[0], &items[static_cast<ptrdiff_t>(n)], Less());
      ::std::unique_ptr<JobDef_TasksEntry_DoNotUse> entry;
      for (size_type i = 0; i < n; i++) {
        entry.reset(tasks_.NewEntryWrapper(
            items[static_cast<ptrdiff_t>(i)].second->first, items[static_cast<ptrdiff_t>(i)].second->second));
        ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
            2, *entry, output);
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(items[static_cast<ptrdiff_t>(i)].second);
      }
    } else {
      ::std::unique_ptr<JobDef_TasksEntry_DoNotUse> entry;
      for (::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_iterator
          it = this->tasks().begin();
          it != this->tasks().end(); ++it) {
        entry.reset(tasks_.NewEntryWrapper(
            it->first, it->second));
        ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
            2, *entry, output);
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(&*it);
      }
    }
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.JobDef)
}

::google::protobuf::uint8* JobDef::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.JobDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "diplomacy.tensorflow.JobDef.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->name(), target);
  }

  // map<int32, string> tasks = 2;
  if (!this->tasks().empty()) {
    typedef ::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_pointer
        ConstPtr;
    typedef ::google::protobuf::internal::SortItem< ::google::protobuf::int32, ConstPtr > SortItem;
    typedef ::google::protobuf::internal::CompareByFirstField<SortItem> Less;
    struct Utf8Check {
      static void Check(ConstPtr p) {
        ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
          p->second.data(), static_cast<int>(p->second.length()),
          ::google::protobuf::internal::WireFormatLite::SERIALIZE,
          "diplomacy.tensorflow.JobDef.TasksEntry.value");
      }
    };

    if (deterministic &&
        this->tasks().size() > 1) {
      ::std::unique_ptr<SortItem[]> items(
          new SortItem[this->tasks().size()]);
      typedef ::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::size_type size_type;
      size_type n = 0;
      for (::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_iterator
          it = this->tasks().begin();
          it != this->tasks().end(); ++it, ++n) {
        items[static_cast<ptrdiff_t>(n)] = SortItem(&*it);
      }
      ::std::sort(&items[0], &items[static_cast<ptrdiff_t>(n)], Less());
      ::std::unique_ptr<JobDef_TasksEntry_DoNotUse> entry;
      for (size_type i = 0; i < n; i++) {
        entry.reset(tasks_.NewEntryWrapper(
            items[static_cast<ptrdiff_t>(i)].second->first, items[static_cast<ptrdiff_t>(i)].second->second));
        target = ::google::protobuf::internal::WireFormatLite::
                   InternalWriteMessageNoVirtualToArray(
                       2, *entry, deterministic, target);
;
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(items[static_cast<ptrdiff_t>(i)].second);
      }
    } else {
      ::std::unique_ptr<JobDef_TasksEntry_DoNotUse> entry;
      for (::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_iterator
          it = this->tasks().begin();
          it != this->tasks().end(); ++it) {
        entry.reset(tasks_.NewEntryWrapper(
            it->first, it->second));
        target = ::google::protobuf::internal::WireFormatLite::
                   InternalWriteMessageNoVirtualToArray(
                       2, *entry, deterministic, target);
;
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(&*it);
      }
    }
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.JobDef)
  return target;
}

size_t JobDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.JobDef)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // map<int32, string> tasks = 2;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->tasks_size());
  {
    ::std::unique_ptr<JobDef_TasksEntry_DoNotUse> entry;
    for (::google::protobuf::Map< ::google::protobuf::int32, ::std::string >::const_iterator
        it = this->tasks().begin();
        it != this->tasks().end(); ++it) {
      if (entry.get() != NULL && entry->GetArena() != NULL) {
        entry.release();
      }
      entry.reset(tasks_.NewEntryWrapper(it->first, it->second));
      total_size += ::google::protobuf::internal::WireFormatLite::
          MessageSizeNoVirtual(*entry);
    }
    if (entry.get() != NULL && entry->GetArena() != NULL) {
      entry.release();
    }
  }

  // string name = 1;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void JobDef::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.JobDef)
  GOOGLE_DCHECK_NE(&from, this);
  const JobDef* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const JobDef>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.JobDef)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.JobDef)
    MergeFrom(*source);
  }
}

void JobDef::MergeFrom(const JobDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.JobDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  tasks_.MergeFrom(from.tasks_);
  if (from.name().size() > 0) {
    set_name(from.name());
  }
}

void JobDef::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.JobDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void JobDef::CopyFrom(const JobDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.JobDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool JobDef::IsInitialized() const {
  return true;
}

void JobDef::Swap(JobDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    JobDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void JobDef::UnsafeArenaSwap(JobDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void JobDef::InternalSwap(JobDef* other) {
  using std::swap;
  tasks_.Swap(&other->tasks_);
  name_.Swap(&other->name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata JobDef::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::file_level_metadata[kIndexInFileMessages];
}


// ===================================================================

void ClusterDef::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ClusterDef::kJobFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

ClusterDef::ClusterDef()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_ClusterDef.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:diplomacy.tensorflow.ClusterDef)
}
ClusterDef::ClusterDef(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  job_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_ClusterDef.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:diplomacy.tensorflow.ClusterDef)
}
ClusterDef::ClusterDef(const ClusterDef& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      job_(from.job_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:diplomacy.tensorflow.ClusterDef)
}

void ClusterDef::SharedCtor() {
}

ClusterDef::~ClusterDef() {
  // @@protoc_insertion_point(destructor:diplomacy.tensorflow.ClusterDef)
  SharedDtor();
}

void ClusterDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
}

void ClusterDef::ArenaDtor(void* object) {
  ClusterDef* _this = reinterpret_cast< ClusterDef* >(object);
  (void)_this;
}
void ClusterDef::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void ClusterDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* ClusterDef::descriptor() {
  ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const ClusterDef& ClusterDef::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::scc_info_ClusterDef.base);
  return *internal_default_instance();
}


void ClusterDef::Clear() {
// @@protoc_insertion_point(message_clear_start:diplomacy.tensorflow.ClusterDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  job_.Clear();
  _internal_metadata_.Clear();
}

bool ClusterDef::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:diplomacy.tensorflow.ClusterDef)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .diplomacy.tensorflow.JobDef job = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_job()));
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
  // @@protoc_insertion_point(parse_success:diplomacy.tensorflow.ClusterDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:diplomacy.tensorflow.ClusterDef)
  return false;
#undef DO_
}

void ClusterDef::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:diplomacy.tensorflow.ClusterDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .diplomacy.tensorflow.JobDef job = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->job_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->job(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:diplomacy.tensorflow.ClusterDef)
}

::google::protobuf::uint8* ClusterDef::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:diplomacy.tensorflow.ClusterDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .diplomacy.tensorflow.JobDef job = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->job_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->job(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:diplomacy.tensorflow.ClusterDef)
  return target;
}

size_t ClusterDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:diplomacy.tensorflow.ClusterDef)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .diplomacy.tensorflow.JobDef job = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->job_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->job(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ClusterDef::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:diplomacy.tensorflow.ClusterDef)
  GOOGLE_DCHECK_NE(&from, this);
  const ClusterDef* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const ClusterDef>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:diplomacy.tensorflow.ClusterDef)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:diplomacy.tensorflow.ClusterDef)
    MergeFrom(*source);
  }
}

void ClusterDef::MergeFrom(const ClusterDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:diplomacy.tensorflow.ClusterDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  job_.MergeFrom(from.job_);
}

void ClusterDef::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:diplomacy.tensorflow.ClusterDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ClusterDef::CopyFrom(const ClusterDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:diplomacy.tensorflow.ClusterDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ClusterDef::IsInitialized() const {
  return true;
}

void ClusterDef::Swap(ClusterDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    ClusterDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void ClusterDef::UnsafeArenaSwap(ClusterDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void ClusterDef::InternalSwap(ClusterDef* other) {
  using std::swap;
  CastToBase(&job_)->InternalSwap(CastToBase(&other->job_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata ClusterDef::GetMetadata() const {
  protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_diplomacy_5ftensorflow_2fcore_2fprotobuf_2fcluster_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
}  // namespace diplomacy
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::JobDef_TasksEntry_DoNotUse >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::JobDef* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::JobDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::JobDef >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::diplomacy::tensorflow::ClusterDef* Arena::CreateMaybeMessage< ::diplomacy::tensorflow::ClusterDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::diplomacy::tensorflow::ClusterDef >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
