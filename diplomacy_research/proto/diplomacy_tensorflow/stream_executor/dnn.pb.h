// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: diplomacy_tensorflow/stream_executor/dnn.proto

#ifndef PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto
#define PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto

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
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto 

namespace protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[3];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto
namespace stream_executor {
namespace dnn {
class AlgorithmProto;
class AlgorithmProtoDefaultTypeInternal;
extern AlgorithmProtoDefaultTypeInternal _AlgorithmProto_default_instance_;
class ConvolutionDescriptorProto;
class ConvolutionDescriptorProtoDefaultTypeInternal;
extern ConvolutionDescriptorProtoDefaultTypeInternal _ConvolutionDescriptorProto_default_instance_;
class TensorDescriptorProto;
class TensorDescriptorProtoDefaultTypeInternal;
extern TensorDescriptorProtoDefaultTypeInternal _TensorDescriptorProto_default_instance_;
}  // namespace dnn
}  // namespace stream_executor
namespace google {
namespace protobuf {
template<> ::stream_executor::dnn::AlgorithmProto* Arena::CreateMaybeMessage<::stream_executor::dnn::AlgorithmProto>(Arena*);
template<> ::stream_executor::dnn::ConvolutionDescriptorProto* Arena::CreateMaybeMessage<::stream_executor::dnn::ConvolutionDescriptorProto>(Arena*);
template<> ::stream_executor::dnn::TensorDescriptorProto* Arena::CreateMaybeMessage<::stream_executor::dnn::TensorDescriptorProto>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace stream_executor {
namespace dnn {

enum AlgorithmProto_MathType {
  AlgorithmProto_MathType_DEFAULT_MATH = 0,
  AlgorithmProto_MathType_TENSOR_OP_MATH = 1,
  AlgorithmProto_MathType_AlgorithmProto_MathType_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  AlgorithmProto_MathType_AlgorithmProto_MathType_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool AlgorithmProto_MathType_IsValid(int value);
const AlgorithmProto_MathType AlgorithmProto_MathType_MathType_MIN = AlgorithmProto_MathType_DEFAULT_MATH;
const AlgorithmProto_MathType AlgorithmProto_MathType_MathType_MAX = AlgorithmProto_MathType_TENSOR_OP_MATH;
const int AlgorithmProto_MathType_MathType_ARRAYSIZE = AlgorithmProto_MathType_MathType_MAX + 1;

const ::google::protobuf::EnumDescriptor* AlgorithmProto_MathType_descriptor();
inline const ::std::string& AlgorithmProto_MathType_Name(AlgorithmProto_MathType value) {
  return ::google::protobuf::internal::NameOfEnum(
    AlgorithmProto_MathType_descriptor(), value);
}
inline bool AlgorithmProto_MathType_Parse(
    const ::std::string& name, AlgorithmProto_MathType* value) {
  return ::google::protobuf::internal::ParseNamedEnum<AlgorithmProto_MathType>(
    AlgorithmProto_MathType_descriptor(), name, value);
}
enum DataType {
  kFloat = 0,
  kDouble = 1,
  kHalf = 2,
  kInt8 = 3,
  kInt32 = 4,
  DataType_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  DataType_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool DataType_IsValid(int value);
const DataType DataType_MIN = kFloat;
const DataType DataType_MAX = kInt32;
const int DataType_ARRAYSIZE = DataType_MAX + 1;

const ::google::protobuf::EnumDescriptor* DataType_descriptor();
inline const ::std::string& DataType_Name(DataType value) {
  return ::google::protobuf::internal::NameOfEnum(
    DataType_descriptor(), value);
}
inline bool DataType_Parse(
    const ::std::string& name, DataType* value) {
  return ::google::protobuf::internal::ParseNamedEnum<DataType>(
    DataType_descriptor(), name, value);
}
enum DataLayout {
  kYXDepthBatch = 0,
  kYXBatchDepth = 1,
  kBatchYXDepth = 2,
  kBatchDepthYX = 3,
  kBatchDepthYX4 = 4,
  DataLayout_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  DataLayout_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool DataLayout_IsValid(int value);
const DataLayout DataLayout_MIN = kYXDepthBatch;
const DataLayout DataLayout_MAX = kBatchDepthYX4;
const int DataLayout_ARRAYSIZE = DataLayout_MAX + 1;

const ::google::protobuf::EnumDescriptor* DataLayout_descriptor();
inline const ::std::string& DataLayout_Name(DataLayout value) {
  return ::google::protobuf::internal::NameOfEnum(
    DataLayout_descriptor(), value);
}
inline bool DataLayout_Parse(
    const ::std::string& name, DataLayout* value) {
  return ::google::protobuf::internal::ParseNamedEnum<DataLayout>(
    DataLayout_descriptor(), name, value);
}
enum FilterLayout {
  kOutputInputYX = 0,
  kOutputYXInput = 1,
  kOutputInputYX4 = 2,
  kInputYXOutput = 3,
  kYXInputOutput = 4,
  FilterLayout_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  FilterLayout_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool FilterLayout_IsValid(int value);
const FilterLayout FilterLayout_MIN = kOutputInputYX;
const FilterLayout FilterLayout_MAX = kYXInputOutput;
const int FilterLayout_ARRAYSIZE = FilterLayout_MAX + 1;

const ::google::protobuf::EnumDescriptor* FilterLayout_descriptor();
inline const ::std::string& FilterLayout_Name(FilterLayout value) {
  return ::google::protobuf::internal::NameOfEnum(
    FilterLayout_descriptor(), value);
}
inline bool FilterLayout_Parse(
    const ::std::string& name, FilterLayout* value) {
  return ::google::protobuf::internal::ParseNamedEnum<FilterLayout>(
    FilterLayout_descriptor(), name, value);
}
enum ActivationMode {
  kNone = 0,
  kSigmoid = 1,
  kRelu = 2,
  kRelu6 = 3,
  kReluX = 4,
  kTanh = 5,
  kBandPass = 6,
  ActivationMode_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  ActivationMode_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool ActivationMode_IsValid(int value);
const ActivationMode ActivationMode_MIN = kNone;
const ActivationMode ActivationMode_MAX = kBandPass;
const int ActivationMode_ARRAYSIZE = ActivationMode_MAX + 1;

const ::google::protobuf::EnumDescriptor* ActivationMode_descriptor();
inline const ::std::string& ActivationMode_Name(ActivationMode value) {
  return ::google::protobuf::internal::NameOfEnum(
    ActivationMode_descriptor(), value);
}
inline bool ActivationMode_Parse(
    const ::std::string& name, ActivationMode* value) {
  return ::google::protobuf::internal::ParseNamedEnum<ActivationMode>(
    ActivationMode_descriptor(), name, value);
}
enum ConvolutionMode {
  CROSS_CORRELATION = 0,
  CONVOLUTION = 1,
  ConvolutionMode_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  ConvolutionMode_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool ConvolutionMode_IsValid(int value);
const ConvolutionMode ConvolutionMode_MIN = CROSS_CORRELATION;
const ConvolutionMode ConvolutionMode_MAX = CONVOLUTION;
const int ConvolutionMode_ARRAYSIZE = ConvolutionMode_MAX + 1;

const ::google::protobuf::EnumDescriptor* ConvolutionMode_descriptor();
inline const ::std::string& ConvolutionMode_Name(ConvolutionMode value) {
  return ::google::protobuf::internal::NameOfEnum(
    ConvolutionMode_descriptor(), value);
}
inline bool ConvolutionMode_Parse(
    const ::std::string& name, ConvolutionMode* value) {
  return ::google::protobuf::internal::ParseNamedEnum<ConvolutionMode>(
    ConvolutionMode_descriptor(), name, value);
}
// ===================================================================

class TensorDescriptorProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:stream_executor.dnn.TensorDescriptorProto) */ {
 public:
  TensorDescriptorProto();
  virtual ~TensorDescriptorProto();

  TensorDescriptorProto(const TensorDescriptorProto& from);

  inline TensorDescriptorProto& operator=(const TensorDescriptorProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TensorDescriptorProto(TensorDescriptorProto&& from) noexcept
    : TensorDescriptorProto() {
    *this = ::std::move(from);
  }

  inline TensorDescriptorProto& operator=(TensorDescriptorProto&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const TensorDescriptorProto& default_instance();

  enum LayoutOneofCase {
    kDataLayout = 3,
    kFilterLayout = 4,
    LAYOUT_ONEOF_NOT_SET = 0,
  };

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const TensorDescriptorProto* internal_default_instance() {
    return reinterpret_cast<const TensorDescriptorProto*>(
               &_TensorDescriptorProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(TensorDescriptorProto* other);
  friend void swap(TensorDescriptorProto& a, TensorDescriptorProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TensorDescriptorProto* New() const final {
    return CreateMaybeMessage<TensorDescriptorProto>(NULL);
  }

  TensorDescriptorProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<TensorDescriptorProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const TensorDescriptorProto& from);
  void MergeFrom(const TensorDescriptorProto& from);
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
  void InternalSwap(TensorDescriptorProto* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int64 dimensions = 1;
  int dimensions_size() const;
  void clear_dimensions();
  static const int kDimensionsFieldNumber = 1;
  ::google::protobuf::int64 dimensions(int index) const;
  void set_dimensions(int index, ::google::protobuf::int64 value);
  void add_dimensions(::google::protobuf::int64 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
      dimensions() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
      mutable_dimensions();

  // .stream_executor.dnn.DataType data_type = 2;
  void clear_data_type();
  static const int kDataTypeFieldNumber = 2;
  ::stream_executor::dnn::DataType data_type() const;
  void set_data_type(::stream_executor::dnn::DataType value);

  // .stream_executor.dnn.DataLayout data_layout = 3;
  private:
  bool has_data_layout() const;
  public:
  void clear_data_layout();
  static const int kDataLayoutFieldNumber = 3;
  ::stream_executor::dnn::DataLayout data_layout() const;
  void set_data_layout(::stream_executor::dnn::DataLayout value);

  // .stream_executor.dnn.FilterLayout filter_layout = 4;
  private:
  bool has_filter_layout() const;
  public:
  void clear_filter_layout();
  static const int kFilterLayoutFieldNumber = 4;
  ::stream_executor::dnn::FilterLayout filter_layout() const;
  void set_filter_layout(::stream_executor::dnn::FilterLayout value);

  void clear_layout_oneof();
  LayoutOneofCase layout_oneof_case() const;
  // @@protoc_insertion_point(class_scope:stream_executor.dnn.TensorDescriptorProto)
 private:
  void set_has_data_layout();
  void set_has_filter_layout();

  inline bool has_layout_oneof() const;
  inline void clear_has_layout_oneof();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 > dimensions_;
  mutable int _dimensions_cached_byte_size_;
  int data_type_;
  union LayoutOneofUnion {
    LayoutOneofUnion() {}
    int data_layout_;
    int filter_layout_;
  } layout_oneof_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct ::protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class AlgorithmProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:stream_executor.dnn.AlgorithmProto) */ {
 public:
  AlgorithmProto();
  virtual ~AlgorithmProto();

  AlgorithmProto(const AlgorithmProto& from);

  inline AlgorithmProto& operator=(const AlgorithmProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  AlgorithmProto(AlgorithmProto&& from) noexcept
    : AlgorithmProto() {
    *this = ::std::move(from);
  }

  inline AlgorithmProto& operator=(AlgorithmProto&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const AlgorithmProto& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const AlgorithmProto* internal_default_instance() {
    return reinterpret_cast<const AlgorithmProto*>(
               &_AlgorithmProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(AlgorithmProto* other);
  friend void swap(AlgorithmProto& a, AlgorithmProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline AlgorithmProto* New() const final {
    return CreateMaybeMessage<AlgorithmProto>(NULL);
  }

  AlgorithmProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<AlgorithmProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const AlgorithmProto& from);
  void MergeFrom(const AlgorithmProto& from);
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
  void InternalSwap(AlgorithmProto* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef AlgorithmProto_MathType MathType;
  static const MathType DEFAULT_MATH =
    AlgorithmProto_MathType_DEFAULT_MATH;
  static const MathType TENSOR_OP_MATH =
    AlgorithmProto_MathType_TENSOR_OP_MATH;
  static inline bool MathType_IsValid(int value) {
    return AlgorithmProto_MathType_IsValid(value);
  }
  static const MathType MathType_MIN =
    AlgorithmProto_MathType_MathType_MIN;
  static const MathType MathType_MAX =
    AlgorithmProto_MathType_MathType_MAX;
  static const int MathType_ARRAYSIZE =
    AlgorithmProto_MathType_MathType_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  MathType_descriptor() {
    return AlgorithmProto_MathType_descriptor();
  }
  static inline const ::std::string& MathType_Name(MathType value) {
    return AlgorithmProto_MathType_Name(value);
  }
  static inline bool MathType_Parse(const ::std::string& name,
      MathType* value) {
    return AlgorithmProto_MathType_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // int64 algo_id = 1;
  void clear_algo_id();
  static const int kAlgoIdFieldNumber = 1;
  ::google::protobuf::int64 algo_id() const;
  void set_algo_id(::google::protobuf::int64 value);

  // .stream_executor.dnn.AlgorithmProto.MathType math_type = 2;
  void clear_math_type();
  static const int kMathTypeFieldNumber = 2;
  ::stream_executor::dnn::AlgorithmProto_MathType math_type() const;
  void set_math_type(::stream_executor::dnn::AlgorithmProto_MathType value);

  // @@protoc_insertion_point(class_scope:stream_executor.dnn.AlgorithmProto)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int64 algo_id_;
  int math_type_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class ConvolutionDescriptorProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:stream_executor.dnn.ConvolutionDescriptorProto) */ {
 public:
  ConvolutionDescriptorProto();
  virtual ~ConvolutionDescriptorProto();

  ConvolutionDescriptorProto(const ConvolutionDescriptorProto& from);

  inline ConvolutionDescriptorProto& operator=(const ConvolutionDescriptorProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ConvolutionDescriptorProto(ConvolutionDescriptorProto&& from) noexcept
    : ConvolutionDescriptorProto() {
    *this = ::std::move(from);
  }

  inline ConvolutionDescriptorProto& operator=(ConvolutionDescriptorProto&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ConvolutionDescriptorProto& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ConvolutionDescriptorProto* internal_default_instance() {
    return reinterpret_cast<const ConvolutionDescriptorProto*>(
               &_ConvolutionDescriptorProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void Swap(ConvolutionDescriptorProto* other);
  friend void swap(ConvolutionDescriptorProto& a, ConvolutionDescriptorProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ConvolutionDescriptorProto* New() const final {
    return CreateMaybeMessage<ConvolutionDescriptorProto>(NULL);
  }

  ConvolutionDescriptorProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ConvolutionDescriptorProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ConvolutionDescriptorProto& from);
  void MergeFrom(const ConvolutionDescriptorProto& from);
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
  void InternalSwap(ConvolutionDescriptorProto* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int64 paddings = 1;
  int paddings_size() const;
  void clear_paddings();
  static const int kPaddingsFieldNumber = 1;
  ::google::protobuf::int64 paddings(int index) const;
  void set_paddings(int index, ::google::protobuf::int64 value);
  void add_paddings(::google::protobuf::int64 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
      paddings() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
      mutable_paddings();

  // repeated int64 strides = 2;
  int strides_size() const;
  void clear_strides();
  static const int kStridesFieldNumber = 2;
  ::google::protobuf::int64 strides(int index) const;
  void set_strides(int index, ::google::protobuf::int64 value);
  void add_strides(::google::protobuf::int64 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
      strides() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
      mutable_strides();

  // repeated int64 dilations = 3;
  int dilations_size() const;
  void clear_dilations();
  static const int kDilationsFieldNumber = 3;
  ::google::protobuf::int64 dilations(int index) const;
  void set_dilations(int index, ::google::protobuf::int64 value);
  void add_dilations(::google::protobuf::int64 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
      dilations() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
      mutable_dilations();

  // .stream_executor.dnn.DataType compute_mode = 4;
  void clear_compute_mode();
  static const int kComputeModeFieldNumber = 4;
  ::stream_executor::dnn::DataType compute_mode() const;
  void set_compute_mode(::stream_executor::dnn::DataType value);

  // int32 group_count = 5;
  void clear_group_count();
  static const int kGroupCountFieldNumber = 5;
  ::google::protobuf::int32 group_count() const;
  void set_group_count(::google::protobuf::int32 value);

  // .stream_executor.dnn.ConvolutionMode convolution_mode = 6;
  void clear_convolution_mode();
  static const int kConvolutionModeFieldNumber = 6;
  ::stream_executor::dnn::ConvolutionMode convolution_mode() const;
  void set_convolution_mode(::stream_executor::dnn::ConvolutionMode value);

  // @@protoc_insertion_point(class_scope:stream_executor.dnn.ConvolutionDescriptorProto)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 > paddings_;
  mutable int _paddings_cached_byte_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 > strides_;
  mutable int _strides_cached_byte_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int64 > dilations_;
  mutable int _dilations_cached_byte_size_;
  int compute_mode_;
  ::google::protobuf::int32 group_count_;
  int convolution_mode_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TensorDescriptorProto

// repeated int64 dimensions = 1;
inline int TensorDescriptorProto::dimensions_size() const {
  return dimensions_.size();
}
inline void TensorDescriptorProto::clear_dimensions() {
  dimensions_.Clear();
}
inline ::google::protobuf::int64 TensorDescriptorProto::dimensions(int index) const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.TensorDescriptorProto.dimensions)
  return dimensions_.Get(index);
}
inline void TensorDescriptorProto::set_dimensions(int index, ::google::protobuf::int64 value) {
  dimensions_.Set(index, value);
  // @@protoc_insertion_point(field_set:stream_executor.dnn.TensorDescriptorProto.dimensions)
}
inline void TensorDescriptorProto::add_dimensions(::google::protobuf::int64 value) {
  dimensions_.Add(value);
  // @@protoc_insertion_point(field_add:stream_executor.dnn.TensorDescriptorProto.dimensions)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
TensorDescriptorProto::dimensions() const {
  // @@protoc_insertion_point(field_list:stream_executor.dnn.TensorDescriptorProto.dimensions)
  return dimensions_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
TensorDescriptorProto::mutable_dimensions() {
  // @@protoc_insertion_point(field_mutable_list:stream_executor.dnn.TensorDescriptorProto.dimensions)
  return &dimensions_;
}

// .stream_executor.dnn.DataType data_type = 2;
inline void TensorDescriptorProto::clear_data_type() {
  data_type_ = 0;
}
inline ::stream_executor::dnn::DataType TensorDescriptorProto::data_type() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.TensorDescriptorProto.data_type)
  return static_cast< ::stream_executor::dnn::DataType >(data_type_);
}
inline void TensorDescriptorProto::set_data_type(::stream_executor::dnn::DataType value) {
  
  data_type_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.TensorDescriptorProto.data_type)
}

// .stream_executor.dnn.DataLayout data_layout = 3;
inline bool TensorDescriptorProto::has_data_layout() const {
  return layout_oneof_case() == kDataLayout;
}
inline void TensorDescriptorProto::set_has_data_layout() {
  _oneof_case_[0] = kDataLayout;
}
inline void TensorDescriptorProto::clear_data_layout() {
  if (has_data_layout()) {
    layout_oneof_.data_layout_ = 0;
    clear_has_layout_oneof();
  }
}
inline ::stream_executor::dnn::DataLayout TensorDescriptorProto::data_layout() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.TensorDescriptorProto.data_layout)
  if (has_data_layout()) {
    return static_cast< ::stream_executor::dnn::DataLayout >(layout_oneof_.data_layout_);
  }
  return static_cast< ::stream_executor::dnn::DataLayout >(0);
}
inline void TensorDescriptorProto::set_data_layout(::stream_executor::dnn::DataLayout value) {
  if (!has_data_layout()) {
    clear_layout_oneof();
    set_has_data_layout();
  }
  layout_oneof_.data_layout_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.TensorDescriptorProto.data_layout)
}

// .stream_executor.dnn.FilterLayout filter_layout = 4;
inline bool TensorDescriptorProto::has_filter_layout() const {
  return layout_oneof_case() == kFilterLayout;
}
inline void TensorDescriptorProto::set_has_filter_layout() {
  _oneof_case_[0] = kFilterLayout;
}
inline void TensorDescriptorProto::clear_filter_layout() {
  if (has_filter_layout()) {
    layout_oneof_.filter_layout_ = 0;
    clear_has_layout_oneof();
  }
}
inline ::stream_executor::dnn::FilterLayout TensorDescriptorProto::filter_layout() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.TensorDescriptorProto.filter_layout)
  if (has_filter_layout()) {
    return static_cast< ::stream_executor::dnn::FilterLayout >(layout_oneof_.filter_layout_);
  }
  return static_cast< ::stream_executor::dnn::FilterLayout >(0);
}
inline void TensorDescriptorProto::set_filter_layout(::stream_executor::dnn::FilterLayout value) {
  if (!has_filter_layout()) {
    clear_layout_oneof();
    set_has_filter_layout();
  }
  layout_oneof_.filter_layout_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.TensorDescriptorProto.filter_layout)
}

inline bool TensorDescriptorProto::has_layout_oneof() const {
  return layout_oneof_case() != LAYOUT_ONEOF_NOT_SET;
}
inline void TensorDescriptorProto::clear_has_layout_oneof() {
  _oneof_case_[0] = LAYOUT_ONEOF_NOT_SET;
}
inline TensorDescriptorProto::LayoutOneofCase TensorDescriptorProto::layout_oneof_case() const {
  return TensorDescriptorProto::LayoutOneofCase(_oneof_case_[0]);
}
// -------------------------------------------------------------------

// AlgorithmProto

// int64 algo_id = 1;
inline void AlgorithmProto::clear_algo_id() {
  algo_id_ = GOOGLE_LONGLONG(0);
}
inline ::google::protobuf::int64 AlgorithmProto::algo_id() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.AlgorithmProto.algo_id)
  return algo_id_;
}
inline void AlgorithmProto::set_algo_id(::google::protobuf::int64 value) {
  
  algo_id_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.AlgorithmProto.algo_id)
}

// .stream_executor.dnn.AlgorithmProto.MathType math_type = 2;
inline void AlgorithmProto::clear_math_type() {
  math_type_ = 0;
}
inline ::stream_executor::dnn::AlgorithmProto_MathType AlgorithmProto::math_type() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.AlgorithmProto.math_type)
  return static_cast< ::stream_executor::dnn::AlgorithmProto_MathType >(math_type_);
}
inline void AlgorithmProto::set_math_type(::stream_executor::dnn::AlgorithmProto_MathType value) {
  
  math_type_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.AlgorithmProto.math_type)
}

// -------------------------------------------------------------------

// ConvolutionDescriptorProto

// repeated int64 paddings = 1;
inline int ConvolutionDescriptorProto::paddings_size() const {
  return paddings_.size();
}
inline void ConvolutionDescriptorProto::clear_paddings() {
  paddings_.Clear();
}
inline ::google::protobuf::int64 ConvolutionDescriptorProto::paddings(int index) const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.paddings)
  return paddings_.Get(index);
}
inline void ConvolutionDescriptorProto::set_paddings(int index, ::google::protobuf::int64 value) {
  paddings_.Set(index, value);
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.paddings)
}
inline void ConvolutionDescriptorProto::add_paddings(::google::protobuf::int64 value) {
  paddings_.Add(value);
  // @@protoc_insertion_point(field_add:stream_executor.dnn.ConvolutionDescriptorProto.paddings)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
ConvolutionDescriptorProto::paddings() const {
  // @@protoc_insertion_point(field_list:stream_executor.dnn.ConvolutionDescriptorProto.paddings)
  return paddings_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
ConvolutionDescriptorProto::mutable_paddings() {
  // @@protoc_insertion_point(field_mutable_list:stream_executor.dnn.ConvolutionDescriptorProto.paddings)
  return &paddings_;
}

// repeated int64 strides = 2;
inline int ConvolutionDescriptorProto::strides_size() const {
  return strides_.size();
}
inline void ConvolutionDescriptorProto::clear_strides() {
  strides_.Clear();
}
inline ::google::protobuf::int64 ConvolutionDescriptorProto::strides(int index) const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.strides)
  return strides_.Get(index);
}
inline void ConvolutionDescriptorProto::set_strides(int index, ::google::protobuf::int64 value) {
  strides_.Set(index, value);
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.strides)
}
inline void ConvolutionDescriptorProto::add_strides(::google::protobuf::int64 value) {
  strides_.Add(value);
  // @@protoc_insertion_point(field_add:stream_executor.dnn.ConvolutionDescriptorProto.strides)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
ConvolutionDescriptorProto::strides() const {
  // @@protoc_insertion_point(field_list:stream_executor.dnn.ConvolutionDescriptorProto.strides)
  return strides_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
ConvolutionDescriptorProto::mutable_strides() {
  // @@protoc_insertion_point(field_mutable_list:stream_executor.dnn.ConvolutionDescriptorProto.strides)
  return &strides_;
}

// repeated int64 dilations = 3;
inline int ConvolutionDescriptorProto::dilations_size() const {
  return dilations_.size();
}
inline void ConvolutionDescriptorProto::clear_dilations() {
  dilations_.Clear();
}
inline ::google::protobuf::int64 ConvolutionDescriptorProto::dilations(int index) const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.dilations)
  return dilations_.Get(index);
}
inline void ConvolutionDescriptorProto::set_dilations(int index, ::google::protobuf::int64 value) {
  dilations_.Set(index, value);
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.dilations)
}
inline void ConvolutionDescriptorProto::add_dilations(::google::protobuf::int64 value) {
  dilations_.Add(value);
  // @@protoc_insertion_point(field_add:stream_executor.dnn.ConvolutionDescriptorProto.dilations)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int64 >&
ConvolutionDescriptorProto::dilations() const {
  // @@protoc_insertion_point(field_list:stream_executor.dnn.ConvolutionDescriptorProto.dilations)
  return dilations_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int64 >*
ConvolutionDescriptorProto::mutable_dilations() {
  // @@protoc_insertion_point(field_mutable_list:stream_executor.dnn.ConvolutionDescriptorProto.dilations)
  return &dilations_;
}

// .stream_executor.dnn.DataType compute_mode = 4;
inline void ConvolutionDescriptorProto::clear_compute_mode() {
  compute_mode_ = 0;
}
inline ::stream_executor::dnn::DataType ConvolutionDescriptorProto::compute_mode() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.compute_mode)
  return static_cast< ::stream_executor::dnn::DataType >(compute_mode_);
}
inline void ConvolutionDescriptorProto::set_compute_mode(::stream_executor::dnn::DataType value) {
  
  compute_mode_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.compute_mode)
}

// int32 group_count = 5;
inline void ConvolutionDescriptorProto::clear_group_count() {
  group_count_ = 0;
}
inline ::google::protobuf::int32 ConvolutionDescriptorProto::group_count() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.group_count)
  return group_count_;
}
inline void ConvolutionDescriptorProto::set_group_count(::google::protobuf::int32 value) {
  
  group_count_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.group_count)
}

// .stream_executor.dnn.ConvolutionMode convolution_mode = 6;
inline void ConvolutionDescriptorProto::clear_convolution_mode() {
  convolution_mode_ = 0;
}
inline ::stream_executor::dnn::ConvolutionMode ConvolutionDescriptorProto::convolution_mode() const {
  // @@protoc_insertion_point(field_get:stream_executor.dnn.ConvolutionDescriptorProto.convolution_mode)
  return static_cast< ::stream_executor::dnn::ConvolutionMode >(convolution_mode_);
}
inline void ConvolutionDescriptorProto::set_convolution_mode(::stream_executor::dnn::ConvolutionMode value) {
  
  convolution_mode_ = value;
  // @@protoc_insertion_point(field_set:stream_executor.dnn.ConvolutionDescriptorProto.convolution_mode)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace dnn
}  // namespace stream_executor

namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::stream_executor::dnn::AlgorithmProto_MathType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::AlgorithmProto_MathType>() {
  return ::stream_executor::dnn::AlgorithmProto_MathType_descriptor();
}
template <> struct is_proto_enum< ::stream_executor::dnn::DataType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::DataType>() {
  return ::stream_executor::dnn::DataType_descriptor();
}
template <> struct is_proto_enum< ::stream_executor::dnn::DataLayout> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::DataLayout>() {
  return ::stream_executor::dnn::DataLayout_descriptor();
}
template <> struct is_proto_enum< ::stream_executor::dnn::FilterLayout> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::FilterLayout>() {
  return ::stream_executor::dnn::FilterLayout_descriptor();
}
template <> struct is_proto_enum< ::stream_executor::dnn::ActivationMode> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::ActivationMode>() {
  return ::stream_executor::dnn::ActivationMode_descriptor();
}
template <> struct is_proto_enum< ::stream_executor::dnn::ConvolutionMode> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stream_executor::dnn::ConvolutionMode>() {
  return ::stream_executor::dnn::ConvolutionMode_descriptor();
}

}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_diplomacy_5ftensorflow_2fstream_5fexecutor_2fdnn_2eproto
