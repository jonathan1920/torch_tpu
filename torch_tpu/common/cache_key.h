/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_COMMON_CACHE_KEY_H_
#define TORCH_TPU_COMMON_CACHE_KEY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Dimname.h"
#include "c10/core/Device.h"
#include "c10/core/Layout.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Optional.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/xla_data.pb.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {

namespace internal {

// Formatters for different types of parameter values.
// For each given value type, they must guarantee that different values
// produce different strings.
absl::StatusOr<std::string> FormatParamCacheKey(at::Scalar value);
[[nodiscard]] std::string FormatParamCacheKey(const c10::SymInt& value);
[[nodiscard]] std::string FormatParamCacheKey(c10::SymIntArrayRef value);
[[nodiscard]] inline std::string FormatParamCacheKey(
    const at::ScalarType value) {
  return c10::toString(value);
}
[[nodiscard]] inline std::string FormatParamCacheKey(
    const at::Generator& value) {
  // We don't support random generator parameters yet.
  return "";
}
[[nodiscard]] inline std::string FormatParamCacheKey(const at::Dimname value) {
  return std::string(value.symbol().toQualString());
}
[[nodiscard]] std::string FormatParamCacheKey(c10d::ReduceOp value);
[[nodiscard]] inline std::string FormatParamCacheKey(
    const mlir::ElementType value) {
  return std::string(ToShortName(value));
}
[[nodiscard]] inline std::string FormatParamCacheKey(const double value) {
  return LosslessToString(value);
}
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] inline std::string FormatParamCacheKey(const T value) {
  return absl::StrCat(value);
}
[[nodiscard]] inline std::string FormatParamCacheKey(const bool value) {
  return value ? "true" : "false";
}
[[nodiscard]] std::string FormatParamCacheKey(std::string_view value);
[[nodiscard]] inline std::string FormatParamCacheKey(const char* const value) {
  return FormatParamCacheKey(std::string_view(value));
}
[[nodiscard]] inline std::string FormatParamCacheKey(const std::string& value) {
  return FormatParamCacheKey(std::string_view(value));
}
[[nodiscard]] std::string FormatParamCacheKey(absl::Span<const int64_t> value);
[[nodiscard]] std::string FormatParamCacheKey(at::Layout value);
[[nodiscard]] std::string FormatParamCacheKey(at::MemoryFormat value);
[[nodiscard]] std::string FormatParamCacheKey(at::Device value);
[[nodiscard]] std::string FormatParamCacheKey(const at::ITensorListRef& value);

// To guarantee that the correct overload of FormatParamCacheKey() is found,
// overloads that invoke other FormatParamCacheKey() overloads should be
// declared first and their definitions should be put after all the other
// overloads.
template <typename T>
absl::StatusOr<std::string> FormatParamCacheKey(const std::optional<T>& value);
template <typename T>
[[nodiscard]] std::string FormatParamCacheKey(c10::OptionalArrayRef<T> value);
template <typename T, std::size_t N>
[[nodiscard]] std::string FormatParamCacheKey(const std::array<T, N>& value);
template <typename T>
absl::StatusOr<std::string> FormatParamCacheKey(const std::vector<T>& value);

template <typename T>
absl::StatusOr<std::string> FormatParamCacheKey(const std::optional<T>& value) {
  // Rely on ADL to find the appropriate FormatParamCacheKey() overload for
  // the given value type.
  return value.has_value() ? FormatParamCacheKey(value.value()) : "";
}
template <typename T>
std::string FormatParamCacheKey(absl::Span<const T> value) {
  return absl::StrCat(
      "[",
      absl::StrJoin(
          value, ",",
          [](std::string* out, const T& elem) {
            absl::StrAppend(
                // Rely on ADL to find the appropriate FormatParamCacheKey()
                // overload for the given element type.
                out, FormatParamCacheKey(elem));
          }),
      "]");
}
template <typename T>
std::string FormatParamCacheKey(const c10::OptionalArrayRef<T> value) {
  if (!value.has_value()) {
    return "";
  }
  return FormatParamCacheKey(absl::MakeConstSpan(*value));
}
template <typename T, std::size_t N>
std::string FormatParamCacheKey(const std::array<T, N>& value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}
template <typename T>
absl::StatusOr<std::string> FormatParamCacheKey(const std::vector<T>& value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::AllreduceOptions& value) {
  return FormatParamCacheKey(value.reduceOp);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::ReduceScatterOptions& value) {
  return FormatParamCacheKey(value.reduceOp);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::BroadcastOptions& value) {
  return FormatParamCacheKey(value.rootRank);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::ScatterOptions& value) {
  return FormatParamCacheKey(value.rootRank);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::AllgatherOptions& value) {
  // AllgatherOptions has no members that affect compilation.
  return "";
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::AllToAllOptions& value) {
  // AllToAllOptions has no members that affect compilation.
  return "";
}

}  // namespace internal

// When defining an op, any parameters that can change the compilation of the
// op without changing the shape or dtype of the inputs or outputs should be
// added to this map.
// For example, relu() does not require any parameters, but gelu() takes an
// "approximate" parameter; two gelu() calls are different if this value is
// different.
// So relu() should use {}, while gelu() should use {"approximate",
// std::string(approximate)};
//
// We chose std::map as the container type because we need the keys to be
// sorted for deterministic cache keys.
//
// We mark the class as nodiscard so that if a function returns this type, we
// will get a compiler error if the return value is ignored. This is to avoid
// wasting time computing cache keys that are not used.
class [[nodiscard]] OpParamCacheKeys {
 private:
  using Map = std::map<std::string, std::string>;

 public:
  class Builder;

  // Implement the read-only container concept.
  using value_type = Map::value_type;
  using const_iterator = Map::const_iterator;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using const_reference = Map::const_reference;
  using const_pointer = Map::const_pointer;
  using const_reverse_iterator = Map::const_reverse_iterator;

  // Makes an empty OpParamCacheKeys.
  OpParamCacheKeys() = default;

  // Make OpParamCacheKeys move-only, as copying may be expensive.
  OpParamCacheKeys(const OpParamCacheKeys&) = delete;
  OpParamCacheKeys& operator=(const OpParamCacheKeys&) = delete;
  OpParamCacheKeys(OpParamCacheKeys&&) = default;
  OpParamCacheKeys& operator=(OpParamCacheKeys&&) = default;

  // Returns a copy of the OpParamCacheKeys.
  OpParamCacheKeys Clone() const { return OpParamCacheKeys(name_to_value_); }

  // Shorthand for making a builder with the given parameter name and
  // value.
  template <typename T>
  [[nodiscard]] static Builder SetParam(std::string_view name, const T& value);

  [[nodiscard]] const_iterator begin() const { return name_to_value_.cbegin(); }
  [[nodiscard]] const_iterator end() const { return name_to_value_.cend(); }

  [[nodiscard]] size_type size() const { return name_to_value_.size(); }
  [[nodiscard]] bool empty() const { return name_to_value_.empty(); }

 private:
  // Makes an OpParamCacheKeys with the given name-to-string-key map.
  explicit OpParamCacheKeys(Map name_to_value)
      : name_to_value_(std::move(name_to_value)) {}

  Map name_to_value_;
};

// Builder class for OpParamCacheKeys. Supports method-chaining syntax:
//   TT_ASSIGN_OR_RETURN(auto params,
//                       *OpParamCacheKeys::SetParam(...)
//                           .SetParam(...));
class OpParamCacheKeys::Builder {
 public:
  // Creates an empty builder that doesn't contain any parameters.
  Builder() = default;

  // This class is move-only.
  Builder(Builder&& other) = default;
  Builder& operator=(Builder&& other) = default;
  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  // Creates a builder that contains the given parameters.
  explicit Builder(OpParamCacheKeys op_param_cache_keys)
      : name_to_value_(std::move(op_param_cache_keys).name_to_value_) {}

  // If the builder is in an error state, does nothing. Otherwise, formats
  // the value as a string and:
  // 1. If the string is not empty, adds the name-string-value pair to the
  //    builder.
  // 2. If the string is empty, remove the parameter from the builder.
  // If formatting fails, the builder enters an error state.
  //
  // The formatting is done by calling FormatParamCacheKey() on the value.
  // To make this API extensible, we rely on Argument Dependent Lookup (ADL)
  // to find the appropriate FormatParamCacheKey() overload for the given
  // value type. In general, the overload for type T should be defined in
  // the same file where T is defined and in the same namespace. When that's
  // not possible (e.g. T is a c10 type, which we don't own), define the
  // overload in the torch_tpu::internal namespace in this file.
  template <typename T>
  Builder& SetParam(std::string_view name, const T& value) {
    if (!first_error_.ok()) {
      return *this;
    }

    using internal::FormatParamCacheKey;

    // Rely on ADL to find the appropriate FormatParamCacheKey() overload for
    // the given value type.
    absl::StatusOr<std::string> str_or = FormatParamCacheKey(value);
    if (str_or.ok()) {
      std::string str = std::move(str_or).value();
      if (str.empty()) {
        name_to_value_.erase(std::string(name));
      } else {
        name_to_value_[std::string(name)] = std::move(str);
      }
    } else {
      first_error_ = std::move(str_or).status();
    }
    return *this;
  }

  // If the builder is in an error state, returns the first error encountered.
  // Otherwise returns the cache keys accumulated so far.
  //
  // After this method is called, the builder is put into the default
  // state.
  absl::StatusOr<OpParamCacheKeys> operator*();

  // Returns the current error state of the builder.
  const absl::Status& status() const { return first_error_; }

 private:
  // The first error encountered during the construction of the parameter
  // cache keys, or OK if no errors were encountered.
  absl::Status first_error_ = absl::OkStatus();
  Map name_to_value_;
};

template <typename T>
[[nodiscard]] OpParamCacheKeys::Builder OpParamCacheKeys::SetParam(
    std::string_view name, const T& value) {
  return std::move(Builder().SetParam(name, value));
}

namespace internal {

// Adds one non-at::Tensor argument to the builder.
template <typename Arg>
inline absl::Status SetParamCacheKey(OpParamCacheKeys::Builder& builder,
                                     const std::string_view arg_name,
                                     const Arg& arg) {
  return builder.SetParam(std::string(arg_name), arg).status();
}

// These overloads ensure that we don't add at::Tensor arguments to the
// OpParamCacheKeys (as their dtypes and shapes are already encoded in the
// cache key automatically.).
inline absl::Status SetParamCacheKey(OpParamCacheKeys::Builder&,
                                     const std::string_view,
                                     const at::Tensor& arg) {
  return absl::OkStatus();
}

inline absl::Status SetParamCacheKey(OpParamCacheKeys::Builder&,
                                     const std::string_view,
                                     const at::ITensorListRef& arg) {
  return absl::OkStatus();
}

inline absl::Status SetParamCacheKey(OpParamCacheKeys::Builder&,
                                     const std::string_view,
                                     const std::vector<at::Tensor>& arg) {
  return absl::OkStatus();
}

inline absl::Status SetParamCacheKey(
    OpParamCacheKeys::Builder&, const std::string_view,
    const std::vector<std::vector<at::Tensor>>& arg) {
  return absl::OkStatus();
}

inline absl::Status SetParamCacheKey(
    OpParamCacheKeys::Builder&, const std::string_view,
    const c10::List<c10::optional<at::Tensor>>& arg) {
  return absl::OkStatus();
}

inline absl::Status SetParamCacheKey(OpParamCacheKeys::Builder& builder,
                                     const std::string_view arg_name,
                                     const c10::optional<at::Tensor>& arg) {
  // Encode arg_name and has_value to avoid a key clash for an op with multiple
  // optional Tensor arguments, for example, op(None, t1) and op(t1, None).
  return SetParamCacheKey(builder, arg_name, arg.has_value());
}

// Returns the name of the next argument in the given substring of a
// comma-separated list of argument names, e.g. "a, b, c".
// It also updates the substring to exclude the returned argument name.
//
// For example, if args_str is "a, b, c", the first call to this function
// will return "a" and update args_str to "b, c". The second call will return
// "b" and update args_str to "c". The third call will return "c" and update
// args_str to "".
[[nodiscard]] std::string_view ParseNextArgName(std::string_view& args_str);

// MakeOpParamCacheKeys() creates an OpParamCacheKeys from the given arguments
// for an op kernel. This function skips all at::Tensor-typed arguments in the
// list, as their dtypes and shapes are already encoded in the cache key
// automatically.
//
// Arguments:
//   args_str: A comma-separated list of argument names, e.g. "a, b, c".
//   args: The arguments. The number of arguments must match the number of
//       argument names in args_str.

// The base case: no arguments.
inline absl::Status MakeOpParamCacheKeysImpl(OpParamCacheKeys::Builder& builder,
                                             const std::string_view args_str) {
  return absl::OkStatus();
}

// The recursive case: at least one argument.
template <typename Arg, typename... Args>
inline absl::Status MakeOpParamCacheKeysImpl(OpParamCacheKeys::Builder& builder,
                                             std::string_view args_str,
                                             const Arg& arg,
                                             const Args&... args) {
  std::string_view arg_name = ParseNextArgName(args_str);
  TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=currently all ops can create
                       // cache keys successfully.
      SetParamCacheKey(builder, arg_name, arg));
  return MakeOpParamCacheKeysImpl(builder, args_str, args...);
}

template <typename... Args>
absl::StatusOr<OpParamCacheKeys> MakeOpParamCacheKeys(
    const std::string_view args_str, const Args&... args) {
  OpParamCacheKeys::Builder builder;
  TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=currently all ops can create
                       // cache keys successfully.
      MakeOpParamCacheKeysImpl(builder, args_str, args...));
  return *std::move(builder);
}

}  // namespace internal

// Give the shapeless and dimensions keys separate types to avoid accidentally
// using the wrong key.
struct ShapelessKey {
  struct Hash {
    [[nodiscard]] inline size_t operator()(const ShapelessKey key) const {
      return key.key;
    }
  };

  [[nodiscard]] bool operator==(const ShapelessKey rhs) const {
    return key == rhs.key;
  }

  FingerprintType key;
};
struct DimensionsKey {
  explicit DimensionsKey(absl::Span<const int64_t> dimensions,
                         bool is_shape_dynamic = false) {
    key = FingerprintCat(dimensions, is_shape_dynamic);
  }

  [[nodiscard]] bool operator==(const DimensionsKey rhs) const {
    return key == rhs.key;
  }

  FingerprintType key;
};

// A CompilationCacheKey is used to identify a compilation in the compilation
// cache. Depending on the hash/eq functions used, it may either uniquely
// identify a cached executable, or non-uniquely identify a set of executables
// that share some property.
struct CompilationCacheKey {
  struct Hash {
    [[nodiscard]] inline size_t operator()(CompilationCacheKey key) const {
      return FingerprintCat(key.shapeless_key.key, key.dimensions_key.key);
    }
  };

  // Compares two CompilationCacheKeys.
  [[nodiscard]] friend bool operator==(CompilationCacheKey lhs,
                                       CompilationCacheKey rhs) {
    return lhs.shapeless_key == rhs.shapeless_key &&
           lhs.dimensions_key == rhs.dimensions_key;
  }

  // Returns a compact string representation of the key, suitable for use in a
  // file name. The format is "<shapeless_key>_<dimensions_key>" where each
  // key part is formatted as a hexadecimal string of 16 digits.
  [[nodiscard]] std::string CompactFormat() const;

  // The fingerprint of all graph properties except for dimension sizes and
  // dynamic bounds.
  ShapelessKey shapeless_key;
  // The fingerprint of only the dimension sizes and dynamic bounds.
  DimensionsKey dimensions_key;
};

static_assert(
    sizeof(CompilationCacheKey) <= 3 * sizeof(void*),
    "Per C++ Core Guidelines, structs whose size is at most 3 pointers are "
    "more efficient to pass by value than by reference. We pass "
    "CompilationCacheKey by value today. We need to revisit the decision if "
    "its size grows.");

std::ostream& operator<<(std::ostream& os, CompilationCacheKey key);
std::string ToString(CompilationCacheKey key);

// Bounds of a dimension. If the lower and upper bounds are the same, the
// dimension is static and not dynamic. If *all* dimensions are static, the
// graph is static.
//
// A valid DimensionBounds has 0 <= lower <= upper.
struct DimensionBounds {
  int64_t lower = -1;
  int64_t upper = -1;
};

// The metadata necessary to compare two graphs when they have a shape-dynamic
// hash collision and determine if they are compatible in terms of bounded
// dynamism.
struct ShapeDynamismMetadata {
  // The lower and upper bounds of each dimension in the graph's inputs.
  std::vector<DimensionBounds> input_dimension_bounds;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CACHE_KEY_H_
