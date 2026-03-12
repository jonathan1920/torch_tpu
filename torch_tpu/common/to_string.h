/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_COMMON_TO_STRING_H_
#define TORCH_TPU_COMMON_TO_STRING_H_

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// This library defines the ToString() function template and overloads that can
// convert an arbitrary type used in TorchTPU to a human-readable string useful
// for logging and error messages. Depending on the input type, ToString() may
// return either a std::string or a std::string_view.

// The primary template is for types where ToString(const T&) is not defined.
template <typename T>
std::string ToString(const T& x);

[[nodiscard]] std::string ToString(c10d::ReduceOp::RedOpType reduce_op_type);

[[nodiscard]] std::string ToString(const at::Tensor& tensor,
                                   const std::string& name = "");
[[nodiscard]] std::string ToString(const at::Scalar& scalar,
                                   const std::string& name = "");

[[nodiscard]] std::string ToString(mlir::Type type);

// Returns a string representation of the given type as a PyTorch dtype name
// (e.g. "float32", "int8"). This is suitable for use in user messages.
[[nodiscard]] std::string_view ToString(at::ScalarType scalar_type);

// Returns a string representation of the given span. Requires the element type
// to support ToString().
template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> span) {
  std::string result = "[";
  absl::StrAppend(
      &result, absl::StrJoin(span, ", ", [](std::string* out, const T& elem) {
        absl::StrAppend(out, ToString(elem));
      }));
  absl::StrAppend(&result, "]");
  return result;
}

template <typename T, size_t N>
[[nodiscard]] std::string ToString(const absl::InlinedVector<T, N>& vec) {
  return ToString(absl::MakeSpan(vec));
}

// Returns a string representation of the given pair.
// Both element types in the pair must support ToString().
template <typename T1, typename T2>
[[nodiscard]] std::string ToString(const std::pair<T1, T2>& pair) {
  return absl::StrCat("(", ToString(pair.first), ", ", ToString(pair.second),
                      ")");
}

// Returns a string representation of the given vector. Requires the element
// type to support ToString().
template <typename T>
[[nodiscard]] std::string ToString(const std::vector<T>& vec) {
  return ToString(absl::MakeSpan(vec));
}

// Returns a string representation of the given array. Requires the element
// type to support ToString().
template <typename T>
[[nodiscard]] std::string ToString(mlir::ArrayRef<T> vec) {
  return ToString(absl::MakeSpan(vec));
}
template <typename T>
[[nodiscard]] std::string ToString(c10::ArrayRef<T> vec) {
  return ToString(absl::MakeSpan(vec));
}

// Trait: does T support absl::StrCat()?
template <typename T, typename = void>
struct supports_absl_strcat : std::false_type {};
template <typename T>
struct supports_absl_strcat<
    T, std::void_t<decltype(absl::StrCat(std::declval<const T&>()))>>
    : std::true_type {};

// Trait: does T support operator<<()?
template <typename T, typename = void>
struct supports_ostream : std::false_type {};
template <typename T>
struct supports_ostream<T, std::void_t<decltype(std::declval<std::ostream&>()
                                                << std::declval<const T&>())>>
    : std::true_type {};

// Trait: does T support .ToString()?
template <typename T, typename = void>
struct supports_tostring_method : std::false_type {};
template <typename T>
struct supports_tostring_method<
    T, std::void_t<decltype(std::declval<const T&>().ToString())>>
    : std::true_type {};

template <typename T>
inline constexpr bool always_false_v = false;

// The primary ToString() template. It delegates to absl::StrCat(), the
// .ToString() method, or operator<< depending on which is supported by T.
template <typename T>
std::string ToString(const T& x) {
  if constexpr (supports_absl_strcat<T>::value) {
    // Prefer absl::StrCat() as it's generally more efficient.
    return absl::StrCat(x);
  } else if constexpr (supports_tostring_method<T>::value) {
    // Next, prefer .ToString() as it's generally more efficient than
    // operator<<.
    return x.ToString();
  } else if constexpr (supports_ostream<T>::value) {
    std::ostringstream os;
    os << x;
    return os.str();
  } else {
    static_assert(always_false_v<T>,
                  "ToString() requires the type to support absl::StrCat, "
                  "operator<<, or x.ToString().");
  }
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_TO_STRING_H_
