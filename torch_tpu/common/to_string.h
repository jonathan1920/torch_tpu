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
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// This library defines a ToString() function template that can convert an
// arbitrary type used in TorchTPU to a human-readable string useful for
// logging and error messages. Depending on the input type, ToString() may
// return either a std::string or a std::string_view.

// Converts a c10d::ReduceOp::RedOpType to a string.
[[nodiscard]] std::string ToString(c10d::ReduceOp::RedOpType reduce_op_type);

[[nodiscard]] std::string ToString(const at::Tensor& tensor,
                                   const std::string& name = "");
[[nodiscard]] std::string ToString(const at::Scalar& scalar,
                                   const std::string& name = "");

// Returns a string representation of the given MLIR type.
[[nodiscard]] std::string ToString(mlir::Type type);

// Returns a string representation of the given type as a PyTorch dtype name
// (e.g. "float32", "int8"). This is suitable for use in user messages.
[[nodiscard]] std::string_view ToString(at::ScalarType scalar_type);

// Returns a string representation of the given span. Requires the element type
// to be streamable.
template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> vec) {
  // TODO: switch to absl::StrCat() and absl::StrJoin().
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    // TODO: switch to ToString() for each element.
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T, size_t N>
[[nodiscard]] std::string ToString(const absl::InlinedVector<T, N>& vec) {
  return ToString(absl::MakeSpan(vec));
}

// Returns a string representation of the given span of pairs.
// Both element types in the pair must be streamable.
template <typename T1, typename T2>
[[nodiscard]] std::string ToString(absl::Span<const std::pair<T1, T2>> vec) {
  // TODO: switch to absl::StrCat() and absl::StrJoin().
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    // TODO: switch to ToString() for each element.
    const auto& [first, second] = vec[i];
    ss << "(" << first << ", " << second << ")";
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

// Returns a string representation of the given vector. Requires the element
// type to be streamable.
template <typename T>
[[nodiscard]] std::string ToString(const std::vector<T>& vec) {
  return ToString(absl::MakeSpan(vec));
}

// Returns a string representation of the given vector. Requires the element
// type to be streamable.
template <typename T>
[[nodiscard]] std::string ToString(mlir::ArrayRef<T> vec) {
  return ToString(absl::MakeSpan(vec));
}

template <typename T>
[[nodiscard]] std::string ToString(c10::ArrayRef<T> vec) {
  return ToString(absl::MakeSpan(vec));
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_TO_STRING_H_
