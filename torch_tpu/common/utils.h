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

#ifndef TORCH_TPU_UTILS_H_
#define TORCH_TPU_UTILS_H_

// Generic utilities for torch_tpu.

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "c10/util/ArrayRef.h"
#include "torch/headeronly/util/complex.h"

// 1 if this is a Google-internal version of torch_tpu. Otherwise 0.
#define TT_IS_INTERNAL_TORCH_TPU /* Comment for forcing a line break. */ \
  0

// TODO: b/442629517 - Passing by explicit reference is deprecated, but external
// builds use an older absl version that doesn't have implicit reference APIs.
#define TT_MUTEX_LOCK(lock, mu) absl::MutexLock lock(&(mu))
#define TT_READER_MUTEX_LOCK(lock, mu) absl::ReaderMutexLock lock(&(mu))

namespace torch_tpu {

// Returns a string representation of the given span. Requires the element type
// to be streamable.
template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
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
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << "(" << vec[i].first << ", " << vec[i].second << ")";
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

// Returns a string representation of the given value that can be losslessly
// converted back to the original value. Use this when computing the computation
// cache key for a double/complex value to avoid different values mapping to the
// same cache key.
[[nodiscard]] std::string LosslessToString(double value);
[[nodiscard]] std::string LosslessToString(const c10::complex<double>& value);

// Log a very long string as multiple lines. This is to work around ABSL_LOG's
// limitation on a maximum line length.
void LogLines(std::string_view s);

// The following hash function is suggested by Gemini and by a Google search.
template <typename T>
inline void HashCombine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_UTILS_H_
