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

#ifndef TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_
#define TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_

// Utilities for computing fingerprints.

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "tsl/platform/fingerprint.h"

namespace torch_tpu {

// Represents the fingerprint of a chunk of data.
using FingerprintType = uint64_t;  // 8 bytes

template <typename T>
[[nodiscard]] FingerprintType Fingerprint(const T& t);

namespace internal {

// Helper for computing fingerprints of various types. We use a struct instead
// of a function to allow for partial specialization.
//
// The primary template is used for non-integral types.
template <typename T,
          bool kIsSmallIntegral =
              // is_integral_v does NOT include enums. This is intentional as
              // enum numerical values are not guaranteed to be stable.
          std::is_integral_v<T> && sizeof(T) <= sizeof(FingerprintType)>
struct Fingerprint64Impl {
  static_assert(!kIsSmallIntegral,
                "The primary template should be instantiated only for "
                "non-integral types.");

  // Returns the fingerprint of the given value.
  [[nodiscard]] static FingerprintType Compute(const T& value) {
    return tsl::Fingerprint64(value);
  }
};

// Specialization for OpName.
template <>
struct Fingerprint64Impl<OpName, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const OpName op_name) {
    // The string op names are unique and stable as they are used for
    // registering ops with PyTorch.
    return Fingerprint(ToString(op_name));
  }
};

// Specialization for mlir::ElementType.
template <>
struct Fingerprint64Impl<mlir::ElementType, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(
      const mlir::ElementType element_type) {
    // The short names for ElementType are unique and stable as they are used
    // for parameter cache keys.
    return Fingerprint(ToShortString(element_type));
  }
};

// Partial specialization for absl::Span.
template <typename T>
struct Fingerprint64Impl<absl::Span<T>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(absl::Span<T> span) {
    FingerprintType fp = span.size();
    for (const T& element : span) {
      fp = tsl::FingerprintCat64(fp, Fingerprint(element));
    }
    return fp;
  }
};

// Partial specialization for std::vector.
template <typename T>
struct Fingerprint64Impl<std::vector<T>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::vector<T>& span) {
    return Fingerprint(absl::MakeConstSpan(span));
  }
};

// Partial specialization for absl::InlinedVector.
template <typename T, size_t N>
struct Fingerprint64Impl<absl::InlinedVector<T, N>,
                         /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(
      const absl::InlinedVector<T, N>& span) {
    return Fingerprint(absl::MakeConstSpan(span));
  }
};

// Partial specialization for std::map.
template <typename K, typename V>
struct Fingerprint64Impl<std::map<K, V>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::map<K, V>& m) {
    FingerprintType fp = m.size();
    for (const auto& pair : m) {
      fp = tsl::FingerprintCat64(fp, Fingerprint(pair));
    }
    return fp;
  }
};

// Partial specialization for std::pair.
template <typename T, typename U>
struct Fingerprint64Impl<std::pair<T, U>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::pair<T, U>& pair) {
    return tsl::FingerprintCat64(Fingerprint(pair.first),
                                 Fingerprint(pair.second));
  }
};

// Partial specialization for integral types that are small enough to fit
// in a FingerprintType.
template <typename T>
struct Fingerprint64Impl<T, /*kIsSmallIntegral=*/true> {
  [[nodiscard]] static FingerprintType Compute(T value) {
    // Since casting T to FingerprintType doesn't lose information, we can
    // safely use it as the fingerprint. This is super fast.
    return static_cast<FingerprintType>(value);
  }
};

}  // namespace internal

// Returns the fingerprint of the given value.
template <typename T>
[[nodiscard]] FingerprintType Fingerprint(const T& t) {
  return internal::Fingerprint64Impl<T>::Compute(t);
}

// FingerprintCat(ts...) returns the fingerprint of the given values.
//
// The base cases.
[[nodiscard]] inline FingerprintType FingerprintCat() { return 0; }
template <typename T>
[[nodiscard]] FingerprintType FingerprintCat(const T& t) {
  return Fingerprint(t);
}
// The recursive case.
template <typename T, typename... Ts>
[[nodiscard]] FingerprintType FingerprintCat(const T& t, const Ts&... ts) {
  return tsl::FingerprintCat64(Fingerprint(t), FingerprintCat(ts...));
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_
