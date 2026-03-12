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

#include "torch_tpu/common/aten_utils.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

std::string ToString(const at::ScalarType& scalar_type,
                     const std::string& name) {
  std::stringstream os;
  if (!name.empty()) {
    os << name << ": ";
  }
  os << "(type: " << static_cast<int32_t>(scalar_type) << ")";
  return os.str();
}

bool operator==(const HashableScalar& lhs, const HashableScalar& rhs) {
  // If the target types are different, they are not equal.
  if (lhs.scalar_type != rhs.scalar_type) return false;

  // If the target types are the same, then we check that the host types and
  // values are the same as well.
  switch (lhs.scalar.type()) {
    case c10::ScalarType::ComplexDouble: {
      return rhs.scalar.type() == c10::ScalarType::ComplexDouble &&
             lhs.scalar.toComplexDouble() == rhs.scalar.toComplexDouble();
    }
    case c10::ScalarType::Double:
      return rhs.scalar.type() == c10::ScalarType::Double &&
             lhs.scalar.toDouble() == rhs.scalar.toDouble();
    case c10::ScalarType::UInt64:
      return rhs.scalar.type() == c10::ScalarType::UInt64 &&
             lhs.scalar.toUInt64() == rhs.scalar.toUInt64();
    case c10::ScalarType::Long:
      return rhs.scalar.type() == c10::ScalarType::Long &&
             lhs.scalar.toLong() == rhs.scalar.toLong();
    case c10::ScalarType::Bool:
      return rhs.scalar.type() == c10::ScalarType::Bool &&
             lhs.scalar.toBool() == rhs.scalar.toBool();
    default:
      return false;
  }
}

}  // namespace torch_tpu
