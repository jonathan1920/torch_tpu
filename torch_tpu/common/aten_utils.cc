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

#include "absl/strings/str_join.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

std::string ToString(const at::Tensor& tensor, const std::string& name) {
  std::stringstream ss;
  if (!name.empty()) {
    ss << name << ": ";
  }
  if (!tensor.defined()) {
    ss << "undefined";
    return ss.str();
  }
  ss << "shape=[" << absl::StrJoin(tensor.sizes(), ",") << "], ";
  ss << "strides=[" << absl::StrJoin(tensor.strides(), ",") << "], ";
  ss << "numel=" << tensor.numel() << ", ";
  ss << "dtype=" << tensor.scalar_type() << ", ";
  ss << "device=" << tensor.device() << ", ";
  ss << "is_contiguous=" << tensor.is_contiguous() << ", ";
  ss << "is_cpu=" << tensor.is_cpu() << ", ";
  ss << "is_cuda=" << tensor.is_cuda() << ", ";
  ss << "is_meta=" << tensor.is_meta() << ", ";
  ss << "storage_offset=" << tensor.storage_offset() << ", ";

  if (tensor.defined() && tensor.storage().unsafeGetStorageImpl() != nullptr) {
    ss << "storage_use_count=" << tensor.storage().use_count() << ", ";
    if (tensor.storage().data_ptr().get_context() != nullptr) {
      // FIXME: The following code is commented out because it causes ASAN
      // violations on ops_test.py.
      //
      // DeviceBufferRef* buffer_ref = static_cast<DeviceBufferRef*>(
      //     tensor.storage().data_ptr().get_context());
      // if (buffer_ref != nullptr && !buffer_ref->dimensions.empty()) {
      //   ss << "buffer_ref_dims= not null [" << buffer_ref->dimensions.size()
      //      << "],\n";
      // } else {
      //   ss << "buffer_ref_dims=null";
      // }
    } else {
      ss << "context=null (but storage is defined)";
    }
  } else {
    ss << "storage_is_null(undefined)";
  }
  return ss.str();
}

std::string ToString(const at::Scalar& scalar, const std::string& name) {
  std::stringstream os;
  if (!name.empty()) {
    os << name << ": ";
  }
  if (scalar.isFloatingPoint()) {
    os << scalar.toDouble();
  } else if (scalar.isIntegral(/*include_bool=*/false)) {
    os << scalar.toLong();
  } else if (scalar.isBoolean()) {
    os << (scalar.toBool() ? 1 : 0);
  } else if (scalar.isComplex()) {
    os << scalar.toComplexDouble();
  } else {
    os << "(type: " << static_cast<int32_t>(scalar.type()) << ", value: ?)";
  }
  return os.str();
}

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
