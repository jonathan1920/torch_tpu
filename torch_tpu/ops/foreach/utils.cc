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

#include "torch_tpu/ops/foreach/utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {
c10::ScalarType GetOutputDtypeFromTensorAndScalar(const at::Tensor& tensor,
                                                  const at::Scalar& scalar) {
  if ((IsFloatingPoint(tensor) && scalar.isFloatingPoint()) ||
      (IsInteger(tensor) && scalar.isIntegral(/*includeBool=*/true))) {
    // If both tensor and scalar are floating point or integral, then use the
    // tensor's dtype, unless the tensor is boolean (see below).
    return tensor.scalar_type();
  } else if (IsIntegral(tensor) && scalar.isFloatingPoint()) {
    // If tensor is integral and scalar is floating point, then use the
    // default floating point dtype.
    return c10::get_default_dtype_as_scalartype();
  } else {
    // If the tensor is boolean and the scalar is integer, just normally
    // promote the types to the scalar's integer type.
    return c10::promoteTypes(tensor.scalar_type(), scalar.type());
  }
}
}  // namespace

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&, mlir::MlirOp&)>
        tensor_transform,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_other = other[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_other,
                        CastIfNeeded(current_other, out_dtypes[i]));

    std::array<mlir::MlirOp, 2> broadcasted_ops;
    TT_ASSIGN_OR_RETURN(broadcasted_ops,
                        ApplyBroadcastIfNeeded(current_self, current_other));
    current_self = broadcasted_ops[0];
    current_other = broadcasted_ops[1];

    mlir::MlirOp result = tensor_transform(current_self, current_other);
    results.push_back(result);
  }
  return results;
}

std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers) {
  std::vector<at::Tensor> result;
  result.reserve(result_buffers.size());
  for (auto& result_buffer : result_buffers) {
    result.push_back(MakeTensor(std::move(result_buffer)));
  }
  return result;
}

absl::Status ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                                   at::TensorList self) {
  for (int i = 0; i < result_buffers.size(); ++i) {
    TT_RETURN_IF_ERROR(
        AssignBufferToAtTensor(std::move(result_buffers[i]), self[i]));
  }
  return absl::OkStatus();
}

std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list) {
  std::vector<absl::Span<const int64_t>> dims_list;
  dims_list.reserve(tensor_list.size());
  for (int i = 0; i < tensor_list.size(); ++i) {
    dims_list.push_back(tensor_list[i].sizes());
  }
  return dims_list;
}

absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    TT_ASSIGN_OR_RETURN(auto output_element_type,
                        ConvertTo<mlir::ElementType>(self[i].scalar_type()));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, at::TensorList other) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, const at::Scalar& scalar) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalar);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<std::vector<mlir::ElementType>> GetFloatingOutputDtypes(
    at::TensorList self) {
  const at::ScalarType default_dtype = c10::get_default_dtype_as_scalartype();
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    c10::ScalarType tensor_type;
    if (IsIntegral(self[i])) {
      tensor_type = default_dtype;
    } else {
      tensor_type = self[i].scalar_type();
    }
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(tensor_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalars[i]);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::Status CheckScalarType(mlir::ElementType out_dtype,
                             mlir::ElementType compute_dtype,
                             at::ScalarType tensor_type,
                             at::ScalarType scalar_type) {
  TT_RET_CHECK(out_dtype == compute_dtype, error::kInvalidArgument)
      << "expected the scalar dtype to be castable to the tensor dtype "
         "(e.g. bool to int or int to float), got "
      << ToString(scalar_type) << " and " << ToString(tensor_type);
  return absl::OkStatus();
}

absl::Status EnsureNotIntegral(at::TensorList self) {
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RET_CHECK(!IsIntegral(self[i]), error::kInvalidArgument)
        << "expected input tensor dtype to be non-integral, got "
        << ToString(self[i].scalar_type());
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
