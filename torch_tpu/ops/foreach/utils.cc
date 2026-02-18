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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

namespace {
c10::ScalarType GetOutputDtypeFromTensorAndScalar(c10::ScalarType tensor_dtype,
                                                  const at::Scalar& scalar) {
  if ((c10::isFloatingType(tensor_dtype) && scalar.isFloatingPoint()) ||
      (c10::isIntegralType(tensor_dtype, /*includeBool=*/false) &&
       scalar.isIntegral(/*includeBool=*/true))) {
    // If both tensor and scalar are floating point or integral, then use the
    // tensor's dtype.
    return tensor_dtype;
  } else if (c10::isIntegralType(tensor_dtype, true) &&
             scalar.isFloatingPoint()) {
    // If tensor is integral and scalar is floating point, then use the
    // default floating point dtype.
    return c10::get_default_dtype_as_scalartype();
  } else {
    // If the tensor is boolean and the scalar is integer, just normally
    // promote the types to the scalar's integer type.
    return c10::promoteTypes(tensor_dtype, scalar.type());
  }
}
}  // namespace

std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers) {
  std::vector<at::Tensor> result;
  result.reserve(result_buffers.size());
  for (auto& result_buffer : result_buffers) {
    result.push_back(MakeTensor(std::move(result_buffer)));
  }
  return result;
}

void ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                           at::TensorList self) {
  for (int i = 0; i < result_buffers.size(); ++i) {
    TT_THROW_IF_ERROR(
        AssignBufferToAtTensor(std::move(result_buffers[i]), self[i]));
  }
}

std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list) {
  std::vector<absl::Span<const int64_t>> dims_list;
  dims_list.reserve(tensor_list.size());
  for (int i = 0; i < tensor_list.size(); ++i) {
    dims_list.push_back(tensor_list[i].sizes());
  }
  return dims_list;
}

std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    TT_ASSIGN_OR_THROW(mlir::ElementType output_element_type,
                       ConvertTo<mlir::ElementType>(self[i].scalar_type()));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self,
                                               at::TensorList other) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type =
        c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
    TT_ASSIGN_OR_THROW(mlir::ElementType output_element_type,
                       ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self,
                                               const at::Scalar& scalar) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i].scalar_type(), scalar);
    TT_ASSIGN_OR_THROW(mlir::ElementType output_element_type,
                       ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

std::vector<mlir::ElementType> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  std::vector<mlir::ElementType> out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i].scalar_type(), scalars[i]);
    TT_ASSIGN_OR_THROW(mlir::ElementType output_element_type,
                       ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

void CheckScalarType(mlir::ElementType out_dtype,
                     mlir::ElementType compute_dtype,
                     at::ScalarType tensor_type, at::ScalarType scalar_type) {
  TT_CHECK_THROW(out_dtype == compute_dtype, error::kInvalidArgument)
      << "expected the scalar dtype to be castable to the tensor dtype "
         "(e.g. bool to int or int to float), got "
      << ToString(scalar_type) << " and " << ToString(tensor_type);
}

}  // namespace torch_tpu
