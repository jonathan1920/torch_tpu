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

#ifndef TORCH_TPU_OPS_FOREACH_UTILS_H_
#define TORCH_TPU_OPS_FOREACH_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/eager/device_buffer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds element-wise binary StableHLO ops for each pair of input tensors.
//
// Casts inputs to `out_dtypes` and broadcasts them to a compatible shape
// before applying `op` element-wise.
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&, mlir::MlirOp&)>
        tensor_transform,
    mlir::MlirBuilder& builder);

// Converts a list of device buffers to a list of tensors.
std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers);

// Assigns the results from device buffers to the input tensor list for
// in-place operations.
absl::Status ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                                   at::TensorList self);

// Returns a list of dimensions for each tensor in the input tensor list.
std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list);

// Returns a list of output dtypes for foreach operations.
// The overloads are for the variants of foreach operations.
absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self);
absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, at::TensorList other);
absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, const at::Scalar& scalar);
absl::StatusOr<std::vector<mlir::ElementType>> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);

// Returns a list of floating point output dtypes for foreach operations.
// Converts integral dtypes to `default_dtype`.
absl::StatusOr<std::vector<mlir::ElementType>> GetFloatingOutputDtypes(
    at::TensorList self);

// Checks if the output type is equal to the compute type. Raises an
// error if not.
absl::Status CheckScalarType(mlir::ElementType out_dtype,
                             mlir::ElementType compute_dtype,
                             at::ScalarType tensor_type,
                             at::ScalarType scalar_type);

// Checks if the input tensor list contains integral types. Raises an
// error if so.
absl::Status EnsureNotIntegral(at::TensorList self);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FOREACH_UTILS_H_
