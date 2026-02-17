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

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Returns a list of dimensions for each tensor in the input tensor list.
std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list);

// Builds element-wise binary StableHLO ops for each pair of input tensors.
//
// Casts inputs to `out_dtypes` and broadcasts them to a compatible shape
// before applying `op` element-wise.
template <typename StablehloOp>
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachShlo(
    absl::Span<mlir::MlirOp> self, absl::Span<mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes, StablehloOp op,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_other = other[i];

    current_self = CastIfNeeded(current_self, out_dtypes[i]).value();
    current_other = CastIfNeeded(current_other, out_dtypes[i]).value();

    std::array<mlir::MlirOp, 2> broadcasted_ops;
    TT_ASSIGN_OR_RETURN(broadcasted_ops,
                        ApplyBroadcastIfNeeded(current_self, current_other));
    current_self = broadcasted_ops[0];
    current_other = broadcasted_ops[1];

    mlir::MlirOp result = op(current_self, current_other);
    results.push_back(result);
  }
  return results;
}

// Converts a list of device buffers to a list of tensors.
std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers);

// Assigns the results from device buffers to the input tensor list for
// in-place operations.
void ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                           at::TensorList self);

// Returns a list of output dtypes for foreach operations.
// The overloads are for the variants of foreach operations.
std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self);
std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self,
                                               at::TensorList other);
std::vector<mlir::ElementType> GetOutputDtypes(at::TensorList self,
                                               const at::Scalar& scalar);
std::vector<mlir::ElementType> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);

// Checks if the output type is equal to the compute type. Raises an
// error if not.
void CheckScalarType(mlir::ElementType out_dtype,
                     mlir::ElementType compute_dtype,
                     at::ScalarType tensor_type, at::ScalarType scalar_type);
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FOREACH_UTILS_H_
