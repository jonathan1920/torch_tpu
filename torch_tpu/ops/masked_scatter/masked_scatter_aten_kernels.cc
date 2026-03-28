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

#include "torch_tpu/ops/masked_scatter/masked_scatter_aten_kernels.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "ATen/ops/empty_like.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/scatter/scatter_aten_kernels.h"
#include "torch_tpu/ops/sort/sort_aten_kernels.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

namespace {

// Return the size of True elements in the mask by summing the mask
absl::StatusOr<at::Tensor> GetMaskedScatterSize(const at::Tensor& mask) {
  c10::ScalarType scalar_dtype = c10::ScalarType::Long;
  TT_ASSIGN_OR_RETURN(
      at::Tensor result,
      ApplySumReduction(mask, std::nullopt, ReductionMode::kDropDims,
                        scalar_dtype));
  return result;
}

absl::StatusOr<at::Tensor> MaskedScatterImpl(const at::Tensor& self,
                                             const at::Tensor& mask,
                                             const at::Tensor& source,
                                             int64_t num_scattered) {
  // Flatten the tensors at all dimensions
  at::Tensor self_flat = self.flatten();
  at::Tensor mask_flat = mask.flatten();
  at::Tensor source_flat = source.flatten();

  // Sort and slice the indices to get elements with true values in mask
  // For example:
  // Mask: [0, 1, 0, 1] -> Sort Desc -> [1, 1, 0, 0]
  // Indices: [1, 3, 0, 2] -> Slice(2) -> [1, 3]
  at::Tensor mask_flat_values = MakeEmptyTensor(
      mask_flat.sizes(), mask_flat.scalar_type(), mask_flat.device());
  at::Tensor mask_flat_indices =
      MakeEmptyTensor(mask_flat.sizes(), at::kLong, mask_flat.device());
  AtenSortValuesStable(mask_flat, /*stable_opt=*/true, /*dim=*/0,
                       /*descending=*/true, mask_flat_values,
                       mask_flat_indices);

  // Slice the indices to get elements with true values in mask. We get the
  // first `num_scattered` elements
  at::Tensor sliced_indices = mask_flat_indices.slice(0, 0, num_scattered, 1);
  at::Tensor source_values = source_flat.slice(0, 0, num_scattered, 1);

  at::Tensor result_flat = at::empty_like(self_flat);
  AtenScatterSrcOut(self_flat, /*dim=*/0, sliced_indices, source_values,
                    result_flat);

  return result_flat;
}
}  // namespace

at::Tensor& AtenMaskedScatter_(at::Tensor& self, const at::Tensor& mask,
                               const at::Tensor& source) {
  TT_KERNEL(OpName::kMaskedScatter_, _, (self, mask, source), {
    TT_ASSIGN_OR_THROW(const auto mask_dtype,
                       ConvertTo<mlir::ElementType>(mask.scalar_type()));
    TT_CHECK_THROW(mask.scalar_type() == c10::ScalarType::Bool,
                   error::kInvalidArgument)
        << "expected Boolean tensor for mask, got " << ToString(mask_dtype);

    TT_ASSIGN_OR_THROW(const auto self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto source_dtype,
                       ConvertTo<mlir::ElementType>(source.scalar_type()));
    TT_CHECK_THROW(self_dtype == source_dtype, error::kInvalidArgument)
        << "expected same dtype for self and source,"
        << " got self dtype " << ToString(self_dtype) << " and source dtype "
        << ToString(source_dtype);

    auto broadcasted = at::broadcast_tensors({mask, self});
    at::Tensor& mask_broadcasted = broadcasted[0];
    at::Tensor& self_broadcasted = broadcasted[1];

    // Get the size of the True elements in the mask and move it to CPU
    TT_ASSIGN_OR_THROW(at::Tensor size, GetMaskedScatterSize(mask_broadcasted));
    int64_t size_cpu = size.cpu().item<int64_t>();

    // If there are no True elements in the mask, return the original tensor
    if (size_cpu == 0) {
      return self;
    }

    TT_ASSIGN_OR_THROW(at::Tensor result_flat,
                       MaskedScatterImpl(self_broadcasted, mask_broadcasted,
                                         source, size_cpu));
    at::Tensor result = AtenView(result_flat, self.sym_sizes());
    AtenCopyFrom(result, self, /*non_blocking=*/true);
    return self;
  });
}
}  // namespace torch_tpu
