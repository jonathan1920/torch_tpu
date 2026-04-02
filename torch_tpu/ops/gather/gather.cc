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

#include "torch_tpu/ops/gather/gather.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::Status CheckGatherInputs(absl::Span<const int64_t> self, int64_t dim,
                               absl::Span<const int64_t> index,
                               bool sparse_grad) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error is caught by caller `Gather`
                 // function.
      sparse_grad == false, error::kUnimplemented)
      << "sparse_grad is not yet supported";

  const int64_t self_rank = self.size();
  const int64_t index_rank = index.size();

  // self and index should have the same rank, except when one of them is a
  // scalar and the other is a vector.
  if (self_rank == 0 || index_rank == 0) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=SafeWrapDim function (called before
                   // this one) catches this error first.
        dim == 0, error::kInvalidArgument)
        << "expected the dim argument to be 0 when either the input or the "
           "index tensors are scalars, got "
        << dim;
    TT_RET_CHECK(self_rank <= 1, error::kInvalidArgument)
        << "expected the input to be a scalar or a 1D tensor when the index "
           "tensor is a scalar, got "
        << self_rank << "D of shape " << ToString(self);
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=SafeWrapDim function (called before
                   // this one) catches 0-dimensional self tensor cases.
        index_rank <= 1, error::kInvalidArgument)
        << "expected the index tensor to be a scalar or a 1D tensor when the "
           "input tensor is a scalar, got "
        << index_rank << "D of shape " << ToString(index);
  } else {
    TT_RET_CHECK(self_rank == index_rank, error::kInvalidArgument)
        << "expected the input and the index tensor to have the same number of "
           "dimensions, got "
        << ToString(self) << " vs " << ToString(index);

    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=SafeWrapDim function (called before
                   // this one) catches this error first.
        dim >= 0 && dim < self_rank, error::kInvalidArgument)
        << "expected the dim argument to be in the range [0, " << self_rank
        << "), got " << dim;
  }

  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildGatherShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, bool sparse_grad,
    mlir::ElementType computation_element_type) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);

  ABSL_VLOG(2) << "BuildGatherShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", index_type: " << mlir::debugString(index_type)
               << ", sparse_grad: " << (sparse_grad ? "true" : "false");

  TT_RETURN_IF_ERROR(CheckGatherInputs(self_type.getShape(), dim,
                                       index_type.getShape(), sparse_grad));

  // self and index should have the same rank, except when one of them is a
  // scalar and the other is a vector.
  if (self_type.getRank() == 0 || index_type.getRank() == 0) {
    if (self_type.getRank() != index_type.getRank()) {
      self = mlir::stablehlo::Reshape(self, index_type.getShape());
    }
    return self;
  }

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_element_type);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = mlir::stablehlo::ConvertElementType(self, computation_type);
  }

  mlir::MlirBuilder& builder = self.getBuilder();
  Dimensions all_dimensions(self_type.getRank());
  absl::c_iota(all_dimensions, 0);

  Dimensions expanded_index_shape = CopyIntVector(index_type.getShape());
  expanded_index_shape.push_back(1);
  mlir::Type index_element_type = index_type.getElementType();

  // gather_indices has one more dimension than `index`, and contains
  // [i_1, i_2, ..., i_{dim-1}, index[i_1, i_2, ..., i_k], i_{dim+1}, ...,  i_k]
  // for each element [i_1, i_2, ..., i_k] from the index space of `index`.
  std::vector<mlir::MlirOp> parts;
  for (int d = 0; d < self_type.getRank(); ++d) {
    if (d == dim) {
      mlir::MlirOp expanded_index =
          mlir::stablehlo::Reshape(index, expanded_index_shape);
      parts.push_back(expanded_index);
    } else {
      mlir::MlirOp j_part = stablehlo::Iota(
          builder,
          makeTensorType(builder.getContext(), index_type.getShape(),
                         index_element_type),
          /*iota_dimension=*/d);
      j_part = mlir::stablehlo::Reshape(j_part, expanded_index_shape);
      parts.push_back(j_part);
    }
  }
  mlir::MlirOp gather_indices =
      stablehlo::Concatenate(builder, parts, /*dimension=*/self_type.getRank());

  stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &self.getContext(),
          /*offset_dims=*/{},
          /*collapsed_slice_dims=*/all_dimensions,
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/all_dimensions,
          /*index_vector_dim=*/index_type.getRank());
  ABSL_VLOG(2) << "BuildGatherShlo: GatherDimensionNumbers = "
               << mlir::debugString(gather_dimension_numbers);

  auto slice_sizes =
      std::vector<int64_t>(self_type.getRank(), 1);  // INT_VEC_OK
  auto result = stablehlo::Gather(self, gather_indices,
                                  gather_dimension_numbers, slice_sizes,
                                  /*indices_are_sorted=*/false);
  return mlir::stablehlo::ConvertElementType(result,
                                             self_type.getElementType());
}

}  // namespace torch_tpu
