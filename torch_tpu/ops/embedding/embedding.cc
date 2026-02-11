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

#include "torch_tpu/ops/embedding/embedding.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {
absl::StatusOr<mlir::MlirOp> BuildRenormRow(mlir::MlirBuilder& builder,
                                            mlir::MlirOp rows, double max_norm,
                                            double norm_type) {
  // rows shape is [..., emb_dim].
  // Could be [N, emb_dim] or [N, M, emb_dim] depending on indices shape.
  auto rows_shape = GetTensorTypeOrDie(rows).getShape();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(rows));
  int64_t embedding_dim_idx = rows_shape.size() - 1;

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp norms,
      BuildPNormShlo(/*input_op=*/rows, /*ord=*/norm_type,
                     /*reduce_dims=*/{embedding_dim_idx},
                     /*reduction_mode=*/ReductionMode::kKeepDims,
                     /*out_type=*/out_type));

  // Replicates PyTorch's renorm logic:
  // ```
  // if (norm > max_norm) {
  //   auto scale = max_norm / (norm + 1e-7);
  //   row *= scale;
  // }
  // ```

  Dimensions broadcast_dims_vec(rows_shape.size());
  absl::c_iota(broadcast_dims_vec, 0);

  // BroadcastInDim handles broadcasting the '1' in the last dimension
  // to match the embedding dimension of 'rows'.
  mlir::MlirOp norms_broadcasted = mlir::stablehlo::BroadcastInDim(
      rows.getType(), norms, broadcast_dims_vec);

  mlir::MlirOp max_norm_const = MakeConstantLike(rows, max_norm);
  mlir::MlirOp tiny_number = MakeConstantLike(rows, 1e-7);

  mlir::MlirOp norm_plus_tiny =
      mlir::stablehlo::Add(norms_broadcasted, tiny_number);
  mlir::MlirOp scale = mlir::stablehlo::Div(max_norm_const, norm_plus_tiny);
  mlir::MlirOp row_with_scale = mlir::stablehlo::Mul(rows, scale);

  mlir::MlirOp norm_gt =
      mlir::stablehlo::Compare(norms_broadcasted, max_norm_const,
                               mlir::stablehlo::ComparisonDirection::GT);

  mlir::MlirOp renorm_row =
      mlir::stablehlo::Select(norm_gt, row_with_scale, rows);

  return renorm_row;
}
}  // namespace
absl::StatusOr<mlir::MlirOp> BuildEmbeddingRenormShlo(mlir::MlirOp weight,
                                                      mlir::MlirOp indices,
                                                      double max_norm,
                                                      double norm_type) {
  // self if the embedding weight is shape [num_emb, emb_dim]
  const mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);

  mlir::MlirBuilder& builder = weight.getBuilder();

  int64_t embedding_dim = weight_type.getDimSize(1);

  mlir::RankedTensorType indices_type = GetTensorTypeOrDie(indices);

  if (indices_type.getRank() == 0) {
    // Special case: if indices is a scalar, then embedding just becomes a
    // dynamic slice.
    mlir::MlirOp zero_scalar =
        MakeScalarConstant(builder, 0, mlir::ElementType::I64);
    mlir::MlirOp start_indices[2] = {indices, zero_scalar};
    int64_t slice_sizes[2] = {1, embedding_dim};
    mlir::MlirOp row =
        mlir::stablehlo::DynamicSlice(weight, start_indices, slice_sizes);

    TT_ASSIGN_OR_RETURN(mlir::MlirOp renorm_row,
                        BuildRenormRow(builder, row, max_norm, norm_type));
    // Squeeze the 1 dim to get an [N] slice.
    return mlir::stablehlo::Reshape(renorm_row, {embedding_dim});
  } else {
    // Otherwise, need to do a gather.
    mlir::MLIRContext& ctx = weight.getContext();
    llvm::ArrayRef<int64_t> operand_batching_dims = {};
    llvm::ArrayRef<int64_t> start_index_batching_dims = {};
    int64_t index_vector_dim = indices_type.getRank();
    // We look up from column 0 of weight (i.e. by row)
    int64_t start_index_map[1] = {0};
    // We collect row slices of shape [1, emb_dim] out of weight for each start
    // index...
    Dimensions slice_sizes({1, embedding_dim});
    // ...then we collapse the 0th dim of each slice, so each lookup index
    // returns a slice of shape [emb_dim].
    int64_t collapsed_slice_dims[1] = {0};
    // We append the [N] dimension to the shape of the indices tensor,
    // to get shape [I0, I1, ..., IM-1, N].
    int64_t offset_dims[1] = {indices_type.getRank()};

    ABSL_VLOG(3) << "BuildEmbeddingShloGather: weight: "
                 << mlir::debugString(weight.getValue())
                 << "\n indices: " << mlir::debugString(indices.getValue())
                 << "\n offset_dims: "
                 << ToString(absl::MakeConstSpan(offset_dims))
                 << ", collapsed_slice_dims: "
                 << ToString(absl::MakeConstSpan(collapsed_slice_dims))
                 << ", operand_batching_dims: "
                 << ToString(absl::MakeConstSpan(operand_batching_dims))
                 << ", start_index_batching_dims: "
                 << ToString(absl::MakeConstSpan(start_index_batching_dims))
                 << ", start_index_map: "
                 << ToString(absl::MakeConstSpan(start_index_map))
                 << ", index_vector_dim: " << index_vector_dim
                 << ", slice_sizes: "
                 << ToString(absl::MakeConstSpan(slice_sizes));

    auto dimension_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
        &ctx, offset_dims, collapsed_slice_dims, operand_batching_dims,
        start_index_batching_dims, start_index_map, index_vector_dim);
    mlir::MlirOp rows = mlir::stablehlo::Gather(weight, indices,
                                                dimension_numbers, slice_sizes);

    return BuildRenormRow(builder, rows, max_norm, norm_type);
  }
}
}  // namespace torch_tpu
