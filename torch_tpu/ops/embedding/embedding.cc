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
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/cumsum/cumsum.h"

namespace torch_tpu {

namespace {

enum class EmbeddingBagMode : int64_t {
  kSum = 0,
  kMean = 1,
  kMax = 2,
};

absl::StatusOr<EmbeddingBagMode> GetEmbeddingBagMode(int64_t mode) {
  if (mode == 0) {
    return EmbeddingBagMode::kSum;
  } else if (mode == 1) {
    return EmbeddingBagMode::kMean;
  } else if (mode == 2) {
    return EmbeddingBagMode::kMax;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected embedding bag mode to be 0, 1, or 2, got: " << mode;
  }
}

struct MaxAggregationResult {
  mlir::MlirOp output;
  mlir::MlirOp max_indices;
};

// Builds a simple scatter operation.
mlir::MlirOp BuildSimpleScatter(mlir::MlirBuilder& builder,
                                mlir::MlirOp operand, mlir::MlirOp indices,
                                mlir::MlirOp updates, bool is_update) {
  auto elem_type = GetTensorTypeOrDie(updates).getElementType();

  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(),
      /*update_window_dims=*/
      is_update ? llvm::ArrayRef<int64_t>{1} : llvm::ArrayRef<int64_t>{},
      /*inserted_window_dims=*/{0},
      /*input_batching_dims=*/{},
      /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{0},
      /*index_vector_dim=*/1);

  auto body = [elem_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        elem_type, rb.getRegion(), rb.getOpBuilder());
  };

  return mlir::stablehlo::Scatter(
      /*operand=*/operand,
      /*scatter_indices=*/indices,
      /*updates=*/updates,
      /*update_computation=*/body,
      /*dimension_numbers=*/scatter_dims)[0];
}

// Builds a tensor of shape [num_indices] where result[i] is the bag ID for
// index i, which is used to map each index to its corresponding bag.
absl::StatusOr<mlir::MlirOp> BuildOffset2Bag(mlir::MlirBuilder& builder,
                                             mlir::MlirOp indices,
                                             mlir::MlirOp offsets,
                                             mlir::Type indices_type) {
  auto zero = MakeScalarConstant(builder, 0, indices_type);
  TT_ASSIGN_OR_RETURN(auto inits, BroadcastIfNeeded(zero, indices));

  auto one = MakeScalarConstant(builder, 1, indices_type);
  TT_ASSIGN_OR_RETURN(auto updates, BroadcastIfNeeded(one, offsets));

  // Reshape offsets to [num_offsets, 1] to match indices shape [N, 1]
  TT_ASSIGN_OR_RETURN(auto offsets_reshaped, Unsqueeze(offsets, /*dim=*/1));

  auto scatter_op = BuildSimpleScatter(builder, inits, offsets_reshaped,
                                       updates, /*is_update=*/false);

  // CumSum (Prefix Sum)
  TT_ASSIGN_OR_RETURN(auto cumsum,
                      BuildCumsumShlo(0, std::nullopt, scatter_op));
  TT_ASSIGN_OR_RETURN(auto one_bcst, BroadcastIfNeeded(one, cumsum));

  return mlir::stablehlo::Subtract(cumsum, one_bcst);
}

// Gathers embedding rows from `weight` based on `indices`.
absl::StatusOr<mlir::MlirOp> BuildEmbeddingGather(mlir::MlirBuilder& builder,
                                                  mlir::MlirOp weight,
                                                  mlir::MlirOp indices) {
  const int64_t emb_dim = GetTensorTypeOrDie(weight).getDimSize(1);
  TT_RET_CHECK(!mlir::ShapedType::isDynamic(emb_dim), error::kUnimplemented)
      << "expected static embedding dimension, got dynamic";

  const int64_t indices_dim = GetTensorTypeOrDie(indices).getRank();

  auto dim_nums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &builder.getContext(),
      /*offset_dims=*/{indices_dim},
      /*collapsed_slice_dims=*/{0},
      /*operand_batching_dims=*/{},
      /*start_index_batching_dims=*/{},
      /*start_index_map=*/{0},
      /*index_vector_dim=*/indices_dim);

  return mlir::stablehlo::Gather(weight, indices, dim_nums,
                                 /*slice_sizes=*/{1, emb_dim});
}

// Builds the scatter operation for MAX mode, which aggregates both the
// embedding values and indices.
absl::StatusOr<MaxAggregationResult> BuildMaxAggregation(
    mlir::MlirBuilder& builder, mlir::MlirOp output_init,
    mlir::MlirOp scatter_indices, mlir::MlirOp masked_gathered,
    mlir::MlirOp indices, mlir::Type weight_elem_type,
    mlir::Type indices_elem_type) {
  // Prepare iota update for indices
  mlir::MlirOp indices_iota_update =
      mlir::stablehlo::IotaLike(masked_gathered, 0, indices_elem_type);

  auto zeros_scalar = MakeScalarConstant(builder, 0, indices_elem_type);
  TT_ASSIGN_OR_RETURN(auto indices_init,
                      BroadcastIfNeeded(zeros_scalar, output_init));

  // Scatter operation for both values and indices
  auto dim_nums = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(),
      /*update_window_dims=*/{1},
      /*inserted_window_dims=*/{0},
      /*input_batching_dims=*/{},
      /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{0},
      /*index_vector_dim=*/1);

  auto scatter_op = mlir::stablehlo::Scatter(
      /*operand=*/{output_init, indices_init},
      /*scatter_indices=*/scatter_indices,
      /*updates=*/{masked_gathered, indices_iota_update},
      /*update_computation=*/
      [&](mlir::RegionBuilder& body) {
        mlir::stablehlo::buildMaxAndArgmaxBody(
            weight_elem_type, indices_elem_type, body.getRegion(),
            body.getOpBuilder());
      },
      /*dimension_numbers=*/dim_nums);

  return MaxAggregationResult{.output = scatter_op[0],
                              .max_indices = scatter_op[1]};
}

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

absl::StatusOr<MlirOpResults<4>> BuildEmbeddingBagShlo(
    mlir::MlirOp weight, mlir::MlirOp indices, mlir::MlirOp offsets,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    std::optional<mlir::MlirOp> per_sample_weights, bool include_last_offset,
    int64_t padding_idx) {
  mlir::MlirBuilder& builder = weight.getBuilder();

  const auto weight_type = GetTensorTypeOrDie(weight);
  const auto weight_elem_type = weight_type.getElementType();
  const auto indices_type = GetTensorTypeOrDie(indices);
  const auto indices_elem_type = indices_type.getElementType();
  const auto offsets_type = GetTensorTypeOrDie(offsets);

  // Notice: This currently uses static shape indices, more work is needed
  // to determine how this op supports bounded dynamic values.
  TT_RET_CHECK(weight_type.hasStaticShape() && indices_type.hasStaticShape() &&
                   offsets_type.hasStaticShape(),
               error::kUnimplemented)
      << "expected static shapes for weight, indices and offsets, got dynamic";

  const int64_t emb_dim = weight_type.getDimSize(1);
  TT_RET_CHECK(!mlir::ShapedType::isDynamic(emb_dim), error::kUnimplemented)
      << "expected static embedding dimension, got dynamic";

  const int64_t offsets_size = offsets_type.getDimSize(0);
  TT_RET_CHECK(!mlir::ShapedType::isDynamic(offsets_size),
               error::kUnimplemented)
      << "expected static offsets size, got dynamic";

  const int64_t batch_size =
      include_last_offset ? offsets_size - 1 : offsets_size;

  bool needs_upcast = weight_elem_type.isF16() || weight_elem_type.isBF16();
  mlir::Type acc_elem_type =
      needs_upcast ? builder.getOpBuilder().getF32Type() : weight_elem_type;

  // Note: more work is needed to support slicing dynamic offsets
  if (include_last_offset) {
    offsets = mlir::stablehlo::Slice(offsets, {0}, {batch_size}, {1});
  }

  TT_ASSIGN_OR_RETURN(EmbeddingBagMode embedding_bag_mode,
                      GetEmbeddingBagMode(mode));

  // 1. Generate offset2bag, which maps each input index to its bag id
  TT_ASSIGN_OR_RETURN(
      auto offset2bag,
      BuildOffset2Bag(builder, indices, offsets, indices_elem_type));

  // 2. Gather the embedding rows based on the input indices
  TT_ASSIGN_OR_RETURN(auto gathered,
                      BuildEmbeddingGather(builder, weight, indices));
  mlir::MlirOp masked_gathered = gathered;
  if (needs_upcast) {
    // Upcast to F32 for aggregation
    TT_ASSIGN_OR_RETURN(masked_gathered, PromoteFloatDtype(masked_gathered));
  }

  // 3. Mask gathered embeddings: if padding_idx is set, then the gathered
  // embedding rows corresponding to padding indices should be set to 0 or -inf
  mlir::MlirOp init_val =
      (embedding_bag_mode == EmbeddingBagMode::kMax)
          ?
          // min finite value for MAX mode
          mlir::stablehlo::Constant(
              builder,
              mlir::DenseElementsAttr::get(
                  mlir::RankedTensorType::get({}, acc_elem_type),
                  GetMinFiniteValueAttr(acc_elem_type, builder.getOpBuilder())))
          :
          // 0 for SUM/MEAN mode
          MakeScalarConstant(builder, 0.0, acc_elem_type);

  mlir::MlirOp is_padding;
  if (padding_idx >= 0) {
    auto pad_scalar =
        MakeScalarConstant(builder, padding_idx, indices_elem_type);
    TT_ASSIGN_OR_RETURN(auto pad_bcst, BroadcastIfNeeded(pad_scalar, indices));
    is_padding = mlir::stablehlo::Compare(
        indices, pad_bcst, mlir::stablehlo::ComparisonDirection::EQ);

    TT_ASSIGN_OR_RETURN(auto is_padding_unsqueezed, Unsqueeze(is_padding, 1));
    TT_ASSIGN_OR_RETURN(auto mask_bcst, BroadcastIfNeeded(is_padding_unsqueezed,
                                                          masked_gathered));
    TT_ASSIGN_OR_RETURN(auto fill_bcst,
                        BroadcastIfNeeded(init_val, masked_gathered));
    masked_gathered =
        mlir::stablehlo::Select(mask_bcst, fill_bcst, masked_gathered);
  }

  // 4. Per-sample weights if provided
  if (per_sample_weights.has_value()) {
    TT_ASSIGN_OR_RETURN(auto psw_unsqueezed,
                        Unsqueeze(per_sample_weights.value(), 1));
    if (needs_upcast) {
      TT_ASSIGN_OR_RETURN(psw_unsqueezed, PromoteFloatDtype(psw_unsqueezed));
    }
    TT_ASSIGN_OR_RETURN(auto psw_bcst,
                        BroadcastIfNeeded(psw_unsqueezed, masked_gathered));
    masked_gathered = mlir::stablehlo::Mul(masked_gathered, psw_bcst);
  }

  // 5. Aggregation (Scatter) -> output [batch_size, emb_dim]
  TT_ASSIGN_OR_RETURN(auto output_init,
                      BroadcastIfNeeded(init_val, {batch_size, emb_dim}));
  TT_ASSIGN_OR_RETURN(auto scatter_indices, Unsqueeze(offset2bag, 1));

  mlir::MlirOp output;
  mlir::MlirOp max_indices;

  auto ones_scalar = MakeScalarConstant(builder, 1, indices_elem_type);
  auto zeros_scalar = MakeScalarConstant(builder, 0, indices_elem_type);

  // MAX mode, also compute max_indices for backward pass
  if (embedding_bag_mode == EmbeddingBagMode::kMax) {
    TT_ASSIGN_OR_RETURN(
        auto max_aggregation_result,
        BuildMaxAggregation(builder, output_init, scatter_indices,
                            masked_gathered, indices, acc_elem_type,
                            indices_elem_type));
    output = max_aggregation_result.output;
    max_indices = max_aggregation_result.max_indices;
  } else {  // SUM or MEAN mode
    output = BuildSimpleScatter(builder, output_init, scatter_indices,
                                masked_gathered,
                                /*is_update=*/true);
    TT_ASSIGN_OR_RETURN(max_indices, BroadcastIfNeeded(zeros_scalar, output));
  }

  // 6. Calculate Bag Size for MEAN mode and handling empty bags
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bag_count_vals,
                      BroadcastIfNeeded(ones_scalar, indices));
  if (padding_idx >= 0) {
    TT_ASSIGN_OR_RETURN(auto zeros, BroadcastIfNeeded(zeros_scalar, indices));
    bag_count_vals = mlir::stablehlo::Select(is_padding, zeros, bag_count_vals);
  }

  TT_ASSIGN_OR_RETURN(auto bag_size_init,
                      BroadcastIfNeeded(zeros_scalar, offsets));
  auto bag_size = BuildSimpleScatter(builder, bag_size_init, scatter_indices,
                                     bag_count_vals, /*is_update=*/false);

  // Handle empty bags to return 0 for all elements in the empty bag
  auto zeros_float_scalar = MakeScalarConstant(builder, 0.0, acc_elem_type);
  auto ones_float_scalar = MakeScalarConstant(builder, 1.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto zeros_float,
                      BroadcastIfNeeded(zeros_float_scalar, output));
  TT_ASSIGN_OR_RETURN(auto ones_float,
                      BroadcastIfNeeded(ones_float_scalar, output));
  TT_ASSIGN_OR_RETURN(auto zeros_idx,
                      BroadcastIfNeeded(zeros_scalar, max_indices));

  auto is_empty_bag = mlir::stablehlo::Compare(
      bag_size, bag_size_init, mlir::stablehlo::ComparisonDirection::EQ);
  TT_ASSIGN_OR_RETURN(auto is_empty_bag_unsqueezed, Unsqueeze(is_empty_bag, 1));
  TT_ASSIGN_OR_RETURN(auto is_empty_bag_bcst,
                      BroadcastIfNeeded(is_empty_bag_unsqueezed, output));

  output = mlir::stablehlo::Select(is_empty_bag_bcst, zeros_float, output);
  max_indices =
      mlir::stablehlo::Select(is_empty_bag_bcst, zeros_idx, max_indices);

  // Handle MEAN mode with safe division
  if (embedding_bag_mode == EmbeddingBagMode::kMean) {
    auto bag_size_float = mlir::stablehlo::Convert(
        mlir::RankedTensorType::get({batch_size}, acc_elem_type), bag_size);
    TT_ASSIGN_OR_RETURN(auto bag_size_float_unsqueezed,
                        Unsqueeze(bag_size_float, 1));
    TT_ASSIGN_OR_RETURN(auto bag_size_bcst,
                        BroadcastIfNeeded(bag_size_float_unsqueezed, output));

    auto bag_is_empty = mlir::stablehlo::Compare(
        bag_size_bcst, zeros_float, mlir::stablehlo::ComparisonDirection::EQ);
    auto safe_divisor =
        mlir::stablehlo::Select(bag_is_empty, ones_float, bag_size_bcst);

    output = mlir::stablehlo::Div(output, safe_divisor);
  }

  // 8. Convert back to original precision (Downcasting)
  if (needs_upcast) {
    auto original_output_type =
        mlir::RankedTensorType::get({batch_size, emb_dim}, weight_elem_type);
    output = mlir::stablehlo::Convert(original_output_type, output);
  }

  return MlirOpResults<4>{output, offset2bag, bag_size, max_indices};
}

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
