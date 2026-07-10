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

#include "torch_tpu/ops/slice_scatter/slice_scatter_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildSliceScatterShlo(mlir::MlirOp self,
                                                   mlir::MlirOp src,
                                                   int64_t dim, int64_t start,
                                                   int64_t end, int64_t step) {
  mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  mlir::RankedTensorType src_type = GetTensorTypeOrDie(src);
  const int64_t rank = self_type.getRank();
  if (self_type.getElementType() != src_type.getElementType()) {
    src = mlir::stablehlo::ConvertElementType(src, self_type.getElementType());
    src_type = GetTensorTypeOrDie(src);
  }

  const int64_t num_updates = src_type.getShape()[dim];
  if (num_updates == mlir::ShapedType::kDynamic) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "Dynamic slice_scatter is not supported yet";
  }

  const int64_t expected_size =
      (start < end) ? (end - start + step - 1) / step : 0;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked in AtenSliceScatter
      num_updates == expected_size, error::kInvalidArgument)
      << "expected src shape size " << expected_size << " at dim " << dim
      << ", got " << num_updates;

  mlir::MlirBuilder& builder = self.getBuilder();
  mlir::MlirOp indices_op;

  const bool apply_update_as_contiguous_block = (step == 1 || num_updates <= 1);

  // Note that the emitted SHLO here is slightly different. That is okay here
  // as the condition for differing SHLO is in the hash key.
  if (apply_update_as_contiguous_block) {
    auto indices_type =
        mlir::RankedTensorType::get({1}, builder.getOpBuilder().getI64Type());
    auto indices_attr = mlir::DenseIntElementsAttr::get(indices_type, {start});
    indices_op = mlir::stablehlo::Constant(builder, indices_attr);
  } else {
    Indices indices_raw;
    indices_raw.reserve(num_updates);
    for (int64_t i = 0; i < num_updates; ++i) {
      indices_raw.push_back(start + i * step);
    }
    auto indices_type = mlir::RankedTensorType::get(
        {num_updates, 1}, builder.getOpBuilder().getI64Type());
    auto indices_attr = mlir::DenseIntElementsAttr::get(
        indices_type, llvm::ArrayRef<int64_t>(indices_raw));
    indices_op = mlir::stablehlo::Constant(builder, indices_attr);
  }

  Dimensions update_window_dims;
  Dimensions inserted_window_dims;
  if (apply_update_as_contiguous_block) {
    update_window_dims.resize(rank);
    std::iota(update_window_dims.begin(), update_window_dims.end(), 0);
  } else {
    update_window_dims.reserve(rank - 1);
    for (int64_t d = 0; d < rank; ++d) {
      if (d != dim) {
        update_window_dims.push_back(d);
      }
    }
    inserted_window_dims = {dim};
  }

  int64_t index_vector_dim = apply_update_as_contiguous_block ? 0 : 1;

  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/update_window_dims,
          /*inserted_window_dims=*/inserted_window_dims,
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{dim},
          /*index_vector_dim=*/index_vector_dim);

  auto block_type = mlir::RankedTensorType::get({}, self_type.getElementType());
  auto region_builder = [block_type](mlir::RegionBuilder& builder) {
    mlir::Argument(builder, block_type);              // current (ignored)
    auto arg1 = mlir::Argument(builder, block_type);  // update
    mlir::stablehlo::Return(builder, {arg1});
  };

  auto result = mlir::stablehlo::Scatter(
      {self}, indices_op, {src}, region_builder, scatter_dimension_numbers,
      /*indices_are_sorted=*/true,
      /*unique_indices=*/true)[0];

  return result;
}

}  // namespace

at::Tensor AtenSliceScatter(const at::Tensor& self, const at::Tensor& src,
                            int64_t dim, std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step) {
  auto slice_scatter_helper =
      [&](const at::Tensor& self,
          const at::Tensor& src) -> absl::StatusOr<at::Tensor> {
    TT_KERNEL(OpName::kSliceScatter, _, (self, src), {
      TT_RET_CHECK(self.dim() > 0, error::kIndexError)
          << "slice_scatter requires self to have at least 1 dimension, got "
             "rank 0";
      TT_RET_CHECK(self.dim() == src.dim(), error::kInvalidArgument)
          << "slice_scatter requires self and src to have the same number of "
             "dimensions, got "
          << self.dim() << " and " << src.dim();
      TT_RET_CHECK(step > 0, error::kInvalidArgument)
          << "step must be greater than 0, got " << step;

      TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
      const int64_t dim_size = self.size(dim);
      int64_t resolved_start = start.value_or(0);
      if (resolved_start < 0) resolved_start += dim_size;
      int64_t resolved_end = end.value_or(dim_size);
      if (resolved_end < 0) resolved_end += dim_size;
      resolved_start =
          std::max<int64_t>(0, std::min<int64_t>(dim_size, resolved_start));
      resolved_end =
          std::max<int64_t>(0, std::min<int64_t>(dim_size, resolved_end));

      const int64_t expected_slice_size =
          (resolved_start < resolved_end)
              ? (resolved_end - resolved_start + step - 1) / step
              : 0;
      TT_RET_CHECK(src.size(dim) == expected_slice_size,
                   error::kInvalidArgument)
          << "expected src shape size " << expected_slice_size << " at dim "
          << dim << ", got " << src.size(dim);

      for (int64_t d = 0; d < self.dim(); ++d) {
        if (d != dim) {
          TT_RET_CHECK(self.size(d) == src.size(d), error::kInvalidArgument)
              << "expected src and self sizes to match at dim " << d
              << ", got self size " << self.size(d) << " and src size "
              << src.size(d);
        }
      }

      TT_ASSIGN_OR_RETURN(const auto output_dtype,
                          ConvertTo<mlir::ElementType>(self.scalar_type()));
      Dimensions output_dims = CopyIntVector(self.sizes());

      auto slice_scatter_builder = [dim, resolved_start, resolved_end, step](
                                       FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self, src] = inputs;
        return BuildSliceScatterShlo(self, src, dim, resolved_start,
                                     resolved_end, step);
      };

      TT_ASSIGN_OR_RETURN(
          auto param_keys,
          TT_MAKE_OP_PARAM_CACHE_KEYS(dim, resolved_start, resolved_end, step));

      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef result_buf,
          DispatchOp<2>(std::move(slice_scatter_builder), {self, src},
                        {.out_dtype = output_dtype,
                         .out_dims = output_dims,
                         .op_param_cache_keys = std::move(param_keys)}));

      TT_ASSIGN_OR_RETURN(
          at::Tensor out,
          MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
      TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
      return out;
    });
  };

  TT_ASSIGN_OR_THROW(at::Tensor result, slice_scatter_helper(self, src));

  return result;
}

}  // namespace torch_tpu
