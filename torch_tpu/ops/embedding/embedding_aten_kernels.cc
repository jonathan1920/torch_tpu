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

#include "torch_tpu/ops/embedding/embedding_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/_unsafe_index_put.h"
#include "ATen/ops/div.h"
#include "ATen/ops/ones_like.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/embedding/embedding.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {


// Maps the dtype from:
//   torch.bfloat16 -> torch.float32
//   torch.float16 -> torch.float32
//   torch.complex32 -> torch.complex64
// Otherwise, return the original dtype.
// This implementation is adapted from:
// https://github.com/pytorch/pytorch/blob/4cf29004749714670fee9e7e3776778faf5ced25/torch/_prims_common/__init__.py#L1439-L1447
at::ScalarType GetComputationDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::BFloat16:
      return at::ScalarType::Float;
    case at::ScalarType::Half:
      return at::ScalarType::Float;
    case at::ScalarType::ComplexFloat:
      return at::ScalarType::ComplexDouble;
    default:
      return dtype;
  }
}

// This expands x until x.dim() == dim. Might be useful as an operator
at::Tensor UnsqueezeToDim(const at::Tensor& x, int dim) {
  at::Tensor result = x;
  for (int i = 0; i < dim - x.dim(); ++i) {
    result = result.unsqueeze(-1);
  }
  return result;
}

}  // namespace

at::Tensor& AtenEmbeddingRenorm_(at::Tensor& self, const at::Tensor& indices,
                                 double max_norm, double norm_type) {
  TT_KERNEL(
      OpName::kEmbeddingRenorm_, op_cache_keys,
      (self, indices, max_norm, norm_type), {
        auto scalar_type = self.scalar_type();
        bool is_float_or_complex =
            c10::isFloatingType(scalar_type) || c10::isComplexType(scalar_type);

        TT_CHECK_THROW(is_float_or_complex, error::kInvalidArgument)
            << "input dtype should be either floating point or complex. Got "
            << scalar_type << " instead.";

        Dimensions output_dims(indices.sizes().begin(), indices.sizes().end());
        output_dims.push_back(self.size(1));
        auto fn = [max_norm, norm_type](FixedSizeSpan<mlir::MlirOp, 2> ops)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [weight, indices] = ops;
          return BuildEmbeddingRenormShlo(weight, indices, max_norm, norm_type);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto renorm_rows,
            DispatchOp<2>(OpName::kEmbeddingRenorm_, std::move(fn),
                          {self, indices},
                          {.out_dtype = element_type,
                           .out_dims = {output_dims},
                           .op_param_cache_keys = std::move(op_cache_keys)}));

        // The CPU and GPU implementations loop through `indices`, compute the
        // renormed row, and update `self` one by one.
        // We instead compute all renormed rows corresponding to `indices` in
        // batch via `BuildEmbeddingRenormShlo`, and then update `self`.
        // This is likely to increase the peak memory usage noticably if the
        // embedding dim * indices.size() is *large* but should be faster.
        auto res = MakeTensor(renorm_rows);
        self.index_put_({indices}, res);
        return self;
      });
}

at::Tensor AtenEmbeddingDenseBackward(const at::Tensor& grad_output,
                                      const at::Tensor& indices,
                                      at::SymInt num_weights,
                                      at::SymInt padding_idx_sym,
                                      bool scale_grad_by_freq) {
  TT_KERNEL(
      OpName::kEmbeddingDenseBackward, _,
      (grad_output, indices, num_weights, padding_idx_sym, scale_grad_by_freq),
      {
        auto dtype = grad_output.scalar_type();
        auto computation_dtype = GetComputationDtype(dtype);
        auto result_dtype = dtype;
        auto converted_grad_output = grad_output.to(computation_dtype);
        auto converted_indices = indices;
        if (indices.dtype() != at::ScalarType::Long) {
          converted_indices = indices.to(at::ScalarType::Long);
        }
        int64_t num_weights_val = num_weights.expect_int();
        if (scale_grad_by_freq) {
          Dimensions counts_shape = {num_weights_val};
          auto counts = converted_indices.new_zeros(counts_shape);
          auto ones = at::ones_like(converted_indices);
          counts = at::_unsafe_index_put(counts, {converted_indices}, ones,
                                         /*accumulate=*/true);
          auto grad_weights_scale =
              counts.index_select(0, converted_indices.reshape(-1))
                  .view(converted_indices.sizes());
          converted_grad_output =
              at::div(converted_grad_output, grad_weights_scale.unsqueeze(-1));
        }
        int64_t padding_idx = padding_idx_sym.expect_int();
        auto mask = UnsqueezeToDim(converted_indices.eq(padding_idx),
                                   converted_grad_output.dim());
        auto grad = converted_grad_output.masked_fill(mask, 0);
        Dimensions grad_weight_shape = {num_weights_val};
        grad_weight_shape.insert(
            grad_weight_shape.end(),
            converted_grad_output.sizes().begin() + indices.dim(),
            converted_grad_output.sizes().end());
        auto grad_weight = grad.new_zeros(grad_weight_shape);
        return at::_unsafe_index_put(grad_weight, {converted_indices}, grad,
                                     /*accumulate=*/true)
            .to(result_dtype);
      });
}

}  // namespace torch_tpu
