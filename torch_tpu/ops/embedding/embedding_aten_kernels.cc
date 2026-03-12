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
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/_unsafe_index_put.h"
#include "ATen/ops/div.h"
#include "ATen/ops/ones_like.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/static_shape_check.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/embedding/embedding.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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

absl::Status CheckWeightType(const at::Tensor& weight) {
  auto scalar_type = weight.scalar_type();

  TT_RET_CHECK(scalar_type == at::ScalarType::Half ||
                   scalar_type == at::ScalarType::BFloat16 ||
                   scalar_type == at::ScalarType::Float ||
                   scalar_type == at::ScalarType::Double,
               error::kInvalidArgument)
      << "expected the weight dtype to be either float16, bfloat16, float32,"
      << " or float64, got " << ToString(scalar_type);

  return absl::OkStatus();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AtenEmbeddingBag(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode,
    bool sparse, const std::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset, int64_t padding_idx) {
  TT_KERNEL(
      OpName::kEmbeddingBag, param_keys,
      (weight, indices, offsets, scale_grad_by_freq, mode, sparse,
       per_sample_weights, include_last_offset, padding_idx),
      {
        TT_THROW_IF_ERROR(CheckWeightType(weight));
        TT_THROW_IF_ERROR(CheckStaticShape(weight, "weight"));
        TT_THROW_IF_ERROR(CheckStaticShape(indices, "indices"));
        TT_THROW_IF_ERROR(CheckStaticShape(offsets, "offsets"));

        TT_ASSIGN_OR_THROW(const auto weight_dtype,
                           ConvertTo<mlir::ElementType>(weight.scalar_type()));
        TT_ASSIGN_OR_THROW(const auto indices_dtype,
                           ConvertTo<mlir::ElementType>(indices.scalar_type()));
        TT_ASSIGN_OR_THROW(const auto offsets_dtype,
                           ConvertTo<mlir::ElementType>(offsets.scalar_type()));

        int64_t indices_size = indices.numel();
        int64_t batch_size = offsets.numel();
        // If True, the size of offsets is equal to the number of bags + 1
        if (include_last_offset) batch_size -= 1;
        int64_t emb_dim = weight.size(1);
        const bool has_per_sample_weights =
            per_sample_weights.has_value() && per_sample_weights->defined();

        auto op_builder =
            [scale_grad_by_freq, mode, sparse, include_last_offset, padding_idx,
             has_per_sample_weights](absl::Span<const mlir::MlirOp> inputs,
                                     mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<4>> {
          const int64_t expected_inputs = has_per_sample_weights ? 4 : 3;
          ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
              inputs.size(), expected_inputs);

          auto weight_op = inputs[0];
          auto indices_op = inputs[1];
          auto offsets_op = inputs[2];
          std::optional<mlir::MlirOp> per_sample_weights_op;
          if (has_per_sample_weights) {
            per_sample_weights_op = inputs[3];
          }

          return BuildEmbeddingBagShlo(
              weight_op, indices_op, offsets_op, scale_grad_by_freq, mode,
              sparse, per_sample_weights_op, include_last_offset, padding_idx);
        };

        std::vector<at::Tensor> inputs = {weight, indices, offsets};
        if (has_per_sample_weights) {
          inputs.push_back(per_sample_weights.value());
        }

        TT_ASSIGN_OR_THROW(
            (auto [output, offset2bag, bag_sizes, max_indices]),
            (DispatchOp<kDynamicSize, 4>(
                OpName::kEmbeddingBag, std::move(op_builder), inputs,
                {.out_dtypes = {weight_dtype, indices_dtype, offsets_dtype,
                                indices_dtype},
                 .out_dims_list = {{batch_size, emb_dim},
                                   {indices_size},
                                   {batch_size},
                                   {batch_size, emb_dim}},
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std::move(output)),
                               MakeTensor(std::move(offset2bag)),
                               MakeTensor(std::move(bag_sizes)),
                               MakeTensor(std::move(max_indices)));
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenEmbeddingBagForwardOnly(const at::Tensor& weight, const at::Tensor& indices,
                            const at::Tensor& offsets, bool scale_grad_by_freq,
                            int64_t mode, bool sparse,
                            const std::optional<at::Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx) {
  TT_KERNEL(OpName::kEmbeddingBagForwardOnly, _,
            (weight, indices, offsets, scale_grad_by_freq, mode, sparse,
             per_sample_weights, include_last_offset, padding_idx),
            {
              return AtenEmbeddingBag(
                  weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                  per_sample_weights, include_last_offset, padding_idx);
            });
}

at::Tensor& AtenEmbeddingRenorm_(at::Tensor& self, const at::Tensor& indices,
                                 double max_norm, double norm_type) {
  TT_KERNEL(
      OpName::kEmbeddingRenorm_, op_cache_keys,
      (self, indices, max_norm, norm_type), {
        TT_CHECK_THROW(IsFloatingPoint(self) || IsComplex(self),
                       error::kInvalidArgument)
            << "expected the input dtype to be either floating point or "
               "complex, got "
            << ToString(self.scalar_type());

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
