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

#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "ATen/Context.h"
#include "ATen/SDPBackend.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "c10/util/Exception.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/scaled_dot_product_attention_backward_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/scaled_dot_product_attention_forward_mlir_embed.h"

ABSL_FLAG(bool, torch_tpu_internal_sdpa_use_custom_kernel, false,
          "Use a custom kernel for scaled dot product attention.");

namespace torch_tpu {

namespace {

mlir::MlirOp flatten_batch_dims(const mlir::MlirOp& mlir_op, int batch_size,
                                int num_batch_dims) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(mlir_op);
  int rank = type.getRank();
  int num_non_batch_dims = rank - num_batch_dims;
  Dimensions new_dims(num_non_batch_dims + 1);
  new_dims[0] = batch_size;
  for (int i = 0; i < num_non_batch_dims; i++)
    new_dims[i + 1] = type.getDimSize(i + num_batch_dims);

  return mlir::stablehlo::Reshape(mlir_op, new_dims);
}

mlir::MlirOp unflatten_batch_dims(const mlir::MlirOp& mlir_op,
                                  const Dimensions& shape) {
  Dimensions shape_vec(shape.begin(), shape.end());
  return mlir::stablehlo::Reshape(mlir_op, shape_vec);
}

mlir::MlirOp unflatten_batch_dims(
    const mlir::MlirOp& mlir_op, const mlir::MlirOp& mlir_op_with_target_dims) {
  const mlir::RankedTensorType type =
      GetTensorTypeOrDie(mlir_op_with_target_dims);
  return mlir::stablehlo::Reshape(mlir_op, type.getShape());
}

absl::StatusOr<at::Tensor> ScaledDotProductFusedAttentionImpl(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value) {
  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);
  Dimensions output_dims(query.sizes().begin(), query.sizes().end());
  output_dims[rank - 1] = value.size(rank - 1);
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  auto op_builder = [rank, batch_size,
                     output_dims](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [query_mlir, key_mlir, value_mlir] = inputs;

    mlir::MlirBuilder& builder = query_mlir.getBuilder();

    std::string_view kernel_mlir(
        (char*)scaled_dot_product_attention_forward_mlir_data,
        scaled_dot_product_attention_forward_mlir_len);

    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> imported_kernel,
        xla::ParseMlirModuleString(kernel_mlir, builder.getContext()));

    // Flatten batch dimensions of inputs.
    mlir::MlirOp query_batch =
        flatten_batch_dims(query_mlir, batch_size, rank - 3);
    mlir::MlirOp key_batch = flatten_batch_dims(key_mlir, batch_size, rank - 3);
    mlir::MlirOp value_batch =
        flatten_batch_dims(value_mlir, batch_size, rank - 3);

    // Specialize the kernel given the inputs.
    std::array<mlir::MlirOp, 3> input_array = {query_batch, key_batch,
                                               value_batch};

    TT_ASSIGN_OR_RETURN(
        auto results, BuildSpecializedMlirKernel(builder, imported_kernel.get(),
                                                 absl::MakeSpan(input_array)));

    mlir::MlirOp& out_batch = results[0];

    // Unflatten batch dimensions of output.
    mlir::MlirOp out = unflatten_batch_dims(out_batch, output_dims);

    return out;
  };

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<3>(
          OpName::kScaledDotProductFusedAttentionOverrideable,
          std::move(op_builder), {query, key, value},
          {.out_dtype = out_dtype, .out_dims = absl::MakeSpan(output_dims)})));
  return MakeTensor(std::move(results));
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionBackwardImpl(const at::Tensor& grad_out,
                                           const at::Tensor& query,
                                           const at::Tensor& key,
                                           const at::Tensor& value) {
  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  auto op_builder = [rank, batch_size](FixedSizeSpan<mlir::MlirOp, 4> inputs)
      -> absl::StatusOr<MlirOpResults<3>> {
    auto& [grad_out_mlir, query_mlir, key_mlir, value_mlir] = inputs;

    mlir::MlirBuilder& builder = grad_out_mlir.getBuilder();

    std::string_view kernel_mlir(
        (char*)scaled_dot_product_attention_backward_mlir_data,
        scaled_dot_product_attention_backward_mlir_len);

    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> imported_kernel,
        xla::ParseMlirModuleString(kernel_mlir, builder.getContext()));

    mlir::MlirOp grad_out_batch =
        flatten_batch_dims(grad_out_mlir, batch_size, rank - 3);
    mlir::MlirOp query_batch =
        flatten_batch_dims(query_mlir, batch_size, rank - 3);
    mlir::MlirOp key_batch = flatten_batch_dims(key_mlir, batch_size, rank - 3);
    mlir::MlirOp value_batch =
        flatten_batch_dims(value_mlir, batch_size, rank - 3);

    std::array<mlir::MlirOp, 4> input_array = {grad_out_batch, query_batch,
                                               key_batch, value_batch};

    TT_ASSIGN_OR_RETURN(
        auto results, BuildSpecializedMlirKernel(builder, imported_kernel.get(),
                                                 absl::MakeSpan(input_array)));

    mlir::MlirOp& grad_query_batch = results[0];
    mlir::MlirOp& grad_key_batch = results[1];
    mlir::MlirOp& grad_value_batch = results[2];

    mlir::MlirOp grad_query =
        unflatten_batch_dims(grad_query_batch, query_mlir);
    mlir::MlirOp grad_key = unflatten_batch_dims(grad_key_batch, key_mlir);
    mlir::MlirOp grad_value =
        unflatten_batch_dims(grad_value_batch, value_mlir);

    return {{grad_query, grad_key, grad_value}};
  };

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<4, 3>(
          OpName::kScaledDotProductFusedAttentionOverrideableBackward,
          std::move(op_builder), {grad_out, query, key, value},
          {.out_dtypes = {out_dtype, out_dtype, out_dtype},
           .out_dims_list = {query.sizes(), key.sizes(), value.sizes()}})));

  return std::make_tuple(MakeTensor(std::move(results[0])),
                         MakeTensor(std::move(results[1])),
                         MakeTensor(std::move(results[2])));
}

}  // namespace

// Torch's extensibility is pretty gross here due to weird SDPA
// implementation.
// *  at::native::scaled_dot_product_attention calls _fused_sdp_choice_stub,
//    which then dispatches to either _fused_sdp_choice_cpp (for CPU) or
//    _fused_sdp_choice_cuda. This can be overridden by
//    REGISTER_PRIVATEUSE1_DISPATCH, which we do in tpu_aten_kernels.cc.
// *  However, these then use either sdp::select_sdp_backend_cpp (CPU) or
//    sdp::select_sdp_backend (CUDA). select_sdp_backend is *not* TORCH_API
//    overrideable, and these hardcode which backends are supported on
//    CPU and CUDA.
//
// So we have to inline the logic from sdp::select_sdp_backend* that accesses
// the at::globalContext() flags, and return either overrideable or math,
// depending on whether the arguments are supported by our current
// implementation.
//
// Also, to support easy migration for users, we only warn on an unsupported
// backend instead of erroring (we only error if *all* backends are disabled).
int64_t AtenFusedSdpChoice(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_mask,
                           double dropout_p, bool is_causal,
                           std::optional<double> scale, bool enable_gqa) {
  TT_KERNEL(
      OpName::kFusedSdpChoice, _,
      (query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa), {
        const auto& ctx = at::globalContext();

        // The first 2 branches check for overrideable SDP and other kernel
        // options, falling through to the next set of specific kernel
        // implementation checks. Otherwise the else immediately falls back to
        // the math backed.
        if (ctx.userEnabledOverrideableSDP()) {
          // No warning needed. Fall through to can_use_overrideable check.
        } else if (ctx.userEnabledFlashSDP() ||
                   ctx.userEnabledMemEfficientSDP() ||
                   ctx.userEnabledCuDNNSDP()) {
          TORCH_WARN_ONCE(
              "TorchTPU only supports OVERRIDEABLE and MATH SDPBackends. All "
              "other backends will use OVERRIDEABLE if possible, or MATH for "
              "unsupported arguments.");
        } else {
          TT_CHECK_THROW(ctx.userEnabledMathSDP(), error::kFailedPrecondition)
              << "cannot use scaled_dot_product_attention with no SDPBackends "
                 "enabled. Please enable either OVERRIDEABLE or MATH for "
                 "torch_tpu";
          return static_cast<int64_t>(at::SDPBackend::math);
        }

        // TODO(elliotenglish): Add support for attn_mask and attributes.
        const bool can_use_overrideable =
            absl::GetFlag(FLAGS_torch_tpu_internal_sdpa_use_custom_kernel) &&
            (!attn_mask.has_value() && is_causal && !scale.has_value() &&
             query.scalar_type() == at::ScalarType::Float &&
             query.ndimension() == key.ndimension() &&
             query.ndimension() == value.ndimension() &&
             query.ndimension() >= 3);

        if (can_use_overrideable) {
          return static_cast<int64_t>(at::SDPBackend::overrideable);
        }

        TORCH_WARN_ONCE(
            "TorchTPU only supports OVERRIDEABLE SDPBackend for "
            "scaled_dot_product_attention when these conditions are met:\n"
            "attn_mask is None\nis_causal is True\nscale is None\nquery "
            "uses float32\nquery, key, and value have the same rank.\n"
            "Falling back to MATH backend.");
        return static_cast<int64_t>(at::SDPBackend::math);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideable(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, double dropout_p,
    bool is_causal, bool return_debug_mask, std::optional<double> scale) {
  ABSL_LOG(INFO) << "AtenScaledDotProductFusedAttentionOverrideable";
  TT_KERNEL(OpName::kScaledDotProductFusedAttentionOverrideable, _,
            (query, key, value, attn_bias, dropout_p, is_causal,
             return_debug_mask, scale),
            {
              // Unused arguments: attn_bias, dropout_p, is_causal,
              // return_debug_mask, scale.
              TT_ASSIGN_OR_THROW(auto out, ScaledDotProductFusedAttentionImpl(
                                               query, key, value));
              // Unused return values: logsumexp, cum_seq_q, cum_seq_k, max_q,
              // max_k, philox_seed, philox_offset, debug_attn_mask.
              return std::make_tuple(
                  out, at::Tensor(), at::Tensor(), at::Tensor(), c10::SymInt(0),
                  c10::SymInt(0), at::Tensor(), at::Tensor(), at::Tensor());
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideableBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    const std::array<bool, 4> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, std::optional<double> scale) {
  ABSL_LOG(INFO) << "AtenScaledDotProductFusedAttentionOverrideableBackward";
  TT_KERNEL(OpName::kScaledDotProductFusedAttentionOverrideableBackward, _,
            (grad_out, query, key, value, attn_bias, grad_input_mask, out,
             logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
             is_causal, philox_seed, philox_offset, scale),
            {
              // Unused arguments: attn_bias, grad_input_mask, out, logsumexp,
              // cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal,
              // philox_seed, philox_offset, scale.
              TT_ASSIGN_OR_THROW(auto out,
                                 ScaledDotProductFusedAttentionBackwardImpl(
                                     grad_out, query, key, value));
              // Unused return value: grad_attn_bias
              return std::make_tuple(std::get<0>(out), std::get<1>(out),
                                     std::get<2>(out), at::Tensor());
            });
}

}  // namespace torch_tpu
