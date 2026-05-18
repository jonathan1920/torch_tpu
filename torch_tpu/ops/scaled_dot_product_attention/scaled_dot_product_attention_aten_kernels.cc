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
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "ATen/Context.h"
#include "ATen/SDPBackend.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_bwd_bf16_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_bwd_bf16_non_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_bwd_f32_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_bwd_f32_non_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_fwd_bf16_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_fwd_bf16_non_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_fwd_f32_causal_mlir_embed.h"
#include "torch_tpu/ops/scaled_dot_product_attention/kernels/sdpa_fwd_f32_non_causal_mlir_embed.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/pjrt/mlir_to_hlo.h"

ABSL_FLAG(bool, torch_tpu_internal_sdpa_use_custom_kernel, true,
          "Use a custom kernel for scaled dot product attention.");

namespace torch_tpu {

namespace {

mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
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

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  const Dimensions& shape) {
  Dimensions shape_vec(shape.begin(), shape.end());
  return mlir::stablehlo::Reshape(mlir_op, shape_vec);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims) {
  const mlir::RankedTensorType type =
      GetTensorTypeOrDie(mlir_op_with_target_dims);
  return mlir::stablehlo::Reshape(mlir_op, type.getShape());
}

int get_batch_size(const at::Tensor& tensor) {
  int batch_size = 1;
  for (int i = 0; i < tensor.ndimension() - 3; i++)
    batch_size *= tensor.size(i);
  return batch_size;
}

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim) {
  double scale = maybe_scale ? *maybe_scale : 1.0 / std::sqrt(head_dim);
  return MakeScalarConstant(builder, scale,
                            builder.getOpBuilder().getF32Type());
}

absl::StatusOr<std::string_view> GetSdpaForwardKernel(at::ScalarType dtype,
                                                      bool is_causal) {
  static const absl::NoDestructor<
      absl::flat_hash_map<std::pair<at::ScalarType, bool>, std::string_view>>
      kKernelMap({
          {{at::kFloat, true},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_fwd_f32_causal_mlir_data),
               sdpa_fwd_f32_causal_mlir_len)},
          {{at::kFloat, false},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_fwd_f32_non_causal_mlir_data),
               sdpa_fwd_f32_non_causal_mlir_len)},
          {{at::kBFloat16, true},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_fwd_bf16_causal_mlir_data),
               sdpa_fwd_bf16_causal_mlir_len)},
          {{at::kBFloat16, false},
           std::string_view(reinterpret_cast<const char*>(
                                sdpa_fwd_bf16_non_causal_mlir_data),
                            sdpa_fwd_bf16_non_causal_mlir_len)},
      });

  auto it = kKernelMap->find({dtype, is_causal});
  if (it == kKernelMap->end()) {
    return TT_ERROR(error::kUnimplemented)
           << "Unsupported dtype for SDPA custom kernel";
  }
  return it->second;
}

absl::StatusOr<std::string_view> GetSdpaBackwardKernel(at::ScalarType dtype,
                                                       bool is_causal) {
  static const absl::NoDestructor<
      absl::flat_hash_map<std::pair<at::ScalarType, bool>, std::string_view>>
      kKernelMap({
          {{at::kFloat, true},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_bwd_f32_causal_mlir_data),
               sdpa_bwd_f32_causal_mlir_len)},
          {{at::kFloat, false},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_bwd_f32_non_causal_mlir_data),
               sdpa_bwd_f32_non_causal_mlir_len)},
          {{at::kBFloat16, true},
           std::string_view(
               reinterpret_cast<const char*>(sdpa_bwd_bf16_causal_mlir_data),
               sdpa_bwd_bf16_causal_mlir_len)},
          {{at::kBFloat16, false},
           std::string_view(reinterpret_cast<const char*>(
                                sdpa_bwd_bf16_non_causal_mlir_data),
                            sdpa_bwd_bf16_non_causal_mlir_len)},
      });

  auto it = kKernelMap->find({dtype, is_causal});
  if (it == kKernelMap->end()) {
    return TT_ERROR(error::kUnimplemented)
           << "Unsupported dtype for SDPA custom kernel";
  }
  return it->second;
}

absl::StatusOr<at::Tensor> ScaledDotProductFusedAttentionImpl(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    bool is_causal, std::optional<double> scale) {
  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims(query.sizes().begin(), query.sizes().end());
  out_dims[rank - 1] = value.size(rank - 1);

  const auto query_scalar_type = query.scalar_type();
  auto op_builder = [rank, batch_size, out_dims, query_scalar_type, is_causal,
                     scale, query](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto [query_mlir, key_mlir, value_mlir] = inputs;

    mlir::MlirBuilder& builder = query_mlir.getBuilder();

    TT_ASSIGN_OR_RETURN(std::string_view kernel_mlir,
                        GetSdpaForwardKernel(query_scalar_type, is_causal));

    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> imported_kernel,
        xla::ParseMlirModuleString(kernel_mlir, builder.getContext()));

    // Flatten batch dimensions of inputs.
    mlir::MlirOp query_4d =
        flatten_batch_dims(query_mlir, batch_size, rank - 3);
    mlir::MlirOp key_4d = flatten_batch_dims(key_mlir, batch_size, rank - 3);
    mlir::MlirOp value_4d =
        flatten_batch_dims(value_mlir, batch_size, rank - 3);

    mlir::MlirOp scale_value =
        GetScaleDefaulted(builder, scale, query.size(rank - 1));

    // Specialize the kernel given the inputs.
    std::array<mlir::MlirOp, 4> input_array = {query_4d, key_4d, value_4d,
                                               scale_value};
    TT_ASSIGN_OR_RETURN(
        auto results, BuildSpecializedMlirKernel(builder, imported_kernel.get(),
                                                 absl::MakeSpan(input_array)));

    // Unflatten batch dimensions of output.
    mlir::MlirOp out_batch = results[0];
    mlir::MlirOp out = unflatten_batch_dims(out_batch, out_dims);

    return out;
  };

  TT_ASSIGN_OR_RETURN(auto param_keys, *OpParamCacheKeysBuilder()
                                            .SetParam("is_causal", is_causal)
                                            .SetParam("scale", scale));

  TT_ASSIGN_OR_RETURN(
      auto results,
      DispatchOp<3>(std::move(op_builder), {query, key, value},
                    {.out_dtype = out_dtype,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(results));
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionBackwardImpl(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, std::optional<double> scale, bool is_causal) {
  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  const auto query_scalar_type = query.scalar_type();
  auto op_builder = [rank, batch_size, query_scalar_type, is_causal, scale,
                     query](FixedSizeSpan<mlir::MlirOp, 4> inputs)
      -> absl::StatusOr<MlirOpResults<3>> {
    auto [grad_out_mlir, query_mlir, key_mlir, value_mlir] = inputs;

    mlir::MlirBuilder& builder = grad_out_mlir.getBuilder();

    TT_ASSIGN_OR_RETURN(std::string_view kernel_mlir,
                        GetSdpaBackwardKernel(query_scalar_type, is_causal));

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

    mlir::MlirOp scale_value =
        GetScaleDefaulted(builder, scale, query.size(rank - 1));

    std::array<mlir::MlirOp, 5> input_array = {
        grad_out_batch, query_batch, key_batch, value_batch, scale_value};
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

  TT_ASSIGN_OR_RETURN(auto param_keys, *OpParamCacheKeysBuilder()
                                            .SetParam("is_causal", is_causal)
                                            .SetParam("scale", scale));

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<4, 3>(
          std::move(op_builder), {grad_out, query, key, value},
          {.out_dtypes = {out_dtype, out_dtype, out_dtype},
           .out_dims_list = {query.sizes(), key.sizes(), value.sizes()},
           .op_param_cache_keys = std::move(param_keys)})));

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
      (query, key, value, IgnoreInCacheKey(attn_mask, "Legacy usage"),
       IgnoreInCacheKey(dropout_p, "Legacy usage"),
       IgnoreInCacheKey(is_causal, "Legacy usage"),
       IgnoreInCacheKey(scale, "Legacy usage"),
       IgnoreInCacheKey(enable_gqa, "Legacy usage")),
      {
        const auto& ctx = at::globalContext();

        // The first 2 branches check for SDP kernel options, falling through
        // to the next set of specific kernel implementation checks.
        // Otherwise the else immediately falls back to the MATH backend.
        if (ctx.userEnabledOverrideableSDP() || ctx.userEnabledFlashSDP() ||
            ctx.userEnabledMemEfficientSDP()) {
          // No warning needed. Fall through to can_use_custom_kernel check.
        } else if (ctx.userEnabledCuDNNSDP()) {
          TORCH_WARN_ONCE(
              "TorchTPU only supports FLASH, EFFICIENT, OVERRIDEABLE, and MATH "
              "SDPBackends for scaled_dot_product_attention. All other backends"
              " will use FLASH if possible, or MATH for unsupported "
              "arguments.");
        } else {
          TT_CHECK_THROW(ctx.userEnabledMathSDP(), error::kFailedPrecondition)
              << "no viable SDPBackend found: all supported backends are "
                 "disabled, including the fallback MATH backend; enable at "
                 "least one of FLASH, EFFICIENT, OVERRIDEABLE, or MATH for "
                 "TorchTPU";
          return static_cast<int64_t>(at::SDPBackend::math);
        }

        // TODO(elliotenglish): Add support for attn_mask and attributes.
        constexpr int min_block_size = 128;
        constexpr int config_block_size = 512;
        constexpr int min_batch_size = 1;

        // TODO(b/483131156): Support all shapes.
        // We don't support all shapes because the kernel is specialized to
        // block_size 512 and we don't want to generate a new kernel for every
        // possible sequence length and head dim.
        const bool is_supported_flash_attention_shape =
            query.ndimension() >= 4 &&
            get_batch_size(query) >= min_batch_size &&
            query.size(query.ndimension() - 2) % config_block_size == 0 &&
            key.size(key.ndimension() - 2) % config_block_size == 0 &&
            value.size(value.ndimension() - 2) % config_block_size == 0 &&
            (query.size(query.ndimension() - 1) < min_block_size ||
             query.size(query.ndimension() - 1) % min_block_size == 0);

        const bool can_use_flash =
            (!attn_mask.has_value() && dropout_p == 0.0 &&
             (query.scalar_type() == at::ScalarType::Float ||
              query.scalar_type() == at::ScalarType::BFloat16) &&
             query.ndimension() == key.ndimension() &&
             query.ndimension() == value.ndimension() &&
             is_supported_flash_attention_shape);

        if (can_use_flash) {
          return static_cast<int64_t>(at::SDPBackend::flash_attention);
        }

        // Use c10::str rather than absl::StrCat as it will convert enums into
        // their string representation.
        std::string failure_reason = c10::str(
            "TorchTPU only supports FLASH, EFFICIENT, OVERRIDEABLE SDPBackend "
            "for scaled_dot_product_attention when these conditions are met:\n"
            "- attn_mask is None (current: ",
            (attn_mask.has_value() ? "present" : "None"),
            ")\n"
            "- dropout_p is 0.0 (current: ",
            dropout_p,
            ")\n"
            "- inputs are float32 or bfloat16 (current: ",
            query.scalar_type(),
            ")\n"
            "- inputs have the same rank (query: ",
            query.ndimension(), ", key: ", key.ndimension(),
            ", value: ", value.ndimension(),
            ")\n"
            "- query rank is at least 4 (current: ",
            query.ndimension(),
            ")\n"
            "- batch size is at least 1 (current: ",
            get_batch_size(query),
            ")\n"
            "- Sequence lengths (dim - 2) are divisible by 512 (query: ",
            query.size(query.ndimension() - 2),
            ", key: ", key.size(key.ndimension() - 2),
            ", value: ", value.size(value.ndimension() - 2),
            ")\n"
            "- Head dimension (dim - 1) is less than 128 or divisible by 128 "
            "(query: ",
            query.size(query.ndimension() - 1), ")");

        TT_CHECK_THROW(ctx.userEnabledMathSDP(), error::kFailedPrecondition)
            << failure_reason << "\nFallback MATH backend is disabled.";
        TORCH_WARN_ONCE(failure_reason, "\nFalling back to MATH backend.");
        return static_cast<int64_t>(at::SDPBackend::math);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
GenerateResults(const at::Tensor& out) {
  int64_t batch = get_batch_size(out);
  int64_t heads = out.size(out.ndimension() - 3);
  int64_t q_seq_len = out.size(out.ndimension() - 2);

  at::Tensor logsumexp =
      at::zeros({batch, heads, q_seq_len}, out.options().dtype(at::kFloat));
  at::Tensor cum_seq_q = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  at::Tensor cum_seq_k = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  at::Tensor philox_seed = at::zeros({1}, out.options().dtype(at::kLong));
  at::Tensor philox_offset = at::zeros({1}, out.options().dtype(at::kLong));
  at::Tensor debug_mask = at::zeros({1}, out.options().dtype(at::kBool));

  return std::make_tuple(std::move(out), std::move(logsumexp),
                         std::move(cum_seq_q), std::move(cum_seq_k),
                         c10::SymInt(0), c10::SymInt(0), std::move(philox_seed),
                         std::move(philox_offset), std::move(debug_mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideable(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, double dropout_p,
    bool is_causal, bool return_debug_mask, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFusedAttentionOverrideable, _,
            (query, key, value, IgnoreInCacheKey(attn_bias, "Legacy usage"),
             IgnoreInCacheKey(dropout_p, "Legacy usage"),
             IgnoreInCacheKey(is_causal, "Legacy usage"),
             IgnoreInCacheKey(return_debug_mask, "Legacy usage"),
             IgnoreInCacheKey(scale, "Legacy usage")),
            {
              TT_ASSIGN_OR_THROW(auto out,
                                 ScaledDotProductFusedAttentionImpl(
                                     query, key, value, is_causal, scale));

              return GenerateResults(out);
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideableBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFusedAttentionOverrideableBackward, _,
            (grad_out, query, key, value, attn_bias,
             IgnoreInCacheKey(grad_input_mask, "Legacy usage"), out, logsumexp,
             cum_seq_q, cum_seq_k, IgnoreInCacheKey(max_q, "Legacy usage"),
             IgnoreInCacheKey(max_k, "Legacy usage"),
             IgnoreInCacheKey(dropout_p, "Legacy usage"),
             IgnoreInCacheKey(is_causal, "Legacy usage"), philox_seed,
             philox_offset, IgnoreInCacheKey(scale, "Legacy usage")),
            {
              // Unused arguments: grad_input_mask, out, logsumexp,
              // cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal,
              // philox_seed, philox_offset, scale.
              TT_ASSIGN_OR_THROW(
                  auto out, ScaledDotProductFusedAttentionBackwardImpl(
                                grad_out, query, key, value, scale, is_causal));

              return std::make_tuple(
                  grad_input_mask[0] ? std::get<0>(out)
                                     : at::zeros({0}, grad_out.options()),
                  grad_input_mask[1] ? std::get<1>(out)
                                     : at::zeros({0}, grad_out.options()),
                  grad_input_mask[2] ? std::get<2>(out)
                                     : at::zeros({0}, grad_out.options()),
                  at::zeros({0}, grad_out.options()));
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductEfficientAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductEfficientAttention, _,
            (query, key, value, IgnoreInCacheKey(attn_bias, "Legacy usage"),
             IgnoreInCacheKey(compute_log_sumexp, "Legacy usage"),
             IgnoreInCacheKey(dropout_p, "Legacy usage"),
             IgnoreInCacheKey(is_causal, "Legacy usage"),
             IgnoreInCacheKey(scale, "Legacy usage")),
            {
              // Unused arguments: attn_bias, compute_log_sumexp, dropout_p,
              // scale.
              TT_ASSIGN_OR_THROW(auto out,
                                 ScaledDotProductFusedAttentionImpl(
                                     query, key, value, is_causal, scale));
              int64_t batch = get_batch_size(query);
              int64_t heads = query.size(query.ndimension() - 3);
              int64_t q_seq_len = query.size(query.ndimension() - 2);

              at::Tensor logsumexp = at::zeros(
                  {batch, heads, q_seq_len}, query.options().dtype(at::kFloat));
              at::Tensor philox_seed =
                  at::zeros({1}, query.options().dtype(at::kLong));
              at::Tensor philox_offset =
                  at::zeros({1}, query.options().dtype(at::kLong));

              return std::make_tuple(std::move(out), std::move(logsumexp),
                                     std::move(philox_seed),
                                     std::move(philox_offset));
            });
}

std::tuple<at::Tensor, at::Tensor> AtenScaledDotProductFlashAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFlashAttention, _,
            (query, key, value, IgnoreInCacheKey(dropout_p, "Unused"),
             IgnoreInCacheKey(is_causal, "Constructed in implementation"),
             IgnoreInCacheKey(attn_mask, "Unused"),
             IgnoreInCacheKey(scale, "Constructed in implementation")),
            {
              TT_ASSIGN_OR_THROW(auto out,
                                 ScaledDotProductFusedAttentionImpl(
                                     query, key, value, is_causal, scale));
              int64_t batch = get_batch_size(query);
              int64_t heads = query.size(query.ndimension() - 3);
              int64_t q_seq_len = query.size(query.ndimension() - 2);
              // We don't currently use this for the backward pass, so we return
              // a dummy tensor.
              at::Tensor logsumexp = at::zeros({batch, heads, q_seq_len},
                                               out.options().dtype(at::kFloat));
              return {out, logsumexp};
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFlashAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFlashAttentionBackward, _,
            (grad_out, query, key, value, out, logsumexp,
             IgnoreInCacheKey(dropout_p, "Unused"),
             IgnoreInCacheKey(is_causal, "Constructed in implementation"),
             IgnoreInCacheKey(attn_mask, "Unused"),
             IgnoreInCacheKey(scale, "Constructed in implementation")),
            {
              TT_ASSIGN_OR_THROW(
                  auto out, ScaledDotProductFusedAttentionBackwardImpl(
                                grad_out, query, key, value, scale, is_causal));

              return out;
            });
}

}  // namespace torch_tpu
