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

#include "torch_tpu/ops/convolution/convolution_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/promote_types.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/xla_data.pb.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/convolution/convolution.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

// This helper function handles PyTorch's parameter broadcasting for
// convolutions.  It ensures that convolution parameters (specifically stride,
// padding, dilation, and output_padding) have the correct number of elements to
// match the input's spatial dimensions (e.g., 2 elements for conv2d, 1 for
// conv1d).  This function is needed because in PyTorch a user can provide
// convolution parameters in two ways:
//
//   1. A tuple matching the rank: For conv2d, providing stride=(2, 2).
//   2. A single integer (broadcasted): Providing stride=2.
//
//  When the PyTorch dispatcher calls the backend kernel, these are often passed
//  as at::IntArrayRef. If a single value is passed (like [2]), the backend must
//  "expand" it to match the spatial dimensions (becoming [2, 2]) before it can
//  be used for output shape calculations or used to generate StableHLO, which
//  expects the full array.  Without this expansion, the code below would see an
//  input with 2 spatial dimensions but a stride with only 1 dimension.
Dimensions ExpandIfNecessary(at::IntArrayRef param, int num_spatial_dims) {
  if (param.size() == 1) {
    return Dimensions(num_spatial_dims, param[0]);
  }
  return Dimensions(param.begin(), param.end());
}

absl::StatusOr<Dimensions> GetOutputDimensions(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  const int num_spatial_dims = input.dim() - 2;
  TT_RET_CHECK(num_spatial_dims > 0, error::kInvalidArgument)
      << "Convolution requires at least one spatial dimension";

  TT_RET_CHECK(stride.size() == num_spatial_dims, error::kInvalidArgument)
      << "Inconsistent number of spatial dimensions in stride";
  TT_RET_CHECK(padding.size() == num_spatial_dims, error::kInvalidArgument)
      << "Inconsistent number of spatial dimensions in padding";
  TT_RET_CHECK(dilation.size() == num_spatial_dims, error::kInvalidArgument)
      << "Inconsistent number of spatial dimensions in dilation";
  TT_RET_CHECK(output_padding.size() == num_spatial_dims,
               error::kInvalidArgument)
      << "Inconsistent number of spatial dimensions in output_padding";

  TT_RET_CHECK(weight.dim() == num_spatial_dims + 2, error::kInvalidArgument)
      << "Weight should be shaped as (C_out, C_in / groups, *spatial_dims)";

  if (transposed) {
    TT_RET_CHECK(weight.size(0) == input.size(1), error::kInvalidArgument)
        << "Dimension 0 of the weight filter must equal the number of "
           "input channels for transposed convolution";
  } else {
    TT_RET_CHECK(weight.size(1) * groups == input.size(1),
                 error::kInvalidArgument)
        << "Dimension 1 of the weight filter must equal the number of "
           "input channels divided by the number of groups";
  }

  if (bias_opt.has_value()) {
    TT_RET_CHECK(bias_opt->dim() == 1, error::kInvalidArgument)
        << "Bias should be shaped as (C_out,)";
    int64_t out_channels =
        transposed ? weight.size(1) * groups : weight.size(0);
    TT_RET_CHECK(bias_opt->size(0) == out_channels, error::kInvalidArgument)
        << "Bias and weight must have the same number of output channels. Got "
        << bias_opt->size(0) << " channels in bias and " << out_channels
        << " channels in weight";
  }

  Dimensions output_sizes;
  output_sizes.reserve(num_spatial_dims + 2);
  output_sizes.push_back(input.size(0));  // Batch
  if (transposed) {
    output_sizes.push_back(weight.size(1) * groups);  // Output features
  } else {
    output_sizes.push_back(weight.size(0));  // Output features
  }

  for (int i = 0; i < num_spatial_dims; ++i) {
    if (transposed) {
      // H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size -
      // 1)
      // + output_padding + 1
      int64_t output_dim_size = (input.size(i + 2) - 1) * stride[i];
      output_dim_size -= 2 * padding[i];
      output_dim_size += dilation[i] * (weight.size(i + 2) - 1);
      output_dim_size += output_padding[i] + 1;
      output_sizes.push_back(output_dim_size);
    } else {
      // Output shape formula for spatial dimensions:
      // https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
      int64_t output_dim_size = input.size(i + 2);
      output_dim_size += padding[i] * 2;
      output_dim_size += stride[i] - 1;
      output_dim_size -= (weight.size(i + 2) - 1) * dilation[i];
      // Truncating integer division is intended
      output_dim_size /= stride[i];
      output_sizes.push_back(output_dim_size);
    }
  }
  return output_sizes;
}

absl::Status IsSupportedDtype(const at::Tensor& tensor) {
  TT_RET_CHECK(tensor.scalar_type() != at::ScalarType::Undefined,
               error::kInvalidArgument)
      << "Tensor scalar type is undefined: caffe2::TypeMeta is "
      << tensor.unsafeGetTensorImpl()->dtype().name();
  TT_RET_CHECK(tensor.scalar_type() != at::ScalarType::Bool,
               error::kFailedPrecondition)
      << "PyTorch does not support convolution with tensors of type "
      << c10::toString(tensor.scalar_type()) << " (even though XLA does)";
  TT_RET_CHECK(tensor.scalar_type() != at::ScalarType::Long,
               error::kUnimplemented)
      << "XLA does not currently support convolution with tensors of type "
      << c10::toString(tensor.scalar_type()) << " (even though PyTorch does)";
  return absl::OkStatus();
}

absl::StatusOr<at::ScalarType> GetPromotedDtype(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt) {
  TT_RETURN_IF_ERROR(IsSupportedDtype(input)).SetPrepend()
      << "input tensor has unsupported type: ";
  TT_RETURN_IF_ERROR(IsSupportedDtype(weight)).SetPrepend()
      << "weight tensor has unsupported type: ";
  at::ScalarType promoted_dtype =
      at::promote_types(input.scalar_type(), weight.scalar_type());
  if (bias_opt.has_value()) {
    TT_RETURN_IF_ERROR(IsSupportedDtype(*bias_opt)).SetPrepend()
        << "bias tensor has unsupported type: ";
    promoted_dtype = at::promote_types(promoted_dtype, bias_opt->scalar_type());
  }
  return promoted_dtype;
}

absl::StatusOr<DeviceBufferRef> ConvolutionBinary(
    const OpName op_name, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    OpParamCacheKeys param_keys) {
  const int num_spatial_dims = input.dim() - 2;
  Dimensions expanded_stride = ExpandIfNecessary(stride, num_spatial_dims);
  Dimensions expanded_padding = ExpandIfNecessary(padding, num_spatial_dims);
  Dimensions expanded_dilation = ExpandIfNecessary(dilation, num_spatial_dims);
  Dimensions expanded_output_padding =
      ExpandIfNecessary(output_padding, num_spatial_dims);

  TT_ASSIGN_OR_RETURN(
      const Dimensions output_dims,
      GetOutputDimensions(input, weight, std::nullopt, expanded_stride,
                          expanded_padding, expanded_dilation, transposed,
                          expanded_output_padding, groups));
  TT_ASSIGN_OR_RETURN(at::ScalarType promoted_dtype,
                      GetPromotedDtype(input, weight, std::nullopt));
  TT_ASSIGN_OR_RETURN(const auto mlir_dtype,
                      ConvertTo<mlir::ElementType>(promoted_dtype));

  auto op_builder =
      [stride = Strides(expanded_stride.begin(), expanded_stride.end()),
       padding = Dimensions(expanded_padding.begin(), expanded_padding.end()),
       dilation =
           Dimensions(expanded_dilation.begin(), expanded_dilation.end()),
       transposed,
       output_padding = Dimensions(expanded_output_padding.begin(),
                                   expanded_output_padding.end()),
       groups, output_dims = Dimensions(output_dims.begin(), output_dims.end()),
       mlir_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [input, weight] = inputs;
        return BuildConvolution(input, weight, std::nullopt, stride, padding,
                                dilation, transposed, output_padding, groups,
                                output_dims, mlir_dtype);
      };
  const auto elem_type = mlir_dtype;
  return DispatchOp<2>(op_name, std::move(op_builder), {input, weight},
                       {.out_dtype = elem_type,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> ConvolutionTernary(
    const OpName op_name, const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, OpParamCacheKeys param_keys) {
  const int num_spatial_dims = input.dim() - 2;
  Dimensions expanded_stride = ExpandIfNecessary(stride, num_spatial_dims);
  Dimensions expanded_padding = ExpandIfNecessary(padding, num_spatial_dims);
  Dimensions expanded_dilation = ExpandIfNecessary(dilation, num_spatial_dims);
  Dimensions expanded_output_padding =
      ExpandIfNecessary(output_padding, num_spatial_dims);

  TT_ASSIGN_OR_RETURN(
      Dimensions output_dims,
      GetOutputDimensions(input, weight, bias, expanded_stride,
                          expanded_padding, expanded_dilation, transposed,
                          expanded_output_padding, groups));
  TT_ASSIGN_OR_RETURN(at::ScalarType promoted_dtype,
                      GetPromotedDtype(input, weight, bias));
  TT_ASSIGN_OR_RETURN(const auto mlir_dtype,
                      ConvertTo<mlir::ElementType>(promoted_dtype));

  auto op_builder =
      [stride = Strides(expanded_stride.begin(), expanded_stride.end()),
       padding = Dimensions(expanded_padding.begin(), expanded_padding.end()),
       dilation =
           Dimensions(expanded_dilation.begin(), expanded_dilation.end()),
       transposed,
       output_padding = Dimensions(expanded_output_padding.begin(),
                                   expanded_output_padding.end()),
       groups, output_dims = Dimensions(output_dims.begin(), output_dims.end()),
       mlir_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [input, weight, bias] = inputs;
        return BuildConvolution(input, weight, bias, stride, padding, dilation,
                                transposed, output_padding, groups, output_dims,
                                mlir_dtype);
      };
  const auto elem_type = mlir_dtype;
  return DispatchOp<3>(op_name, std::move(op_builder), {input, weight, bias},
                       {.out_dtype = elem_type,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> Convolution(
    const OpName op_name, const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups,
    OpParamCacheKeys param_keys) {
  // For some reason, we sometimes get a dispatch with a bias tensor that is
  // non-null, but undefined.
  if (bias_opt.has_value() && bias_opt->defined()) {
    return ConvolutionTernary(op_name, input, weight, bias_opt.value(), stride,
                              padding, dilation, transposed, output_padding,
                              groups, std::move(param_keys));
  }
  return ConvolutionBinary(op_name, input, weight, stride, padding, dilation,
                           transposed, output_padding, groups,
                           std::move(param_keys));
}

absl::StatusOr<at::ScalarType> GetPromotedDtypeBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight) {
  TT_RETURN_IF_ERROR(IsSupportedDtype(grad_output)).SetPrepend()
      << "grad_output tensor has unsupported type: ";
  TT_RETURN_IF_ERROR(IsSupportedDtype(input)).SetPrepend()
      << "input tensor has unsupported type: ";
  TT_RETURN_IF_ERROR(IsSupportedDtype(weight)).SetPrepend()
      << "weight tensor has unsupported type: ";
  at::ScalarType promoted_dtype =
      at::promote_types(input.scalar_type(), weight.scalar_type());
  return at::promote_types(promoted_dtype, grad_output.scalar_type());
}

}  // namespace

at::Tensor AtenConvolution(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           at::IntArrayRef dilation, bool transposed,
                           at::IntArrayRef output_padding, int64_t groups) {
  TT_KERNEL(OpName::kConvolution, param_keys,
            (input, weight, bias_opt, stride, padding, dilation, transposed,
             output_padding, groups),
            {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  Convolution(OpName::kConvolution, input, weight, bias_opt,
                              stride, padding, dilation, transposed,
                              output_padding, groups, std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenConvolutionOut(const at::Tensor& input,
                               const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias_opt,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool transposed,
                               at::IntArrayRef output_padding, int64_t groups,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kConvolutionOut, param_keys,
            (input, weight, bias_opt, stride, padding, dilation, transposed,
             output_padding, groups, out),
            {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  Convolution(OpName::kConvolutionOut, input, weight, bias_opt,
                              stride, padding, dilation, transposed,
                              output_padding, groups, std::move(param_keys)));
              at::native::resize_output(out, result.dimensions());
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenConvolutionBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  //  Parameter bias_sizes, if specified, indicates that a bias was used in the
  //  forward pass and contains the shape of the bias. While the bias shape can
  //  be computed from other inputs, it is provided to this function for ease of
  //  use. The bias shape is (weight.shape[0]) for normal convolution and
  //  (weight.shape[1] * groups) for transposed convolution.
  if (output_mask[2]) {
    if (transposed) {
      TT_CHECK_THROW(
          !bias_sizes || ((bias_sizes->size() == 1) &&
                          ((*bias_sizes)[0] == (weight.size(1) * groups))),
          error::kUnimplemented)
          << "Unexpected bias_sizes: " << ToString(*bias_sizes);
    } else {
      TT_CHECK_THROW(!bias_sizes || ((bias_sizes->size() == 1) &&
                                     ((*bias_sizes)[0] == weight.size(0))),
                     error::kUnimplemented)
          << "Unexpected bias_sizes: " << ToString(*bias_sizes);
    }
  }

  TT_KERNEL(
      OpName::kConvolutionBackward, param_keys,
      (grad_output, input, weight, stride, padding, dilation, transposed,
       output_padding, groups, output_mask),
      {
        const int num_spatial_dims = input.dim() - 2;
        Dimensions expanded_stride =
            ExpandIfNecessary(stride, num_spatial_dims);
        Dimensions expanded_padding =
            ExpandIfNecessary(padding, num_spatial_dims);
        Dimensions expanded_dilation =
            ExpandIfNecessary(dilation, num_spatial_dims);
        Dimensions expanded_output_padding =
            ExpandIfNecessary(output_padding, num_spatial_dims);

        TT_ASSIGN_OR_THROW(
            at::ScalarType promoted_dtype,
            GetPromotedDtypeBackward(grad_output, input, weight));
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(promoted_dtype));

        auto input_sizes = input.sizes();
        auto weight_sizes = weight.sizes();

        auto op_builder =
            [stride = expanded_stride, padding = expanded_padding,
             dilation = expanded_dilation,
             output_padding = expanded_output_padding, groups, transposed,
             output_mask,
             input_dims = Dimensions(input_sizes.begin(), input_sizes.end()),
             weight_dims = Dimensions(weight_sizes.begin(), weight_sizes.end()),
             output_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<MlirOpResults<3>> {
          auto& [grad_out, in, w] = inputs;
          mlir::MlirOp grad_in, grad_w, grad_b;

          auto make_undefined = [&grad_out, output_dtype]() {
            return MakeZeroSizedTensor(
                grad_out.getBuilder(),
                mlir::getElementType(grad_out.getContext(), output_dtype));
          };

          if (output_mask[0]) {
            TT_ASSIGN_OR_RETURN(
                grad_in, BuildConvolutionBackwardInput(
                             grad_out, w, input_dims, stride, padding, dilation,
                             groups, transposed, output_padding, output_dtype));
          } else {
            grad_in = make_undefined();
          }

          if (output_mask[1]) {
            TT_ASSIGN_OR_RETURN(
                grad_w,
                BuildConvolutionBackwardWeight(
                    in, grad_out, weight_dims, stride, padding, dilation,
                    groups, transposed, output_padding, output_dtype));
          } else {
            grad_w = make_undefined();
          }

          if (output_mask[2]) {
            TT_ASSIGN_OR_RETURN(
                grad_b, BuildConvolutionBackwardBias(grad_out, output_padding,
                                                     output_dtype));
          } else {
            grad_b = make_undefined();
          }

          return MlirOpResults<3>{grad_in, grad_w, grad_b};
        };

        std::array<mlir::ElementType, 3> out_dtypes = {
            output_dtype, output_dtype, output_dtype};
        Dimensions bias_dims = {transposed ? groups * weight.size(1)
                                           : weight.size(0)};
        Dimensions empty_dims = {0};

        std::array<absl::Span<const int64_t>, 3> out_dims_list = {
            output_mask[0] ? absl::Span<const int64_t>(input_sizes)
                           : absl::Span<const int64_t>(empty_dims),
            output_mask[1] ? absl::Span<const int64_t>(weight_sizes)
                           : absl::Span<const int64_t>(empty_dims),
            output_mask[2] ? absl::Span<const int64_t>(bias_dims)
                           : absl::Span<const int64_t>(empty_dims)};

        TT_ASSIGN_OR_THROW(
            auto results,
            (DispatchOp<3, 3>(
                OpName::kConvolutionBackward, std::move(op_builder),
                {grad_output, input, weight},
                {.out_dtypes =
                     FixedSizeSpan<const mlir::ElementType, 3>(out_dtypes),
                 .out_dims_list =
                     FixedSizeSpan<const absl::Span<const int64_t>, 3>(
                         out_dims_list),
                 .op_param_cache_keys = std::move(param_keys)})));

        auto to_tensor = [&](int idx) -> at::Tensor {
          if (!output_mask[idx]) return at::Tensor();
          return MakeTensor(std::move(results[idx]));
        };

        return std::make_tuple(to_tensor(0), to_tensor(1), to_tensor(2));
      });
}

}  // namespace torch_tpu
