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

#include "torch_tpu/ops/binary_cross_entropy/binary_cross_entropy_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildBinaryCrossEntropyShlo(
    mlir::MlirOp input, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::Type mlir_type = input_type.getElementType();

  // raw_log_input = log(input)
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(mlir_type));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp raw_log_input,
                      BuildLogShlo(input, element_type));

  // neg_input = -input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp neg_input, BuildNegShlo(input));
  // raw_log_1_minus_input = log(1 - input)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp raw_log_1_minus_input,
                      BuildLog1pShlo(neg_input, element_type));

  // min_log = -100.0
  const mlir::MlirOp min_log = MakeScalarConstant(builder, -100.0, mlir_type);

  // log_input = max(log(input), -100.0)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp log_input,
                      BuildMaximumShlo(raw_log_input, min_log));
  // log_1_minus_input = max(log(1 - input), -100.0)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp log_1_minus_input,
                      BuildMaximumShlo(raw_log_1_minus_input, min_log));

  // loss = term1 - term2
  //      = (target - 1) * log_1_minus_input - target * log_input

  const mlir::MlirOp one = MakeScalarConstant(builder, 1.0, mlir_type);
  // target_minus_one = target - 1
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp target_minus_one,
                      BuildSubShlo(target, one));

  // term1 = (target - 1) * log_1_minus_input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp term1,
                      BuildMulShlo(target_minus_one, log_1_minus_input));
  // term2 = target * log_input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp term2,
                      BuildMulShlo(target, log_input));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp loss, BuildSubShlo(term1, term2));

  // loss = loss * weight
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(loss, BuildMulShlo(loss, *weight));
  }

  if (reduction == at::Reduction::None) {
    return loss;
  }

  // reduced_loss = sum(loss)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp reduced_loss,
                      BuildSumShlo(loss, GetAllDimensions(loss)));

  if (reduction == at::Reduction::Sum) {
    return reduced_loss;
  }

  if (reduction == at::Reduction::Mean) {
    // numel = count(input)
    const mlir::MlirOp numel = GetNumElements(input, mlir_type);
    // return reduced_loss / numel
    return BuildDivShlo(reduced_loss, numel);
  }

  return TT_ERROR(error::kInvalidArgument)
         << "expected valid reduction mode (Sum, Mean, or None), got "
         << reduction;
}

absl::StatusOr<DeviceBufferRef> DispatchBinaryCrossEntropy(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    OpParamCacheKeys param_keys) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected floating point input, got " << ToString(self.scalar_type());
  TT_RET_CHECK(target.is_floating_point(), error::kInvalidArgument)
      << "expected floating point target, got "
      << ToString(target.scalar_type());
  TT_RET_CHECK(self.sizes() == target.sizes(), error::kInvalidArgument)
      << "expected input and target shapes to match, got " << self.sizes()
      << " vs " << target.sizes();

  const bool has_weight = weight.has_value() && weight->defined();

  auto op_builder =
      [reduction, has_weight](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp self_op = inputs[0];
    const mlir::MlirOp target_op = inputs[1];
    const std::optional<mlir::MlirOp> weight_op =
        has_weight ? std::make_optional(inputs[2]) : std::nullopt;

    return BuildBinaryCrossEntropyShlo(self_op, target_op, weight_op, reduction,
                                       builder);
  };

  std::vector<at::Tensor> inputs = {self, target};
  if (has_weight) {
    inputs.push_back(*weight);
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  const auto comp_type = ToAccumulateType(self.scalar_type());
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_type));

  Dimensions out_dims = (reduction == at::Reduction::None)
                            ? CopyIntVector(self.sizes())
                            : Dimensions();

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = out_dims,
      .computation_dtype = computation_dtype,
      .op_param_cache_keys = std::move(param_keys),
  };

  TT_ASSIGN_OR_RETURN(auto output_buf,
                      (DispatchOp<kDynamicSize, 1>(
                          std::move(op_builder), inputs, std::move(options))));
  return std::move(output_buf);
}

}  // namespace

at::Tensor AtenBinaryCrossEntropy(const at::Tensor& self,
                                  const at::Tensor& target,
                                  const std::optional<at::Tensor>& weight,
                                  int64_t reduction) {
  TT_KERNEL(OpName::kBinaryCrossEntropy, param_keys,
            (self, target, weight, reduction), {
              TT_ASSIGN_OR_THROW(
                  auto output_buf,
                  DispatchBinaryCrossEntropy(self, target, weight, reduction,
                                             std::move(param_keys)));
              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenBinaryCrossEntropyOut(const at::Tensor& self,
                                      const at::Tensor& target,
                                      const std::optional<at::Tensor>& weight,
                                      int64_t reduction, at::Tensor& out) {
  TT_KERNEL(
      OpName::kBinaryCrossEntropyOut, param_keys,
      (self, target, weight, reduction, out), {
        TT_ASSIGN_OR_THROW(auto output_buf, DispatchBinaryCrossEntropy(
                                                self, target, weight, reduction,
                                                std::move(param_keys)));

        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, output_buf.dimensions()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
