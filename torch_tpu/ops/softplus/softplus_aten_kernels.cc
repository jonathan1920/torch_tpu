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

#include "torch_tpu/ops/softplus/softplus_aten_kernels.h"

#include <functional>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildSoftplusShlo(mlir::MlirOp input_op,
                                               const at::Scalar& beta,
                                               const at::Scalar& threshold) {
  auto beta_op = MakeConstantLike(input_op, beta.toDouble());
  auto threshold_op = MakeConstantLike(input_op, threshold.toDouble());

  auto beta_x = mlir::stablehlo::Mul(input_op, beta_op);
  auto exp_beta_x = mlir::stablehlo::Exp(beta_x);
  auto log1p_exp_beta_x = mlir::stablehlo::Log1p(exp_beta_x);
  auto result = mlir::stablehlo::Div(log1p_exp_beta_x, beta_op);

  // Threshold judgment: beta * x > threshold
  // If the condition is met, reverts to the linear function
  mlir::MlirOp compare_gt_threshold = mlir::stablehlo::Compare(
      beta_x, threshold_op, mlir::stablehlo::ComparisonDirection::GT);

  return mlir::stablehlo::Select(compare_gt_threshold, input_op, result);
}

// Returns an MlirUnaryOpBuilder that captures beta and threshold.
MlirUnaryOpBuilder GetSoftplusFunctional(const at::Scalar& beta,
                                         const at::Scalar& threshold) {
  return std::bind(&BuildSoftplusShlo, std::placeholders::_1, beta, threshold);
}
}  // namespace

at::Tensor& AtenSoftplusOut(const at::Tensor& self, const at::Scalar& beta,
                            const at::Scalar& threshold, at::Tensor& out) {
  TT_KERNEL(OpName::kSoftplusOut, param_keys, (self, beta, threshold, out), {
    TT_CHECK_THROW(!isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
                       self.scalar_type() != at::ScalarType::ComplexFloat,
                   error::kUnimplemented)
        << "expected the input dtype to be floating-point, "
        << "got " << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out, OpName::kSoftplusOut, GetSoftplusFunctional(beta, threshold),
        {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}
}  // namespace torch_tpu
