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

#include "torch_tpu/ops/unary.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

absl::StatusOr<mlir::MlirOp> LogN(mlir::MlirOp input_op, int32_t n,
                                  mlir::ElementType default_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_input_op,
                      ConvertIfInteger(input_op, default_dtype));
  const mlir::RankedTensorType converted_input_type =
      GetTensorTypeOrDie(converted_input_op);

  mlir::MlirBuilder& builder = input_op.getBuilder();

  // Defined as LogN(x) = log(x) / log(N):
  mlir::MlirOp log_op = stablehlo::Log(converted_input_op);
  mlir::MlirOp value_n =
      MakeScalarConstant(builder, n, converted_input_type.getElementType());
  mlir::MlirOp log_of_n = stablehlo::Log(value_n);
  mlir::MlirOp result_op = mlir::chlo::BroadcastDiv(log_op, log_of_n);
  return result_op;
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildAbsShlo(mlir::MlirOp input) {
  // Unfortunately shlo.abs() works only on signed ints and floats; hence, this
  // shortcut for unsigned ints.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();
  if (element_type.isUnsignedInteger()) {
    return input;
  }

  auto res = mlir::stablehlo::Abs(input);

  // Given a complex input shlo.abs() returns a complex type even if the result
  // is real, where aten::abs() returns a real value. Hence, the workaround
  // below.
  if (llvm::isa<mlir::ComplexType>(element_type)) {
    res = mlir::stablehlo::Real(res);
  }

  return res;
}

absl::StatusOr<mlir::MlirOp> BuildConjPhysicalShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(c10::ScalarType dtype,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));
  if (!c10::isComplexType(dtype)) {
    return input_op;
  }
  auto real = mlir::stablehlo::Real(input_op);
  auto imag = mlir::stablehlo::Imag(input_op);
  auto neg_imag = mlir::stablehlo::Neg(imag);
  return mlir::stablehlo::Complex(real, neg_imag);
}

absl::StatusOr<mlir::MlirOp> BuildReciprocalShlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_input_op,
                      ConvertIfInteger(input_op, default_mlir_type));
  mlir::MlirOp one_scalar = MakeConstantLike(converted_input_op, 1.0);
  mlir::MlirOp result_op = stablehlo::Div(one_scalar, converted_input_op);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildReluShlo(mlir::MlirOp input_op) {
  mlir::MlirOp zero_scalar = MakeConstantLike(input_op, 0.0);
  mlir::MlirOp result_op = stablehlo::Max(input_op, zero_scalar);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildSiluShlo(mlir::MlirOp input_op) {
  // Only defined for floating point types
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(IsFloatType(input_type), error::kInvalidArgument)
      << "SiLU input tensor element type must be floating point.";

  // Defined as silu(x) = x * sigmoid(x):
  mlir::MlirOp sigmoid_op = stablehlo::Logistic(input_op);
  mlir::MlirOp result_op = stablehlo::Mul(input_op, sigmoid_op);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildTruncShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  // For integer inputs, follows the array-api convention of
  // returning a copy of the input tensor.
  if (input_type.getElementType().isInteger()) {
    return input_op;
  }

  // For floating-point division, we floor the absolute value of the result
  // and multiply by the sign of the original result to achieve truncation
  // towards zero.
  auto abs = stablehlo::Abs(input_op);
  auto floor = stablehlo::Floor(abs);
  auto sign = stablehlo::Sign(input_op);
  mlir::MlirOp result_op = stablehlo::Mul(floor, sign);
  return result_op;
}

// TODO(b/442665129): Test the lift_fresh op when torch.compile is ready.
absl::StatusOr<mlir::MlirOp> BuildLiftFreshShlo(mlir::MlirOp input_op) {
  return input_op;
}

absl::StatusOr<mlir::MlirOp> BuildLog2Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  return LogN(input_op, 2, default_mlir_type);
}

absl::StatusOr<mlir::MlirOp> BuildLog10Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  return LogN(input_op, 10, default_mlir_type);
}

}  // namespace torch_tpu
