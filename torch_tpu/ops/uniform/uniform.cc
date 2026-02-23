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

#include "torch_tpu/ops/uniform/uniform.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

// Converts u64 random bits to a uniform distribution in [from, to). The idea is
// to reinterpret as f64s, and set the exponent bits to logically represent 1.
// This gives us numbers in [1, 2), because of the implicit 1.mantissa in f64s.
// We then subtract 1, and scale to [from, to), and cast back to the desired
// type.
absl::StatusOr<mlir::MlirOp> BitsToUniform(mlir::MlirOp random_bits,
                                           mlir::RankedTensorType output_type,
                                           mlir::MlirOp from, mlir::MlirOp to) {
  mlir::RankedTensorType random_bits_type = GetTensorTypeOrDie(random_bits);
  auto& builder = random_bits.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  // We will create a f64 tensor and cast it down to the desired type.
  // Clear the exponent and sign bits
  mlir::MlirOp clear_exponent_mask =
      MakeConstantLike(random_bits, 0x000F'FFFF'FFFF'FFFFUL);
  mlir::MlirOp random_mantissa =
      stablehlo::And(random_bits, clear_exponent_mask);
  // Set the exponent bits to 0 (f64 bias is 1023, so we store 0 + 1023 =
  // 0x3FF)
  mlir::MlirOp set_exp_to_one_mask =
      MakeConstantLike(random_bits, 0x3FF0'0000'0000'0000UL);
  mlir::MlirOp random_between_one_and_two =
      stablehlo::Or(random_mantissa, set_exp_to_one_mask);
  // Interpret as f64s
  auto random_bits_type_f64 = random_bits_type.clone(op_builder.getF64Type());
  random_between_one_and_two = stablehlo::BitcastConvert(
      random_bits_type_f64, random_between_one_and_two);
  // Subtract 1.0
  mlir::MlirOp one_const = MakeConstantLike(random_between_one_and_two, 1.0);
  mlir::MlirOp random_between_zero_and_one =
      stablehlo::Subtract(random_between_one_and_two, one_const);
  // Scale to [from, to)
  mlir::RankedTensorType from_type = GetTensorTypeOrDie(from);
  from_type = from_type.clone(op_builder.getF64Type());
  from = stablehlo::Convert(from_type, from);
  mlir::RankedTensorType to_type = GetTensorTypeOrDie(to);
  to_type = to_type.clone(op_builder.getF64Type());
  to = stablehlo::Convert(to_type, to);
  auto diff_op = stablehlo::Subtract(to, from);
  diff_op = stablehlo::BroadcastInDim(random_bits_type_f64, diff_op, {});
  auto scaled_op = stablehlo::Mul(diff_op, random_between_zero_and_one);
  from = stablehlo::BroadcastInDim(random_bits_type_f64, from, {});
  auto result_f64 = stablehlo::Add(from, scaled_op);
  return stablehlo::Convert(output_type, result_f64);
}

absl::StatusOr<MlirOpResults<2>> BuildUniformShlo(
    mlir::MlirOp rng_input_state, const double from, const double to,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  const mlir::RankedTensorType rng_input_state_type =
      GetTensorTypeOrDie(rng_input_state);
  auto& builder = rng_input_state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto output_tensor_type =
      makeTensorType(builder.getContext(), sizes, mlir_type);
  mlir::MlirOp from_op =
      MakeScalarConstant(builder, from, mlir::ElementType::F64);
  mlir::MlirOp to_op = MakeScalarConstant(builder, to, mlir::ElementType::F64);
  auto output_tensor_type_uint64 =
      output_tensor_type.clone(op_builder.getIntegerType(64, false));
  auto rng_op = stablehlo::RngBitGeneratorOp::create(
      op_builder, rng_input_state.getValue().getLoc(), rng_input_state_type,
      output_tensor_type_uint64,
      stablehlo::RngAlgorithmAttr::get(op_builder.getContext(),
                                       stablehlo::RngAlgorithm::DEFAULT),
      rng_input_state.getValue());
  mlir::MlirOp rng_output_state =
      mlir::MlirOp(builder, rng_op.getOutputState());
  mlir::MlirOp rng_output_op = mlir::MlirOp(builder, rng_op.getOutput());
  TT_ASSIGN_OR_RETURN(
      auto result,
      BitsToUniform(rng_output_op, output_tensor_type, from_op, to_op));
  return {{rng_output_state, result}};
}

}  // namespace torch_tpu
