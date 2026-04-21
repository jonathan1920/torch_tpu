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

#include "torch_tpu/ops/linalg/qr/qr_lib.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildGeqrfCustomCallOp(
    mlir::MlirOp& input, mlir::RankedTensorType out_a_type,
    mlir::RankedTensorType tau_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr("Qr"));
  mlir::NamedAttribute has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI));

  auto qr_op = mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(),
      /*resultTypes=*/{out_a_type, tau_type},
      /*operands=*/{input.getValue()},
      {call_target_attr, has_side_effect_attr, api_version_attr});

  return {{mlir::MlirOp(builder, qr_op.getResult(0)),
           mlir::MlirOp(builder, qr_op.getResult(1))}};
}

absl::StatusOr<MlirOpResults<2>> BuildGeqrfShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(input);
  const int64_t a_rank = a_type.getRank();
  const llvm::ArrayRef<int64_t> a_shape = a_type.getShape();
  const int64_t m = a_shape[a_rank - 2];
  const int64_t n = a_shape[a_rank - 1];
  const int64_t k = std::min(m, n);

  // tau has shape (..., k)
  Dimensions tau_dims(a_shape.begin(), a_shape.end() - 2);
  tau_dims.push_back(k);
  mlir::RankedTensorType tau_type =
      mlir::RankedTensorType::get(tau_dims, a_type.getElementType());

  return BuildGeqrfCustomCallOp(input, /*out_a_type=*/a_type, tau_type);
}

}  // namespace torch_tpu
