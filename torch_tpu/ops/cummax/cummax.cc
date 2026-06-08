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

#include "torch_tpu/ops/cummax/cummax.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/scan_builder.h"

namespace torch_tpu {

absl::StatusOr<CummaxOutputs> BuildCummaxShlo(const int64_t dim,
                                              mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim, SafeWrapDim(dim, rank));
  const llvm::ArrayRef<int64_t> shape = input_type.getShape();
  llvm::SmallVector<int64_t> carry_shape(shape.begin(), shape.end());
  carry_shape[normalized_dim] = 1;

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::Type element_type = input_type.getElementType();
  const mlir::MlirOp value_init = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, element_type),
          GetMinFiniteValueAttr(element_type, builder.getOpBuilder())));

  const mlir::MlirOp index_init =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp value_init_bcast,
                      BroadcastIfNeeded(value_init, carry_shape));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp index_init_bcast,
                      BroadcastIfNeeded(index_init, carry_shape));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp value_out_init,
                      BroadcastIfNeeded(value_init, shape));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp index_out_init,
                      BroadcastIfNeeded(index_init, shape));

  const auto body_builder = [carry_shape](
                                mlir::OpBuilder& op_builder, mlir::Location loc,
                                mlir::Value input_val, mlir::Value index,
                                mlir::ValueRange carries)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    mlir::MlirBuilder builder(op_builder, loc);
    TT_ASSIGN_OR_RETURN(
        const mlir::MlirOp input_idx_op,
        BroadcastIfNeeded(mlir::MlirOp(builder, index), carry_shape));
    const mlir::Value input_idx = input_idx_op.getValue();

    const mlir::Value carry_val = carries[0];
    const mlir::Value gt_pred = mlir::stablehlo::CompareOp::create(
                                    op_builder, loc, carry_val, input_val,
                                    mlir::stablehlo::ComparisonDirection::GT)
                                    .getResult();

    const mlir::Value selected_value =
        mlir::stablehlo::SelectOp::create(op_builder, loc, gt_pred, carry_val,
                                          input_val)
            .getResult();

    const mlir::Value carry_idx = carries[1];
    const mlir::Value argmax_index =
        mlir::stablehlo::SelectOp::create(op_builder, loc, gt_pred, carry_idx,
                                          input_idx)
            .getResult();

    return llvm::SmallVector<mlir::Value>{selected_value, argmax_index};
  };

  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(builder, input, normalized_dim,
                    {value_init_bcast, index_init_bcast},
                    {value_out_init, index_out_init}, std::move(body_builder)));

  return CummaxOutputs{.values = results[0], .indices = results[1]};
}

}  // namespace torch_tpu
