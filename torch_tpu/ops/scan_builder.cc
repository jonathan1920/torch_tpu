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

#include "torch_tpu/ops/scan_builder.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace {

mlir::Value GetScanDimSize(mlir::MlirBuilder& builder, mlir::MlirOp input,
                           const int64_t scan_dim) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  if (!type.isDynamicDim(scan_dim)) {
    return MakeScalarConstant(builder, type.getShape()[scan_dim], i64)
        .getValue();
  }
  mlir::MlirOp size = mlir::stablehlo::GetDimensionSize(input, scan_dim);
  if (!size.getType().isInteger(64)) {
    size = mlir::stablehlo::ConvertElementType(size, i64);
  }
  return size.getValue();
}

llvm::SmallVector<mlir::MlirOp, 4> CreateStartIndices(
    mlir::MlirBuilder& builder, const mlir::Value index, const int64_t rank,
    const int64_t scan_dim) {
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  llvm::SmallVector<mlir::MlirOp, 4> indices(
      rank, MakeScalarConstant(builder, 0, i64));
  indices[scan_dim] = mlir::MlirOp(builder, index);
  return indices;
}

mlir::Value IncrementLoopIndex(mlir::MlirBuilder& builder,
                               const mlir::Location loc,
                               const mlir::Value index) {
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  return mlir::stablehlo::AddOp::create(
             builder.getOpBuilder(), loc, index,
             MakeScalarConstant(builder, 1, i64).getValue())
      .getResult();
}

struct ScanLoopState {
  int64_t scan_dim;
  llvm::SmallVector<mlir::Type> loop_types;
  llvm::SmallVector<mlir::Value> loop_inits;
  mlir::Location loc;
};

absl::StatusOr<ScanLoopState> PrepareScanLoop(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t dim,
    const llvm::ArrayRef<mlir::MlirOp> carry_inits,
    const llvm::ArrayRef<mlir::MlirOp> output_inits) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim,
                      SafeWrapDim(dim, GetTensorTypeOrDie(input).getRank()));

  TT_RET_CHECK(carry_inits.size() == output_inits.size(),
               error::kInvalidArgument)
      << "expected the number of carry inits (" << carry_inits.size()
      << ") and the number of output inits (" << output_inits.size()
      << ") to match";

  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType index_type =
      mlir::RankedTensorType::get({}, i64);

  llvm::SmallVector<mlir::Type> loop_types = {index_type};
  llvm::SmallVector<mlir::Value> loop_inits = {
      MakeScalarConstant(builder, 0, i64).getValue()};

  for (mlir::MlirOp c : carry_inits) {
    loop_types.push_back(c.getType());
    loop_inits.push_back(c.getValue());
  }
  for (mlir::MlirOp o : output_inits) {
    loop_types.push_back(o.getType());
    loop_inits.push_back(o.getValue());
  }

  return ScanLoopState{scan_dim, std::move(loop_types), std::move(loop_inits),
                       input.getValue().getLoc()};
}

void PopulateScanCondition(mlir::stablehlo::WhileOp& while_op,
                           mlir::MlirBuilder& builder, mlir::MlirOp input,
                           const int64_t scan_dim,
                           const llvm::SmallVector<mlir::Type>& loop_types,
                           const mlir::Location loc) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::Block* const block = op_builder.createBlock(&while_op.getCond());
  block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(block);

  const mlir::Value cond = mlir::stablehlo::CompareOp::create(
                               op_builder, loc, block->getArgument(0),
                               GetScanDimSize(builder, input, scan_dim),
                               mlir::stablehlo::ComparisonDirection::LT)
                               .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);
}

absl::Status PopulateScanBody(mlir::stablehlo::WhileOp& while_op,
                              mlir::MlirBuilder& builder, mlir::MlirOp input,
                              const int64_t scan_dim,
                              const llvm::SmallVector<mlir::Type>& loop_types,
                              const mlir::Location loc, const int num_carries,
                              const ScanBodyBuilder& body_builder) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::Block* const block = op_builder.createBlock(&while_op.getBody());
  block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(block);

  const mlir::Value index = block->getArgument(0);
  const mlir::ValueRange carries = block->getArguments().slice(1, num_carries);
  const mlir::ValueRange outputs = block->getArguments().slice(1 + num_carries);

  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);
  llvm::SmallVector<int64_t> slice_shape;
  for (const mlir::stablehlo::DimensionInfo& d : input_dims) {
    slice_shape.push_back(d.size);
  }
  slice_shape[scan_dim] = 1;

  const llvm::SmallVector<mlir::MlirOp, 4> start_indices =
      CreateStartIndices(builder, index, input_dims.size(), scan_dim);
  mlir::MlirOp slice =
      mlir::stablehlo::DynamicSlice(input, start_indices, slice_shape);

  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (i != scan_dim && type.isDynamicDim(i)) {
      mlir::MlirOp dim_size = mlir::stablehlo::GetDimensionSize(input, i);
      slice = mlir::stablehlo::SetDimensionSize(slice, dim_size, i);
    }
  }

  TT_ASSIGN_OR_RETURN(
      const llvm::SmallVector<mlir::Value> new_carries,
      body_builder(op_builder, loc, slice.getValue(), index, carries));

  TT_RET_CHECK(new_carries.size() == num_carries, error::kInvalidArgument)
      << "expected " << num_carries << " new carries, got "
      << new_carries.size();

  llvm::SmallVector<mlir::Value> next_state = {
      IncrementLoopIndex(builder, loc, index)};
  next_state.insert(next_state.end(), new_carries.begin(), new_carries.end());

  for (int i = 0; i < outputs.size(); ++i) {
    mlir::MlirOp out_op(builder, outputs[i]);
    mlir::MlirOp nc_op(builder, new_carries[i]);
    next_state.push_back(
        mlir::stablehlo::DynamicUpdateSlice(out_op, nc_op, start_indices)
            .getValue());
  }

  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t dim,
    const llvm::ArrayRef<mlir::MlirOp> carry_inits,
    const llvm::ArrayRef<mlir::MlirOp> output_inits,
    const ScanBodyBuilder body_builder) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim, SafeWrapDim(dim, type.getRank()));

  if (type.getShape()[scan_dim] == 0) {
    return DynamicMlirOpResults(output_inits.begin(), output_inits.end());
  }

  TT_ASSIGN_OR_RETURN(
      ScanLoopState state,
      PrepareScanLoop(builder, input, dim, carry_inits, output_inits));

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto while_op = mlir::stablehlo::WhileOp::create(
      op_builder, state.loc, state.loop_types, state.loop_inits);

  PopulateScanCondition(while_op, builder, input, state.scan_dim,
                        state.loop_types, state.loc);

  TT_RETURN_IF_ERROR(PopulateScanBody(while_op, builder, input, state.scan_dim,
                                      state.loop_types, state.loc,
                                      carry_inits.size(), body_builder));

  op_builder.setInsertionPointAfter(while_op);
  DynamicMlirOpResults results;
  results.reserve(output_inits.size());
  for (int i = 0; i < output_inits.size(); ++i) {
    results.push_back(
        mlir::MlirOp(builder, while_op.getResult(1 + carry_inits.size() + i)));
  }

  return results;
}

}  // namespace torch_tpu
