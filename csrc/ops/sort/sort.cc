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

#include "csrc/ops/sort/sort.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

SortShloOutputs BuildSortShlo(mlir::MlirOp input_op, bool stable, int64_t dim,
                              bool descending) {
  const mlir::RankedTensorType inputType = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();

  // Track indices in i32 when the sorted dimension fits in int32,
  // otherwise fall back to i64 to preserve correctness for very large dims.
  // i32 indices are widened back to i64 to match PyTorch's convention.
  const bool fits_i32 =
      inputType.getShape()[dim] <= std::numeric_limits<int32_t>::max();
  const mlir::ElementType index_type =
      fits_i32 ? mlir::ElementType::I32 : mlir::ElementType::I64;

  mlir::MlirOp indices = stablehlo::Iota(
      builder,
      makeTensorType(builder.getContext(), inputType.getShape(), index_type),
      dim);
  auto comparator = [inputType, descending, fits_i32](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    const mlir::Type cmp_index_type =
        fits_i32 ? op_builder.getI32Type() : op_builder.getI64Type();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(inputType.getElementType())) {
      compare_type = "TOTALORDER";
    }
    stablehlo::buildSortComparisonBody(
        {inputType.getElementType(), cmp_index_type},
        descending ? stablehlo::ComparisonDirection::GT
                   : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  auto outputs =
      stablehlo::Sort(builder, {input_op, indices}, comparator, dim, stable);
  const mlir::MlirOp result_indices =
      fits_i32
          ? stablehlo::ConvertElementType(outputs[1], mlir::ElementType::I64)
          : outputs[1];
  return SortShloOutputs{.values = outputs[0], .indices = result_indices};
}

}  // namespace torch_tpu
