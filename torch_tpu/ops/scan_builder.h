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

#ifndef TORCH_TPU_OPS_SCAN_BUILDER_H_
#define TORCH_TPU_OPS_SCAN_BUILDER_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// (op_builder, loc, input_slice, index_0d, carries) -> new_carries.
using ScanBodyBuilder =
    absl::AnyInvocable<absl::StatusOr<llvm::SmallVector<mlir::Value>>(
        mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value,
        mlir::ValueRange) const>;

// Lowers a prefix scan along 'dim' using a StableHLO while loop.
absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input, int64_t dim,
    llvm::ArrayRef<mlir::MlirOp> carry_inits,
    llvm::ArrayRef<mlir::MlirOp> output_inits, ScanBodyBuilder body_builder);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCAN_BUILDER_H_
