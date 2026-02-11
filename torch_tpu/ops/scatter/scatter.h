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

#ifndef TORCH_TPU_OPS_SCATTER_SCATTER_H_
#define TORCH_TPU_OPS_SCATTER_SCATTER_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class ScatterOp { kReplace, kAdd, kMul };

[[nodiscard]] inline std::string FormatParamCacheKey(
    const ScatterOp scatter_op) {
  switch (scatter_op) {
    case ScatterOp::kReplace:
      return "replace";
    case ScatterOp::kAdd:
      return "add";
    case ScatterOp::kMul:
      return "multiply";
  }
}

absl::StatusOr<mlir::MlirOp> BuildScatterShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, mlir::MlirOp src,
    ScatterOp scatter_op, mlir::ElementType computation_element_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCATTER_SCATTER_H_
