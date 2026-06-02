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

#include "torch_tpu/ops/precision_context.h"

#include <string_view>

#include "absl/log/absl_check.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/context_manager.h"

namespace torch_tpu {

// Returns the current python thread's precision. It's unsafe in the sense that
// it doesn't ensure that the precision is included in an op's cache key.
[[nodiscard]] mlir::stablehlo::Precision UnsafeGetPrecision() {
  return GetContextState<mlir::stablehlo::Precision>().value_or(
      mlir::stablehlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=root usage
}

[[nodiscard]] mlir::stablehlo::Precision GetAndAddPrecisionTo(
    OpParamCacheKeys& param_keys, const std::string_view param_name) {
  const auto precision = UnsafeGetPrecision();
  ABSL_CHECK_OK(param_keys.SetParam(param_name, precision));  // CRASH_OK
  return precision;
}

}  // namespace torch_tpu
