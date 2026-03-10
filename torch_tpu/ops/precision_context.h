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

#ifndef TORCH_TPU_OPS_PRECISION_CONTEXT_H_
#define TORCH_TPU_OPS_PRECISION_CONTEXT_H_

#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

// Manages the thread-local precision configuration for StableHLO operations.
// This allows different threads to operate with different precision settings
// for matrix multiplications or convolutions, such as DEFAULT, HIGH, or
// HIGHEST.
class PrecisionContext {
 public:
  // Sets the precision for the current thread.
  static void SetPrecision(mlir::stablehlo::Precision precision);
  // Returns the precision for the current thread.
  [[nodiscard]] static mlir::stablehlo::Precision GetPrecision();
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_PRECISION_CONTEXT_H_
