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

#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

namespace {

mlir::stablehlo::Precision& GetCurrentPrecision() {
  static thread_local mlir::stablehlo::Precision current_precision =
      mlir::stablehlo::Precision::DEFAULT;
  return current_precision;
}

}  // namespace

void PrecisionContext::SetPrecision(mlir::stablehlo::Precision precision) {
  GetCurrentPrecision() = precision;
}

mlir::stablehlo::Precision PrecisionContext::GetPrecision() {
  return GetCurrentPrecision();
}

}  // namespace torch_tpu
