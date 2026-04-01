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

#include <memory>

#include "c10/util/ThreadLocalDebugInfo.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

namespace {

// The default SHLO precision.
constexpr auto kDefaultPrecision =        //
    mlir::stablehlo::Precision::DEFAULT;  // EXPLICIT_PRECISION_OK=root usage

// The DebugInfoKind slot to use for the precision context. For now,
// we can only use the TEST_INFO slot.
// TODO(https://github.com/pytorch/pytorch/issues/56027): use a dedicated slot.
constexpr auto kPrecisionSlot = c10::DebugInfoKind::TEST_INFO;

// The state for the precision context. It must inherit from c10::DebugInfoBase
// as required by the PyTorch thread-local-state system.
struct PrecisionContextState : public c10::DebugInfoBase {
  explicit PrecisionContextState(mlir::stablehlo::Precision p) : precision(p) {}
  mlir::stablehlo::Precision precision = kDefaultPrecision;
};

}  // namespace

mlir::stablehlo::Precision GetPrecision() {
  // Get the precision from the current python thread's context.
  const auto* const state = dynamic_cast<const PrecisionContextState*>(
      c10::ThreadLocalDebugInfo::get(kPrecisionSlot));
  return state ? state->precision : kDefaultPrecision;
}

void PushPrecision(mlir::stablehlo::Precision precision) {
  // _push sets the new state for the current python thread.
  c10::ThreadLocalDebugInfo::_push(
      kPrecisionSlot, std::make_shared<PrecisionContextState>(precision));
}

void PopPrecision() {
  // _pop removes the last pushed state for the current python thread.
  c10::ThreadLocalDebugInfo::_pop(kPrecisionSlot);
}

}  // namespace torch_tpu
