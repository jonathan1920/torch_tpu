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

#ifndef TORCH_TPU_COMMON_DYNAMISM_UTILS_H_
#define TORCH_TPU_COMMON_DYNAMISM_UTILS_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {

// Struct to hold a DeviceBufferRef and its inferred MLIR StableHLO dimensions.
struct DeviceRefDimensions {
  const DeviceBufferRef ref;
  mlir::stablehlo::Dimensions dims;
};

// Computes and returns the stablehlo::Dimensions for each output
// DeviceBufferRef based on inputs and execution order. This involves building a
// MLIR graph by processing the operations in execution order.
absl::StatusOr<std::vector<DeviceRefDimensions>> GetTraversalOutputDimensions(
    mlir::MLIRContext& mlir_context,
    const PythonContext* absl_nullable python_context,
    absl::Span<const DeviceBufferRef> inputs,
    absl::Span<const DeviceBufferRef> outputs,
    absl::Span<const std::shared_ptr<DeviceBufferList>> execution_order);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DYNAMISM_UTILS_H_
