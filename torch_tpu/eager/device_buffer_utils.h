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

#ifndef TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_
#define TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_

#include <vector>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/eager/device_buffer.h"
namespace torch_tpu {

// Creates a deferred DeviceBufferList containing a constant tensor value,
// based on a 1D byte array.
// Under the hood, this uses `mlir::DenseElementsAttr::getFromRawBuffer`,
// with special handling for booleans and complex numbers (which have
// differing layout requirements between PyTorch and XLA).
// This value will be "baked into" the MLIR for XLA compilation, allowing for
// greater optimization potential.
// However, this means that any future execution which have a different value
// for the constant will require a re-compilation.
absl::StatusOr<DeviceBufferRef> CreateConstantDeviceBufferRef(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type);

// Creates a DeviceBufferList as if by using the torch.empty() operation
// with fill_uninitialized_memory=True.
// This will be a DeferredOp that fills the buffer with NaNs for floats
// (including complex floats), and max for integers and booleans.
absl::StatusOr<DeviceBufferRef> CreateEmptyDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type);

// Creates a DeviceBufferList with a constant value of "no data".
// Equivalent to CreateConstant with an empty cpu_tensor_data vector.
absl::StatusOr<DeviceBufferRef> CreateZeroSizeDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_
