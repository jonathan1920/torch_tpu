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

#ifndef TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_
#define TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_

#include <array>
#include <cstdint>

#include "absl/status/statusor.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Pads the dynamic dimension of the input buffer to the upper bound.
absl::StatusOr<DeviceBufferRef> PadDynamicDimension(DeviceBufferRef input,
                                                    int64_t dimension_index,
                                                    int64_t upper_bound);

// Sets the dynamic dimension size of the input buffer to the original size.
// Returns a pair of DeviceBufferRefs, the first is the input buffer with the
// dynamic dimension size set, the second is a buffer that contains the original
// dimension size.
// This has the effect of changing a statically shaped buffer to a dynamically
// shaped buffer.
absl::StatusOr<std::array<DeviceBufferRef, 2>> SetDynamicDimensionSize(
    DeviceBufferRef input, int64_t dimension_index,
    int64_t original_dimension_size);

}  // namespace torch_tpu

#endif  // TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_
