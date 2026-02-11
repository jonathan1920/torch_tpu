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

#ifndef TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_
#define TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Marks a dimension of the given tensor as dynamic.
// The dimension must be in bounds for the tensor.
// The lower and upper bounds must be in bounds for the dimension as well.
// This operation, if successful, adds the following operations
// to the computation graph:
//   0. Extract the DeviceBufferRef associated to the tensor.
//   1. Pads it to the upper bound in the dynamic dimension.
//   2. Builds the original size of the dynamic dimension as a DeviceBufferRef.
//   3. Call set_dimension_size on the padded tensor with the runtime size.
//   4. Reassign the tensor with the result of step 3.
// We also materialize step 1, so that the inputs to step 3 are oblivious to
// the runtime size of the dynamic dimension.
absl::Status MarkDynamic(const at::Tensor& tensor, int64_t dimension,
                         int64_t lower_bound, int64_t upper_bound);

absl::StatusOr<absl::Span<const BoundedDynamicDimension>> GetDynamismInfo(
    const at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_
