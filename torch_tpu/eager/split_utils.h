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

#ifndef TORCH_TPU_EAGER_SPLIT_UTILS_H_
#define TORCH_TPU_EAGER_SPLIT_UTILS_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// Creates a new set of traversals, one for each split point in the original
// traversal.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
ApplySplitPoints(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_SPLIT_UTILS_H_
