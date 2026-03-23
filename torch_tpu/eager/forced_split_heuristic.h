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

#ifndef TORCH_TPU_EAGER_FORCED_SPLIT_HEURISTIC_H_
#define TORCH_TPU_EAGER_FORCED_SPLIT_HEURISTIC_H_

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Materializes all nodes that are marked as split points.

void ForcedSplitHeuristic(
    const DeviceBufferList& node,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_nodes);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_FORCED_SPLIT_HEURISTIC_H_
