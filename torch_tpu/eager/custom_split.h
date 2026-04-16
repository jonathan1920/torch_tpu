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

#ifndef TORCH_TPU_EAGER_CUSTOM_SPLIT_H_
#define TORCH_TPU_EAGER_CUSTOM_SPLIT_H_

#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// Returns a set of split points for the given traversal, based on a set of
// heuristics that can be enabled/disabled via flags.
absl::flat_hash_set<const DeviceBufferList*> CustomSplitRule(
    const Traversal& traversal);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_CUSTOM_SPLIT_H_
