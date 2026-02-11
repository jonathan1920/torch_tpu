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

#include "torch_tpu/eager/dynamic_op_split_heuristic.h"

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
DynamicOpSplitHeuristic::ApplyOn(const Traversal& traversal) {
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_nodes;
  for (const auto& node : traversal.execution_order()) {
    bool is_dynamic = false;
    for (int i = 0; i < node->size(); ++i) {
      if (!node->dynamic_dimensions(i).empty()) {
        is_dynamic = true;
        break;
      }
    }
    if (is_dynamic) {
      materialization_nodes.insert(node.get());
    }
  }
  return materialization_nodes;
}

}  // namespace torch_tpu
