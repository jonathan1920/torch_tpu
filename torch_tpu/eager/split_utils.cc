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

#include "torch_tpu/eager/split_utils.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

absl::StatusOr<std::vector<Traversal>> ApplySplitPoints(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points) {
  std::vector<Traversal> traversals;
  for (const auto& node : traversal.execution_order()) {
    if (!split_points.contains(node.get())) continue;
    TT_ASSIGN_OR_RETURN(auto new_traversal,
                        Traversal::Create({node}, split_points));
    traversals.push_back(std::move(new_traversal));
  }
  return traversals;
}

}  // namespace torch_tpu
