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

#include "torch_tpu/eager/safe_materialization_rule.h"

#include <cstdint>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Returns the set of nodes that must be materialized.
// This includes:
// - All required outputs.
// - All nodes with dynamic dimensions.
// - All nodes required by OpSplitMode.
// - All nodes that have live tensors.
// This assumes the entire execution order is going to be materialized.
absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
GetMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs) {
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_points;
  for (const auto& node : execution_order) {
    const auto deferred_op = node->deferred_op();
    if (!deferred_op) continue;
    const auto split_mode = deferred_op->split_mode();

    if (IsSplitBefore(split_mode)) {
      for (const auto& input : deferred_op->inputs()) {
        if (input.is_deferred()) {
          materialization_points.insert(input.device_buffer_list().get());
        }
      }
    }

    if (IsSplitAfter(split_mode) || !node->is_stale() ||
        required_outputs.contains(node.get())) {
      materialization_points.insert(node.get());
      continue;
    }

    for (int i = 0; i < node->size(); ++i) {
      if (!node->dynamic_dimensions(i).empty()) {
        materialization_points.insert(node.get());
        break;
      }
    }
  }
  return materialization_points;
}

// Ensures that if a tensor is used after a materialization point, it is also
// materialized before the materialization point.
//
// For example, a function like:
// ```
//   x = foo()
//   y = bar()
//   z = baz(x)
//   print(y.item())
// ```
// will have:
//   execution_order: [foo, bar, baz]
//   required_outputs: {bar}
//   materialization_points: {bar}
// This function will add a materialization point for `x = foo()`, and will
// remove any materialization points after the last required output.
void SplitAllMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_points) {
  // The index of the node we're currently planning to materialize.
  // Will be nullopt for all nodes after the highest-index required output.
  std::optional<uint64_t> materializing_index = std::nullopt;

  for (auto node_it = execution_order.rbegin();
       node_it != execution_order.rend(); ++node_it) {
    const auto& node = *node_it;

    if (required_outputs.contains(node.get())) {
      // Found a required output, ensure that materializing_index has a value
      // and stop dropping materialization points.
      materializing_index = node->creation_index();
    } else if (!materializing_index.has_value()) {
      // Remove any materialization points after the last required output.
      materialization_points.erase(node.get());
    } else if (materialization_points.contains(node.get())) {
      // Retain any materialization points before the last required output.
      materializing_index = node->creation_index();
    } else if (node->last_child_index() > materializing_index.value()) {
      // This node has an edge from node->creation_index() to
      // node->last_child_index(), which crosses the current
      // materializing_index.
      // We need to materialize this node so that the materialization order of
      // {node->creation_index(), materializing_index, node->last_child_index()}
      // is maintained.
      materialization_points.insert(node.get());
      materializing_index = node->creation_index();
    }
  }
}

}  // namespace

absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
EnforceOrderedMaterialization(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs) {
  tsl::profiler::TraceMe t("EnforceOrderedMaterialization");
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_points = GetMaterializationPoints(
          traversal.execution_order(), required_outputs);
  SplitAllMaterializationPoints(traversal.execution_order(), required_outputs,
                                materialization_points);
  return materialization_points;
}

}  // namespace torch_tpu
