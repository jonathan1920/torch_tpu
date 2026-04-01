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

#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/dynamic_op_split_heuristic.h"
#include "torch_tpu/eager/forced_split_heuristic.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

void SafeMaterializationRule::InsertLiveEdges(const DeviceBufferList& node) {
  const auto* deferred_op = node.deferred_op();
  if (!deferred_op) return;
  for (const auto& input : deferred_op->inputs()) {
    if (input.deferred_op()) {
      live_edges_set.insert(input.device_buffer_list().get());
    }
  }
}

void SafeMaterializationRule::VisitNode(
    const DeviceBufferList& node,
    absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes) {
  const auto* deferred_op = node.deferred_op();
  if (!deferred_op) return;

  live_edges_set.erase(&node);
  AddMaterializations(node, materialization_nodes);
  InsertLiveEdges(node);
}

void SafeMaterializationRule::MaterializeLiveEdges(
    absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes) {
  for (const auto* edge : live_edges_set) {
    materialization_nodes.insert(edge);
  }
  live_edges_set.clear();
}

void SafeMaterializationRule::AddMaterializations(
    const DeviceBufferList& node,
    absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes) {
  if (required_outputs.contains(&node)) {
    materialization_nodes.insert(&node);
    found_required_output = true;
  }
  if (!found_required_output) {
    return;
  }

  DynamicOpSplitHeuristic(node, materialization_nodes);
  ForcedSplitHeuristic(node, materialization_nodes);
  if (!node.is_stale()) {
    materialization_nodes.insert(&node);
  }

  if (materialization_nodes.contains(&node)) {
    MaterializeLiveEdges(materialization_nodes);
  }
}

void SafeMaterializationRule::operator()(
    const Traversal& traversal,
    absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes) {
  tsl::profiler::TraceMe t("SafeMaterializationRule");
  // TODO(bawilson): pass in required outputs explicitly to differentiate
  // from leaf nodes used for DFS traversal.
  for (const auto& output : traversal.outputs()) {
    required_outputs.insert(output.device_buffer_list().get());
  }
  for (auto it = traversal.execution_order().rbegin();
       it != traversal.execution_order().rend(); ++it) {
    VisitNode(**it, materialization_nodes);
  }
}

}  // namespace torch_tpu
