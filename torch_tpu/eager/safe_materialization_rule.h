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

#ifndef TORCH_TPU_EAGER_SAFE_MATERIALIZATION_RULE_H_
#define TORCH_TPU_EAGER_SAFE_MATERIALIZATION_RULE_H_

#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// Enforces the following rule:
//
// In order to materialize a node, all earlier nodes must first be either
// materialized, or dropped.
//
// Strict eager mode (kDeferNever) accomplishes this by materializing every
// node; but, we want to try to avoid breaking the graph into too many small
// traversals, as this hurts performance.
//
// We need to determine when it is safe to drop a node, and when it can't.
// A node can't be dropped if it is still live (i.e. has a c10::DataPtr, meaning
// it is still needed by aten) or is depended on by a node that will be
// be materialized later.
// We also can't drop a node if another correctness rule
// (DynamicOpSplitHeuristic or ForcedSplitHeuristic) requires us to materialize
// it.
//
// We start from a set of required outputs, and ensure that everything before
// them is materialized or dropped. We still need to consider all later tensors
// in the graph (retrieved via AddLeafNodes/Subgraph) because these may hold
// references to in-scope tensors, preventing them from being dropped.
struct SafeMaterializationRule {
  void operator()(
      const Traversal& traversal,
      absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes);

  // Reverse order visitor for the execution order. Because references are
  // output-to-input, going backwards ensures that by the time we visit a node,
  // we have the full picture of its outstanding references and can therefore
  // infer if it is droppable or not.
  void VisitNode(
      const DeviceBufferList& node,
      absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes);

  // After updating the live edges, add any materialization points, possibly for
  // both the given node and any inputs it has.
  void AddMaterializations(
      const DeviceBufferList& node,
      absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes);

  // Given a node k, insert all nodes i that it has an edge to.
  // If there's a materialization point after k but before i, then i will be
  // materialized.
  void InsertLiveEdges(const DeviceBufferList& node);

  // If there is an edge from node i to node k, and node j such that i < j < k
  // is materialized, then we must materialize node i.
  // This function is called when we identify some node j being materialized;
  // live_edges is populated by later nodes k pointing to earlier nodes i that
  // need to be materialized.
  void MaterializeLiveEdges(
      absl::flat_hash_set<const DeviceBufferList*>& materialization_nodes);

  // Condition to check when we enter the required execution region.
  // Flips to true at the first (highest creation index) required output.
  bool found_required_output = false;

  // If there is an edge from node i to node k, and node j such that i < j < k
  // is materialized, then we must materialize node i.
  // We iterate in reverse order over `k`s, building up a set of nodes `i` that
  // have dependency edges, clearing them as we go pass them as nodes `j`.
  // As soon as we encounter a materialization point j, we drain the set and
  // mark all nodes as materialization points.
  absl::flat_hash_set<const DeviceBufferList*> live_edges_set;

  // Input parameter; these nodes are guaranteed to be materialized.
  absl::flat_hash_set<const DeviceBufferList*> required_outputs;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_SAFE_MATERIALIZATION_RULE_H_
