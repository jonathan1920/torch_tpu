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

#include "torch_tpu/eager/split_traversal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/dynamic_op_split_heuristic.h"
#include "torch_tpu/eager/fanout_heuristic.h"
#include "torch_tpu/eager/forced_split_heuristic.h"
#include "torch_tpu/eager/reexecution_heuristic.h"
#include "torch_tpu/eager/repeated_subsequence_heuristic.h"
#include "torch_tpu/eager/stale_heuristic.h"
#include "torch_tpu/eager/traversal.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(bool, torch_tpu_internal_fanout_heuristic, true,
          "Use a materialization heuristic that looks at node fanout.");
ABSL_FLAG(bool, torch_tpu_internal_reexecution_heuristic, true,
          "Use a materialization heuristic that materializes on reexecution.");
ABSL_FLAG(bool, torch_tpu_internal_repeated_subsequence_heuristic, false,
          "Use a heuristic that materializes endpoints of repeated "
          "subsequences.");
ABSL_FLAG(bool, torch_tpu_internal_stale_heuristic, true,
          "Use a materialization heuristic that materializes around the stale "
          "regions of a graph.");

namespace torch_tpu {

namespace {

// Config struct indicating which materialization heuristics are enabled.
struct EnabledHeuristics {
  void Initialize() {
    reexecution = absl::GetFlag(FLAGS_torch_tpu_internal_reexecution_heuristic);
    fanout = absl::GetFlag(FLAGS_torch_tpu_internal_fanout_heuristic);
    repeated_subsequence =
        absl::GetFlag(FLAGS_torch_tpu_internal_repeated_subsequence_heuristic);
    stale = absl::GetFlag(FLAGS_torch_tpu_internal_stale_heuristic);
    initialized = true;
  }

  // Whether the heuristics have been initialized.
  bool initialized = false;
  bool reexecution = false;
  bool forced_split = true;      // always enabled
  bool dynamic_op_split = true;  // always enabled
  bool fanout = false;
  bool repeated_subsequence = false;
  bool stale = false;
};

// Applies all enabled materialization heuristics on a given `traversal` and
// return the set of nodes in `traversal` that at least one heuristic decides
// to materialize.
[[nodiscard]] absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
ApplyAllMaterializationHeuristicsOn(const Traversal& traversal) {
  static EnabledHeuristics enabled_heuristics;
  if (!enabled_heuristics.initialized) {
    enabled_heuristics.Initialize();
  }

  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      nodes_to_materialize;
  {
    tsl::profiler::TraceMe t("LocalHeuristics");
    for (const auto& node : traversal.execution_order()) {
      if (enabled_heuristics.reexecution) {
        ReexecutionHeuristic(*node, nodes_to_materialize);
      }
      if (enabled_heuristics.forced_split) {
        ForcedSplitHeuristic(*node, nodes_to_materialize);
      }
      if (enabled_heuristics.dynamic_op_split) {
        DynamicOpSplitHeuristic(*node, nodes_to_materialize);
      }
      if (enabled_heuristics.fanout) {
        FanoutHeuristic(*node, nodes_to_materialize);
      }
      if (enabled_heuristics.stale) {
        StaleHeuristic(*node, nodes_to_materialize);
      }
    }
  }
  if (enabled_heuristics.repeated_subsequence) {
    tsl::profiler::TraceMe t("RepeatedSubsequenceHeuristic");
    RepeatedSubsequenceHeuristic(traversal, nodes_to_materialize);
  }

  // If an output is also a live boundary node, we don't need to redundantly
  // return it as a materialization node.
  for (const auto& output : traversal.outputs()) {
    nodes_to_materialize.erase(output.device_buffer_list().get());
  }

  return nodes_to_materialize;
}

// Sorts a list of objects in-place according to a given permutation.
// sort_order[i] = j indicates that objects[i] should be moved to position j
// after the function returns.
template <typename T>
void ApplySortOrderInplace(absl::Span<T> objects,
                           absl::Span<int64_t> sort_order) {
  ABSL_CHECK_EQ(sort_order.size(), objects.size());  // CRASH_OK
  for (int64_t i = 0; i < sort_order.size(); ++i) {
    // Find the first object in the next permutation cycle.
    if (sort_order[i] == -1) continue;  // already repositioned
    if (sort_order[i] == i) {
      // already in the right place, mark it complete
      sort_order[i] = -1;
      continue;
    }
    // Found a non-trivial permutation cycle.
    // Example: objects = [D, A, C, B], sort_order = [3, 0, 2, 1]
    // initialize with start = 0, dest = 3
    // Swap D, B. objects = [B, A, C, D], sort_order[3] = -1, dest = 1
    // Swap B, A. objects = [A, B, C, D], sort_order[1] = -1, dest = 0
    // dest=start=0, sort_order[0] = -1
    // Finish: objects = [A, B, C, D], sort_order = [-1, -1, 2, -1]
    const int64_t start = i;
    int64_t dest = sort_order[i];
    while (dest != i) {
      std::swap(objects[start], objects[dest]);
      const int64_t next = sort_order[dest];
      sort_order[dest] = -1;
      dest = next;
    }
    sort_order[start] = -1;
  }
}

// Reorders the traversals to prefer materializing the earliest-dispatched
// output DeferredOps first.
//
// A list of Traversals must be topologically sorted to be executable;
// otherwise, the enqueue process will deadlock as a traversal will wait for
// an input that isn't available.
//
// However, topological sorts are not unique. If traversal C depends on the
// outputs of traversals A and B, then either [A, B, C] or [B, A, C] are both
// valid topological sortings, and produce the same logical results at the end.
//
// The decision of which ordering to use is therefore only a question of
// performance optimization; we want to minimize the peak memory requirement
// during the execution of the sequence. Solving for the optimal ordering
// explicitly is challenging (though not impossible).
//
// But, if we assume that the user's dispatch order is optimized for true
// op-by-op eager execution (as is the case on PyTorch/CUDA eager), then
// aligning to this order should produce the optimal peak memory usage.
void ReorderByCreationIndex(std::vector<Traversal>& traversals) {
  if (traversals.size() <= 1) {
    // Nothing to do.
    return;
  }
  tsl::profiler::TraceMe t("ReorderByCreationIndex");

  using TraversalIndex = int64_t;

  // A map from one TraversalIndex to all other indexes that depend on it.
  // dependencies_out[i] = {j, ...} means that {traversals[j], ...} all depend
  // on the output node of traversals[i].
  // We assume most Traversals will have <= 2 dependencies, so we use
  // an `InlinedVector` with capacity 2 to reduce heap allocations.
  std::vector<absl::InlinedVector<TraversalIndex, 2>> dependencies_out(
      traversals.size());
  // The number of prior Traversals that each TraversalIndex is waiting on.
  // When dependencies_in[i] == 0, then traversals[i] is ready to be enqueued.
  std::vector<int64_t> dependencies_in(traversals.size());  // INT_VEC_OK

  // Helper lambda to get the creation index of the indexed Traversal.
  const auto traversal_creation_index = [&](TraversalIndex index) {
    const auto* deferred_op = traversals[index].outputs()[0].deferred_op();
    if (!deferred_op) {
      // This should never happen, but if it does, a Traversal with no deferred
      // outputs can be immediately materialized.
      return std::numeric_limits<int64_t>::min();
    }
    return deferred_op->creation_index();
  };
  // Compare function for std::priority_queue, to order by creation index
  // (lower index = higher priority = "greater" in the max-heap).
  const auto early_traversals_first = [&](TraversalIndex lhs,
                                          TraversalIndex rhs) {
    return traversal_creation_index(lhs) > traversal_creation_index(rhs);
  };
  std::priority_queue<TraversalIndex, std::vector<TraversalIndex>,
                      decltype(early_traversals_first)>
      pq(early_traversals_first);

  // Initialize edge list and priority queue.
  {
    // Temporary map to translate from DeviceBufferList outputs to the
    // corresponding TraversalIndex. Only needed during initialization.
    absl::flat_hash_map<const DeviceBufferList*, TraversalIndex>
        output_to_index;
    // One DeferredOp can depend on the same DeviceBufferList output multiple
    // times, but we only need to treat this as a single edge between
    // DeferredOps.
    absl::flat_hash_set<const DeviceBufferList*> deduped_inputs;
    for (TraversalIndex i = 0; i < traversals.size(); ++i) {
      const auto& traversal = traversals[i];
      deduped_inputs.clear();
      for (const auto& input : traversal.inputs()) {
        const auto* input_node = input.device_buffer_list().get();
        if (!deduped_inputs.insert(input_node).second) continue;
        if (const auto it = output_to_index.find(input_node);
            it != output_to_index.end()) {
          dependencies_out[it->second].push_back(i);
          dependencies_in[i]++;
        }
      }
      output_to_index[traversal.outputs()[0].device_buffer_list().get()] = i;
      if (dependencies_in[i] == 0) {
        pq.push(i);
      }
    }
  }

  // We want to avoid creating a second `std::vector<Traversal>` as a Traversal
  // can be quite large. Instead we record the destination indices of the
  // priority queue, and then sort the Traversals themselves into the final
  // order once the full map is populated.
  std::vector<TraversalIndex> traversal_to_final_index(traversals.size());

  // Sequentially pop from the priority queue to populate the sorted list.
  // As each Traversal is enqueued, it may make other Traversals ready, which
  // are then added to the priority queue.
  int64_t num_swaps = 0;  // Debug logging only, see below
  TraversalIndex final_index = 0;
  while (!pq.empty()) {
    const TraversalIndex i = pq.top();
    pq.pop();
    if (ABSL_VLOG_IS_ON(1) && i < final_index) {
      num_swaps += (final_index - i);
    }
    traversal_to_final_index[i] = final_index++;
    for (const auto j : dependencies_out[i]) {
      if (--dependencies_in[j] == 0) {
        pq.push(j);
      }
    }
  }
  if (ABSL_VLOG_IS_ON(1)) {
    // The Kendall "tau" rank correlation coefficient measures the similarity
    // between two orderings. 1.0 indicates the same ordering, -1.0 indicates a
    // complete reversal, and 0.0 indicates no correlation between the two.
    // This is computed as 1 - 4 * num_swaps / (n * (n - 1)), where num_swaps is
    // the number of swaps that a bubble sort algorithm would perform to sort
    // one list into the other's order.
    // See https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    double kendall_tau = static_cast<double>(num_swaps);
    kendall_tau /= traversals.size();
    kendall_tau /= (traversals.size() - 1);
    kendall_tau *= 4.0;
    kendall_tau = 1.0 - kendall_tau;
    ABSL_VLOG(1) << ">>> ReorderByCreationIndex: reordered "
                 << traversals.size() << " traversals, correlation of "
                 << kendall_tau;
  }
  ApplySortOrderInplace(absl::MakeSpan(traversals),
                        absl::MakeSpan(traversal_to_final_index));
}

}  // namespace

absl::StatusOr<std::vector<Traversal>> SplitTraversal(Traversal traversal) {
  ABSL_VLOG(1) << ">>> SplitTraversal " << traversal.execution_order().size();
  tsl::profiler::TraceMe t("SplitTraversal");

  std::vector<Traversal> traversals;

  // Here we collect a set of materialization nodes internal to the input
  // `traversal`.
  auto materialization_nodes_set =
      ApplyAllMaterializationHeuristicsOn(traversal);
  ABSL_VLOG(1) << "Found " << materialization_nodes_set.size()
               << " materialization nodes";

  if (materialization_nodes_set.empty()) {
    // Shortcut in the simple case where there is no internal materialization
    // node.
    traversals.push_back(std::move(traversal));
    return traversals;
  }

  // In the following code, we split the original traversal into multiple
  // traversals, based on the identified materialization points.

  // We add traversal outputs to the materialization nodes -- they do need to be
  // materialized.
  for (const auto& node_ref : traversal.outputs()) {
    const auto& node = node_ref.device_buffer_list();
    materialization_nodes_set.insert(node.get());
  }

  // Split traversal into one node per materialization point, maintaining the
  // DFS execution order. This is a valid ordering, but may not be optimal.
  std::vector<DeviceBufferRef> traversal_outputs;
  for (const auto& node : traversal.execution_order()) {
    if (!materialization_nodes_set.contains(node.get())) continue;
    traversal_outputs.clear();
    for (auto i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto buffer_ref, DeviceBufferRef::Create(node, i));
      traversal_outputs.push_back(std::move(buffer_ref));
    }
    TT_ASSIGN_OR_RETURN(auto new_traversal,
                        Traversal::Create(std::move(traversal_outputs),
                                          materialization_nodes_set));
    traversals.push_back(std::move(new_traversal));
  }
  // Reorder the traversals to prefer materializing the earliest-dispatched
  // output DeferredOps first. This is another topological sort which is
  // expected to be better, assuming the user's code is organized for eager
  // execution.
  ReorderByCreationIndex(traversals);
  return traversals;
}

}  // namespace torch_tpu
