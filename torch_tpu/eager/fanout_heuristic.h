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

#ifndef TORCH_TPU_EAGER_FANOUT_HEURISTIC_H_
#define TORCH_TPU_EAGER_FANOUT_HEURISTIC_H_

#include <string_view>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialization_heuristics.h"

namespace torch_tpu {

// Materializes all nodes whose fanout is >= 2.
//
// The fanout of a node is the number of its direct consumers. For example, in
//
//   i0    i1
//    |\    |
//    | \   |
//    |  x  y   x = f(i0)
//    | / \ |   y = g(i1)
//    |/   \|  o0 = h(i0, x)
//    o0   o1  o1 = k(x, y)
//
// i0 and i1 are the inputs of the graph, x and y are intermediate nodes,
// and o0 and o1 are the outputs. The fanouts of the nodes are:
//
//   i0: 2
//   i1: 1
//    x: 2
//    y: 1
//
// Therefore, the fanout heuristic will choose to materialize x.
//
// This includes dependencies both internal and external to a Traversal object.
// The fanout of a node is recorded directly on the DeferredOp (similar to a
// refcount); this means that even if only node `o0` is materialized, `x` will
// still be materialized by this heuristic as well, anticipating its future use
// in `o1`.
//
// The primary benefit of this heuristic is to avoid recomputing the same
// intermediate tensor multiple times.
//
// For example, in the graph above, the intermediate tensor x is needed to
// compute both o0 and o1. Without the heuristic, the operation that creates x
// might be duplicated — once for computing o0 and again for computing o1.
// This leads to redundant work. With the heuristic, x is materialized and
// reused for both o0 and o1 (in a single execution, or sequential executions).
class FanoutHeuristic : public MaterializationHeuristic {
 public:
  FanoutHeuristic() = default;

  // This class is neither copyable nor movable.
  FanoutHeuristic(const FanoutHeuristic&) = delete;
  FanoutHeuristic& operator=(const FanoutHeuristic&) = delete;
  FanoutHeuristic(FanoutHeuristic&&) = delete;
  FanoutHeuristic& operator=(FanoutHeuristic&&) = delete;

 protected:
  std::string_view Name() const override { return "FanoutHeuristic"; }
  bool Enabled() const override;
  void ApplyOnNode(const DeviceBufferList& node,
                   absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
                       materialization_nodes) override;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_FANOUT_HEURISTIC_H_
