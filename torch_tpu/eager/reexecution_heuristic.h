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

#ifndef TORCH_TPU_EAGER_REEXECUTION_HEURISTIC_H_
#define TORCH_TPU_EAGER_REEXECUTION_HEURISTIC_H_

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialization_heuristics.h"

namespace torch_tpu {

// Materialize all nodes that were part of a previous materialization pass.
//
// During iteration with persistent state, (most commonly a training loop),
// it's possible for tensors to hold DeferredOps that refer to previous loop
// iteration values. For example, printing the squares of integers:
// ```
//   a = torch.zeros(1)
//   for _ in range(1000):
//     a_sqr = a.sqr()
//     print(a_sqr.cpu())
//     a = a + 1
// ```
// This would cause `a_sqr` to be materialized on every iteration, but the value
// of `a` would end up being an ever-increasing chain of DeferredOps like
// `add(add(...add(zeros(1), 1), ...), 1), 1)` growing with the number of
// iterations, which causes repeated recompilation and out-of-memory errors.
//
// This heuristic would compute `sqr(zeros(1))` and flag `zeros(1)` as executed
// on the first iteration. The second iteration would initially traverse to
// `sqr(add(zeros(1), 1)))`, but would recognize that `zeros(1)` had already
// been computed, and split the execution to `zeros(1) -> sqr(add(0, 1))`.
// The third iteration would then initially traverse to `sqr(add(add(0, 1), 1))`
// but would recognize that `add(0, 1)` had already been computed, and split
// the execution to `add(0, 1) -> sqr(add(1, 1))`.
//
// This would stabilize to each iteration only computing `add(i-1, 1)` and
// `sqr(add(i, 1))`, which is optimal for both cache and memory usage.
class ReexecutionHeuristic : public MaterializationHeuristic {
 public:
  ReexecutionHeuristic() = default;

 protected:
  void ApplyOnNode(const DeviceBufferList& node,
                   absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
                       materialization_nodes) override;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_REEXECUTION_HEURISTIC_H_
