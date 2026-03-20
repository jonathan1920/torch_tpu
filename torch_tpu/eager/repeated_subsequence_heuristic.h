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

#ifndef TORCH_TPU_EAGER_REPEATED_SUBSEQUENCE_HEURISTIC_H_
#define TORCH_TPU_EAGER_REPEATED_SUBSEQUENCE_HEURISTIC_H_

#include <string_view>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialization_heuristics.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// Materializes all endpoints of repeated and non-overlapping subsequences.
//
// For example, in the graph
//
//   ... a -> x = f(a) -> y = g(x) -> z = f(y) -> u = g(z) -> ...
//
// the computation of y from a and the computation of u from y follow the same
// logic (f() and then g()). Therefore, this heuristic will materialize the
// endpoints of the first subsequence (a and y) and the endpoints of the second
// subsequence (y and u). This will create two subgraphs that are compiled
// and executed separately:
//
//   a -> x -> y
//   y -> z -> u
//
// In case these computations have the same cache key (i.e. they have the same
// input/output dtypes/shapes and parameters), we only need to compile them
// once, potentially reducing overall compilation time.
//
// Note that this heuristic currently only looks at the names of the ops when
// determining whether two computations are the same. The dtypes, shapes, and
// parameters are ignored. Hence it may materialize more than strictly needed.
class RepeatedSubsequenceHeuristic : public MaterializationHeuristic {
 public:
  RepeatedSubsequenceHeuristic() = default;

  // This class is neither copyable nor movable.
  RepeatedSubsequenceHeuristic(const RepeatedSubsequenceHeuristic&) = delete;
  RepeatedSubsequenceHeuristic& operator=(const RepeatedSubsequenceHeuristic&) =
      delete;
  RepeatedSubsequenceHeuristic(RepeatedSubsequenceHeuristic&&) = delete;
  RepeatedSubsequenceHeuristic& operator=(RepeatedSubsequenceHeuristic&&) =
      delete;

 protected:
  std::string_view Name() const override {
    return "RepeatedSubsequenceHeuristic";
  }
  bool Enabled() const override;
  void ApplyOn(const Traversal& traversal,
               absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
                   materialization_nodes) override;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_REPEATED_SUBSEQUENCE_HEURISTIC_H_
