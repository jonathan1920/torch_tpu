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

#ifndef TORCH_TPU_EAGER_MATERIALIZATION_HEURISTICS_H_
#define TORCH_TPU_EAGER_MATERIALIZATION_HEURISTICS_H_

#include <string_view>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

// TODO(b/452025916): add unit tests for materialization heuristics.

namespace torch_tpu {

// Interface for materialization heuristics. A heuristic looks at a
// given `Traversal` and decides which deferred ops in it should be
// materialized.
class MaterializationHeuristic {
 public:
  virtual ~MaterializationHeuristic() = default;

  // Returns a representative name for this heuristic. Used for
  // debugging/logging purposes. Must return the same value throughout the
  // lifetime of the object.
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // Returns whether this heuristic is enabled.
  [[nodiscard]] virtual bool Enabled() const = 0;

  // Updates a set of deferred ops in `traversal` that this heuristic decides to
  // materialize. While it's harmless, there's no need to have any output nodes
  // in the set, as output nodes are always materialized, regardless of the
  // heuristics.
  virtual void ApplyOn(
      const Traversal& traversal,
      absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          materialization_nodes) = 0;

 protected:
  MaterializationHeuristic() = default;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_MATERIALIZATION_HEURISTICS_H_
