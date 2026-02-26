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

#ifndef TORCH_TPU_EAGER_STALE_HEURISTIC_H_
#define TORCH_TPU_EAGER_STALE_HEURISTIC_H_

#include <string_view>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialization_heuristics.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// Materialize at the boundary point of each "stale" region of a graph.
//
// A DeviceBufferRef is "stale" if there is no live c10::StorageImpl using it;
// stale buffers are only being kept alive due to being used by other later
// DeferredOps.
//
// If a buffer is stale, then the user will never materialize it (cannot call
// `tensor.cpu()` if `tensor` does not exist), and will also never dispatch a
// new DeferredOp using that buffer (cannot call `aten::foo(tensor)` without
// `tensor` existing).
//
// This is also able to capture the effect of PyTorch in-place ops, like
// `a.add_(b)`; this will cause the old buffer for `a` to be marked as
// stale, while the new `a` is live.
//
// If we materialize all live DeferredOps that directly depend on a stale
// buffer, then all of these nodes will be immediately freed, and we will avoid
// any possible re-execution with these stale buffers. (Re-execution is still
// possible for live buffers).
class StaleHeuristic : public MaterializationHeuristic {
 public:
  StaleHeuristic() = default;

  // This class is neither copyable nor movable.
  StaleHeuristic(const StaleHeuristic&) = delete;
  StaleHeuristic& operator=(const StaleHeuristic&) = delete;
  StaleHeuristic(StaleHeuristic&&) = delete;
  StaleHeuristic& operator=(StaleHeuristic&&) = delete;

 protected:
  std::string_view Name() const override { return "StaleHeuristic"; }
  bool Enabled() const override;
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull> ApplyOn(
      const Traversal& traversal) override;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_STALE_HEURISTIC_H_
