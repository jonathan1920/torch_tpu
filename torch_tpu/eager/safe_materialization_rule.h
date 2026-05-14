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

#include "absl/base/nullability.h"
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
//
// The return value is the set of all nodes that must be materialized, assuming
// one output node per Traversal.
absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
EnforceOrderedMaterialization(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_SAFE_MATERIALIZATION_RULE_H_
