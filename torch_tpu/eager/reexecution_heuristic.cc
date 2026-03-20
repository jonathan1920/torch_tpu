// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/eager/reexecution_heuristic.h"

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

void ReexecutionHeuristic::ApplyOnNode(
    const DeviceBufferList& node,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_nodes) {
  // Look for "boundary" nodes, where the input node has already been executed,
  // but the output node has not, and add that input to the materialization
  // set.
  //
  // For example, suppose we initially construct a graph like:
  //   a <-- b <-- c
  //         \
  //           <-- d
  // If we materialize `c`, then this will result in:
  //   a(has_been_executed) <-- b(has_been_executed) <-- d
  //
  // The flag for has_been_executed (see DeferredOp::has_been_executed() in
  // device_buffer.h) is flipped during materialization (see materialization.cc)
  // on all nodes except for the output nodes (as they will no longer have any
  // DeferredOp).
  //
  // If we then execute `d`, we note that there is a boundary between `b` and
  // `d`, where `b` has already been executed by `d` has not. This means that
  // `b` is the boundary of the previously-executed subgraph, so we materialize
  // it. But we don't materialize `a` since there are no direct dependencies
  // between `a` and `d`.
  const auto* child_deferred_op = node.deferred_op();
  ABSL_CHECK(child_deferred_op)  // CRASH_OK
      << "Found traversal node that's not a deferred op. This is a torch_tpu "
         "bug.";
  if (!child_deferred_op->has_been_executed()) {
    for (const auto& input : child_deferred_op->inputs()) {
      const SharedDeviceBufferList& input_list = input.device_buffer_list();
      const auto* parent_deferred_op = input_list->deferred_op();
      if (parent_deferred_op && parent_deferred_op->has_been_executed()) {
        // Boundary node; child_deferred_op has not been executed, but
        // parent_deferred_op has.
        materialization_nodes.insert(input_list.get());
      }
    }
  }
}

}  // namespace torch_tpu
