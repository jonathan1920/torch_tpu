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

#include "torch_tpu/eager/repeated_subsequence_heuristic.h"

#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "torch_tpu/common/repeated_subsequence.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_names.h"

ABSL_FLAG(bool, torch_tpu_internal_repeated_subsequence_heuristic, false,
          "Use a heuristic that materializes endpoints of repeated "
          "subsequences.");
ABSL_FLAG(int, torch_tpu_internal_repeated_subsequence_heuristic_min_length, 10,
          "Minimal length for a subsequence to be considered by the repeated "
          "subsequence heuristic.");

namespace torch_tpu {

bool RepeatedSubsequenceHeuristic::Enabled() const {
  return absl::GetFlag(FLAGS_torch_tpu_internal_repeated_subsequence_heuristic);
}

absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
RepeatedSubsequenceHeuristic::ApplyOn(const Traversal& traversal) {
  ABSL_VLOG(1) << Name() << "::ApplyOn";

  std::vector<OpName> op_names;
  op_names.reserve(traversal.execution_order().size());
  for (const auto& node : traversal.execution_order()) {
    const DeferredOp* const deferred_op = node->deferred_op();
    ABSL_CHECK(deferred_op)  // CRASH_OK
        << "Found traversal node that's not a deferred op. This is a torch_tpu "
           "bug.";
    op_names.push_back(deferred_op->op_name());
  }

  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_nodes;

  // Below we look for repeated subsequences in the original sequence of
  // ops. Note however that for simplicity (and speed) we look only at op names,
  // i.e., we don't check if the identified subsequences are functionally
  // equivalent as doing so would be very expensive. Fortunately, that is a
  // benign approximation since in case of a false positive this heuristic would
  // only materialize more ops than necessary, but it would not lead to invalid
  // model execution.
  const int min_subseq_len = absl::GetFlag(
      FLAGS_torch_tpu_internal_repeated_subsequence_heuristic_min_length);
  const auto repeats =
      FindRepeatedSubsequences(absl::Span<OpName>(op_names), min_subseq_len);
  if (repeats.empty()) {
    return materialization_nodes;
  }

  const auto exec_order = traversal.execution_order();
  for (const auto& repeat : repeats) {
    const auto offsets =
        FindRepetitionOffsets(absl::Span<OpName>(op_names), repeat);
    ABSL_VLOG(2) << ">>> FOUND REPEATED SUBSEQUENCE size=" << repeat.size()
                 << " offsets=" << ToString(offsets);
    for (const auto offset : offsets) {
      materialization_nodes.insert(exec_order[offset].get());
      materialization_nodes.insert(
          exec_order[offset + repeat.size() - 1].get());
    }
  }

  return materialization_nodes;
}

}  // namespace torch_tpu
