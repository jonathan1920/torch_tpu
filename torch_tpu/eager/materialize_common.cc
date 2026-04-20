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

#include "torch_tpu/eager/materialize_common.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {
namespace {

absl::Status SetOutputNodesAsMaterialized(std::vector<DeviceBufferRef>& outputs,
                                          PjRtBufferPointers results) {
  tsl::profiler::TraceMe trace("SetOutputNodesAsMaterialized");
  ABSL_VLOG(1) << "[SetOutputNodesAsMaterialized] Starting";
  PjRtBufferPointers buffers_to_assign;
  buffers_to_assign.reserve(outputs.size());
  auto i = 0;
  while (i < outputs.size()) {
    ABSL_VLOG(1) << "[AssignMaterializedBuffers] Assigning materialized "
                    "buffers for node "
                 << i;
    auto* absl_nonnull node = outputs[i].device_buffer_list().get();
    buffers_to_assign.clear();
    buffers_to_assign.reserve(node->size());
    if (i + node->size() > results.size()) {
      ABSL_VLOG(1)
          << "[AssignMaterializedBuffers] Not enough results to materialize "
             "node "
          << node;
      return TT_ERROR(error::kFailedPrecondition)
             << "Not enough results to materialize node " << node;
    }
    for (auto j = 0; j < node->size(); ++j) {
      ABSL_VLOG(1)
          << "[AssignMaterializedBuffers] Assigning materialized buffer for "
             "node "
          << node << " index " << j;
      buffers_to_assign.push_back(std::move(results[i + j]));
    }
    TT_RETURN_IF_ERROR(node->SetAsMaterialized(std::move(buffers_to_assign)))
        << "Failed to set node " << node << " as materialized";
    i += node->size();
  }
  ABSL_VLOG(1) << "[AssignMaterializedBuffers] Assigned materialized buffers "
                  "for nodes";
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::PjRtBuffer* absl_nullable>> GetArgumentBuffers(
    absl::Span<const DeviceBufferRef> arguments) {
  std::vector<xla::PjRtBuffer*> root_args;
  for (const auto&& [index, argument] : llvm::enumerate(arguments)) {
    switch (argument.state()) {
      case DeviceBufferRefState::kMaterialized: {
        ABSL_VLOG(1) << "kMaterialized DeviceBufferRef index: " << index;
        TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * pjrt_buffer,
                            argument.GetOrMaterializeBuffer());
        root_args.push_back(pjrt_buffer);
        break;
      }
      case DeviceBufferRefState::kPlaceholder:
        return TT_ERROR(error::kInternal)
               << "Materialize was called on a placeholder tensor. This "
                  "should never happen.\nkPlaceholder tensors should only "
                  "appear in compiled mode, which should never try to "
                  "materialize tensors."
               << argument.DebugString();

      case DeviceBufferRefState::kDeferred:
        return TT_ERROR(error::kInternal)
               << "Traversal input is unexpectedly deferred";
      default:
        return TT_ERROR(error::kInternal)
               << "Traversal input has unknown state";
    }
  }
  return root_args;
}

}  // namespace

// PRECONDITION: executables must be a sequence that is compatible with the
// arguments and composable, as they will be executed in order and the outputs
// of an executable are the arguments to the next. The final execuatable returns
// a complete set of outputs for each node in sequential blocks. That is, either
// all outputs of a node appear contiguously and in the same order as in
// results, or none of them do. Furthermore, the order of nodes must be
// consistent between the execution and results. For example, if node A is size
// 3, node B is size 1, and node C is size 2, then the output order must be [A0,
// A1, A2, B1, C0, C1]. This is established by all Materialize() functions in
// this file.
absl::Status ExecuteMaterializationJob(
    absl::Span<const DeviceBufferRef> arguments,
    absl::Span<const DeviceBufferRef> outputs,
    std::vector<SharedLoadedExecutableWithMetadata> executables,
    std::string_view task_name) {
  tsl::profiler::TraceMe trace("ExecuteMaterializationJob");
  ABSL_VLOG(1) << "[ExecuteMaterializationJob]: task_name=" << task_name
               << " input arg count: " << arguments.size()
               << " output arg count: " << outputs.size()
               << " executables count: " << executables.size();
  ABSL_CHECK(!executables.empty()) << "No executables to execute";  // CRASH_OK

  TT_ASSIGN_OR_RETURN(std::vector<xla::PjRtBuffer*> argument_buffers,
                      GetArgumentBuffers(arguments));

  ABSL_VLOG(1)
      << "[ExecuteMaterializationJob]: Arguments materialization completed "
         "for task name="
      << task_name << ", argument_buffers size: " << argument_buffers.size();

  std::vector<PjRtBufferPointers> intermediate_results;
  for (auto& executable : executables) {
    TT_ASSIGN_OR_RETURN(PjRtBufferPointers results,
                        Execute(executable, std::move(argument_buffers)),
                        _.SetPrepend()
                            << "failed to enqueue execution for task_name="
                            << task_name << ": ");
    argument_buffers.clear();
    argument_buffers.reserve(results.size());
    for (const auto& result : results) {
      argument_buffers.push_back(result.get());
    }
    intermediate_results.push_back(std::move(results));
  }
  PjRtBufferPointers final_results = std::move(intermediate_results.back());

  ABSL_VLOG(1)
      << "[ExecuteMaterializationJob]: Enqueued execution for task_name="
      << task_name << ", results size: " << final_results.size();

  ABSL_VLOG(1) << "[ExecuteMaterializationJob]: Materialization enqueue has "
                  "completed for task_name="
               << task_name;

  std::vector<DeviceBufferRef> output_refs;
  for (const auto& output : outputs) {
    output_refs.push_back(output);
  }

  return SetOutputNodesAsMaterialized(output_refs, std::move(final_results));
}

}  // namespace torch_tpu
