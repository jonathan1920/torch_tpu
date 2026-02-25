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
#include "torch_tpu/_internal/sync/sync.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/traversal.h"
#include "xla/hlo/translate/register.h"

namespace torch_tpu {

using BufferRefToVarMap = absl::flat_hash_map<DeviceBufferRef, std::string>;

absl::Status SynchronizeTensors(absl::Span<const at::Tensor> tensors) {
  // Materialize the tensors and get their DeviceBufferRefs.
  // GetMaterialized() will materialize each base tensor and each view.
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> buffer_refs,
                      GetMaterialized(tensors));

  // Wait for all of the PjRtBuffers to be ready.
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    // GetMaterialized() should always return a materialized or zero-sized
    // buffer ref.
    // If a buffer ref is zero-sized, we might not have anything to wait for.
    TT_ASSIGN_OR_RETURN(auto& pjrt_buffer, buffer_ref.buffer());
    auto future = pjrt_buffer.GetReadyFuture();
    TT_RETURN_IF_ERROR(future.Await());
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> IsMaterialized(const at::Tensor& tensor) {
  // Since view tensors are ephemeral and re-materialized every time, there's
  // no point in checking whether the view is materialized.
  // Instead, we check the base buffer.
  TT_ASSIGN_OR_RETURN(auto base_buffer_ref, GetBaseBufferFromAtTensor(tensor));
  // A zero-sized buffer is considered materialized.
  return (base_buffer_ref.state() == DeviceBufferRefState::kMaterialized ||
          base_buffer_ref.state() == DeviceBufferRefState::kZeroSize);
}

absl::StatusOr<bool> IsReady(const at::Tensor& tensor) {
  // Since view tensors are ephemeral and re-materialized every time, there's
  // no point in checking whether the view is ready.
  // Instead, we check the base buffer.
  TT_ASSIGN_OR_RETURN(auto buffer_ref, GetBaseBufferFromAtTensor(tensor));
  if (buffer_ref.state() == DeviceBufferRefState::kZeroSize) {
    // A zero-sized buffer is considered ready.
    return true;
  }
  TT_ASSIGN_OR_RETURN(auto& maybe_pjrt_buffer, buffer_ref.buffer());
  return maybe_pjrt_buffer.GetReadyFuture().IsReady();
}

namespace {

absl::StatusOr<Traversal> GetTraversal(
    absl::Span<const DeviceBufferRef> buffer_refs) {
  TT_RET_CHECK(!buffer_refs.empty(), error::kInvalidArgument)
      << "tensors must not be empty";
  absl::flat_hash_set<SharedDeviceBufferList> nodes_to_traverse;
  std::vector<DeviceBufferRef> refs_to_traverse;
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    switch (buffer_ref.state()) {
      case DeviceBufferRefState::kZeroSize:
        continue;
      case DeviceBufferRefState::kMaterialized:
      case DeviceBufferRefState::kDeferred: {
        if (nodes_to_traverse.insert(buffer_ref.device_buffer_list()).second) {
          for (int i = 0; i < buffer_ref.device_buffer_list()->size(); ++i) {
            TT_ASSIGN_OR_RETURN(
                DeviceBufferRef ref,
                DeviceBufferRef::Create(buffer_ref.device_buffer_list(), i));
            refs_to_traverse.push_back(std::move(ref));
          }
        }
        continue;
      }
      case DeviceBufferRefState::kPlaceholder:
        return TT_ERROR(error::kInternal)
               << "[GetTraversal] was called on a placeholder "
                  "tensor. This should"
                  " never happen.";
      default:
        return TT_ERROR(error::kInternal)
               << "DeviceBufferRef has unknown state";
    }
  }

  TT_ASSIGN_OR_RETURN(Traversal traversal,
                      Traversal::Create(std::move(refs_to_traverse)));
  ABSL_VLOG(2) << "[GetTraversal] Traversal created";
  if (traversal.is_bounded_dynamic()) {
    TT_ASSIGN_OR_RETURN(auto unused_padding, traversal.ApplyDynamism());
    ABSL_VLOG(2) << "[GetTraversal] Applied dynamism";
  }
  return traversal;
}

}  // namespace

absl::StatusOr<std::string> GetComputationGraphviz(
    absl::Span<const DeviceBufferRef> buffer_refs,
    const BufferRefToVarMap& buffer_ref_to_var) {
  TT_ASSIGN_OR_RETURN(Traversal traversal, GetTraversal(buffer_refs));
  TT_ASSIGN_OR_RETURN(std::string graphviz_string,
                      GetGraphviz(traversal, buffer_ref_to_var));
  return graphviz_string;
}

absl::StatusOr<std::string> GetComputationMlir(
    absl::Span<const DeviceBufferRef> buffer_refs) {
  TT_ASSIGN_OR_RETURN(Traversal traversal, GetTraversal(buffer_refs));
  TT_ASSIGN_OR_RETURN(auto unused_padding, traversal.ApplyDynamism());
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  TT_ASSIGN_OR_RETURN(auto module, traversal.BuildMlirModule(context));
  return mlir::debugString(module.get());
}

}  // namespace torch_tpu
