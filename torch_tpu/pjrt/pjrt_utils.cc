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

#include "torch_tpu/pjrt/pjrt_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorImpl.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/pjrt/pjrt_init.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions) {
  const xla::PrimitiveType type = ConvertTo<xla::PrimitiveType>(element_type);
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL ENTRY] host_data_is_null: "
               << (host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]";

  xla::PjRtClient* const client = GetPjRtClient();
  xla::PjRtDevice* const device = GetPjRtDevice();

  TT_RET_CHECK(client != nullptr, error::kFailedPrecondition)
      << "PjRt client not initialized in TpuMallocAndMemcpyHtoD.";
  TT_RET_CHECK(device != nullptr, error::kFailedPrecondition)
      << "PjRt device not initialized in TpuMallocAndMemcpyHtoD.";

  int64_t num_elements = 1;
  if (dimensions.empty() && type != xla::TUPLE) {
    // Do nothing.
  } else {
    TT_ASSIGN_OR_RETURN(num_elements, NumElements(dimensions));
  }

  absl::Notification host_buffer_transfer_done;
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Getting default memory "
                  "space for device: "
               << device->DebugString();
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace* const memory_space,
                      device->default_memory_space());
  TT_RET_CHECK(memory_space != nullptr, error::kInternal)
      << "Default memory space is null.";
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Got memory space: "
               << memory_space->DebugString();

  xla::PjRtClient::HostBufferSemantics semantics =
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
  const void* effective_host_data = host_data;
  std::vector<char> zeroed_host_data_for_alloc;

  if (host_data == nullptr && num_elements > 0) {
    size_t buffer_size_bytes =
        xla::ShapeUtil::ByteSizeOf(xla::ShapeUtil::MakeShape(type, dimensions));
    if (buffer_size_bytes > 0) {
      ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
                      "num_elements > 0. Creating temporary zeroed host buffer "
                      "of size "
                   << buffer_size_bytes << " bytes.";
      zeroed_host_data_for_alloc.resize(buffer_size_bytes, 0);
      effective_host_data = zeroed_host_data_for_alloc.data();

    } else {
      ABSL_VLOG(1)
          << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
             "num_elements > 0, but calculated byte size is 0. Passing "
             "nullptr as effective_host_data.";
    }
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Calling "
                  "client->BufferFromHostBuffer with"
               << " effective_host_data_is_null: "
               << (effective_host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]"
               << ", semantics: " << static_cast<int>(semantics);

  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buffer,
                      client->BufferFromHostBuffer(
                          effective_host_data,

                          type, dimensions, std::nullopt, semantics,
                          [&host_buffer_transfer_done]() {
                            host_buffer_transfer_done.Notify();
                          },
                          memory_space, nullptr));

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] "
                  "client->BufferFromHostBuffer SUCCEEDED.";

  host_buffer_transfer_done.WaitForNotification();
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Host buffer transfer "
                  "notification received.";

  TT_ASSIGN_OR_RETURN(auto buffer_ref,
                      DeviceBufferList::CreateMaterialized(std::move(buffer)));
  TT_RETURN_IF_ERROR(buffer_ref.buffer().status())
      << "[TpuMallocAndMemcpyHtoD] DeviceBufferRef was copied "
         "H2D, but does not have a valid PjRtBuffer.";
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL EXIT] Created "
                  "DeviceBufferRef. Dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]";
  return buffer_ref;
}

absl::StatusOr<at::Tensor> TpuMemcpyDtoH(const DeviceBufferRef& buffer_ref) {
  ABSL_VLOG(1) << "[TpuMemcpyDtoH ENTRY] buffer_ref: "
               << buffer_ref.DebugString();

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] Extracted buffer_ref: "
               << buffer_ref.DebugString();

  size_t buffer_expected_bytes = buffer_ref.size_bytes();
  const auto buffer_expected_type =
      ConvertTo<xla::PrimitiveType>(buffer_ref.element_type());
  const auto buffer_tensor_type =
      ConvertTo<at::ScalarType>(buffer_ref.element_type());
  absl::Span<const int64_t> buffer_expected_dims = buffer_ref.dimensions();
  if (buffer_expected_bytes == 0) {
    ABSL_VLOG(1) << "[TpuMemcpyDtoH] DeviceBufferRef size_bytes is 0. "
                    "Returning empty vector.";
    return at::empty(
        buffer_expected_dims,
        at::TensorOptions().dtype(buffer_tensor_type).device(at::kCPU));
  }

  TT_ASSIGN_OR_RETURN(
      auto& buffer, buffer_ref.buffer(),
      _ << " - TpuMemcpyDtoH: DeviceBufferRef has nonzero size, "
           "but does not have a PjRtBuffer to copy from.");

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] PjRtBuffer Details - OnDeviceSizeInBytes: "
               << buffer.GetOnDeviceSizeInBytes()
               << ", IsDeleted: " << buffer.IsDeleted()
               << ", IsOnCpu: " << buffer.IsOnCpu()
               << ", Shape: " << buffer.on_device_shape().ToString(true);

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] Calling PjRtBuffer::ToLiteralSync()...";
  std::vector<char> host_data_vec(buffer_expected_bytes);
  at::Tensor cpu_tensor_receiver =
      at::empty(buffer_expected_dims,
                at::TensorOptions().dtype(buffer_tensor_type).device(at::kCPU));
  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      buffer_expected_type, buffer_expected_dims);
  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(cpu_tensor_receiver.data_ptr()), xla_shape);
  auto future = buffer.ToLiteral(literal.get());
  TT_RETURN_IF_ERROR(future.Await());
  return cpu_tensor_receiver;
}

absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutable& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers) {
  xla::ExecuteOptions execute_options{.strict_shape_checking = true};

  std::vector<std::vector<xla::PjRtBuffer*>> execution_arguments;
  execution_arguments.reserve(1);
  execution_arguments.push_back(std::move(argument_buffers));

  TT_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
          results_per_device,
      executable->Execute(execution_arguments, execute_options));

  TT_RET_CHECK(!results_per_device.empty(), error::kInternal)
      << "XLA execution did not return any results.";

  PjRtBufferPointers result_pointers;
  result_pointers.reserve(results_per_device[0].size());
  for (auto&& [i, result] : llvm::enumerate(results_per_device[0])) {
    result_pointers.push_back(std::move(result));
  }

  return result_pointers;
}

}  // namespace torch_tpu
