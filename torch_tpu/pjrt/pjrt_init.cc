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

#include "torch_tpu/pjrt/pjrt_init.h"

#include <algorithm>
#include <memory>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/common/environment.h"
#include "torch_tpu/pjrt/pjrt_client.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/plugin_names.h"

namespace torch_tpu {

absl::StatusOr<PjRtInitializationResult> InitializePjRt(
    const PjRtInitializationOptions& options) {
  ABSL_LOG(INFO) << "InitializePjRt: device_type=" << options.device_type
                 << ", world_size=" << options.world_size;

  if (IsPjRtInitialized()) {
    xla::PjRtClient* client = GetPjRtClient();
    xla::PjRtDevice* device = GetPjRtDevice();
    return PjRtInitializationResult{
        .device_count = client->device_count(),
        .global_device_id = device->global_device_id().value()};
  }

  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
  if (options.device_type == "tpu") {
    // If WORLD_SIZE is set, we are in a distributed environment. The mandatory
    // environment variables must be set by the caller.
    DistributedWorkerConfiguration config;
    if (GetEnvOnce<kWorldSizeEnvVar>()) {
      TT_ASSIGN_OR_RETURN(config, GetDistributedWorkerConfiguration());
      TT_RETURN_IF_ERROR(InitializeDistributedEnvironment(config));
    }
    TT_RETURN_IF_ERROR(::pjrt::InitializePjrtPlugin(kTpuPjrtName));
    device_type = PjRtDeviceType::kTpu;
  } else if (options.device_type == "xla_cuda") {
    TT_RETURN_IF_ERROR(::pjrt::InitializePjrtPlugin(kGpuPjrtName));
    device_type = PjRtDeviceType::kCuda;
  } else if (options.device_type == "xla_cpu") {
    TT_RETURN_IF_ERROR(::pjrt::InitializePjrtPlugin(kCpuPjrtName));
    device_type = PjRtDeviceType::kCpu;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "Unsupported device type: " << options.device_type;
  }

  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      torch_tpu::pjrt::GetPjRtClient(options.device_type));

  TT_RET_CHECK(client != nullptr, error::kInternal)
      << "PjRtClient is null after initialization";

  TT_RET_CHECK(!client->devices().empty(), error::kNotFound)
      << "no PjRt devices found after client initialization";

  const auto& addressable_devices = client->addressable_devices();
  const int device_count = client->device_count();
  const int world_size = std::max(1, options.world_size);

  // Initialize the global device count. This is used by the CompilationCache
  // to determine the number of replicas of the XLA computation.
  TT_RET_CHECK(world_size == 1 || world_size == device_count,
               error::kInvalidArgument)
      << "world size must be 1 or equal to the number of global device count, "
      << "got world size: " << world_size
      << " and device count: " << device_count;

  if (addressable_devices.size() > 1 && world_size == 1) {
    ABSL_LOG(WARNING) << "Only using 1 device out of all the "
                      << addressable_devices.size() << " addressable devices.";
  }

  xla::PjRtDevice* device = addressable_devices[0];
  ABSL_CHECK(device != nullptr)  // CRASH_OK
      << "PjRtDevice is null.";

  int global_device_count = device_count;

  SetPjRtState(client.release(), device, device_type, global_device_count);

  ABSL_VLOG(1) << "PjRt initialized. Using device: " << device->DebugString();
  return PjRtInitializationResult{
      .device_count = global_device_count,
      .global_device_id = device->global_device_id().value()};
}

}  // namespace torch_tpu
