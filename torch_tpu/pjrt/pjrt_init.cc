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

#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"
#include "torch_tpu/eager/device.h"
#include "torch_tpu/common/environment.h"
#include "torch_tpu/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {
namespace {

// g_prjt_client cannot be null, but its pointee (a unique_ptr) can. If the
// unique_ptr is null, PjRt is not initialized.
absl_nullable auto* const absl_nonnull g_pjrt_client =
    new std::unique_ptr<xla::PjRtClient>();

ABSL_CONST_INIT static absl::Mutex device_count_mutex(absl::kConstInit);

static std::optional<int>& get_global_device_count()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(device_count_mutex) {
  static absl::NoDestructor<std::optional<int>> g_global_device_count
      ABSL_GUARDED_BY(device_count_mutex);
  return *g_global_device_count;
}

// Sets the global device count. Returns an error if count < 1 or if the
// global device count has already been set. Thread-safe.
absl::Status SetGlobalDeviceCount(int count)
    ABSL_LOCKS_EXCLUDED(device_count_mutex) {
  TT_RET_CHECK(count >= 1, error::kInvalidArgument)
      << "global device count must be positive, got: " << count;
  TT_MUTEX_LOCK(lock, device_count_mutex);
  std::optional<int>& global_device_count = get_global_device_count();
  TT_RET_CHECK(!global_device_count.has_value(), error::kFailedPrecondition)
      << "global device count is already set.";
  global_device_count = count;
  return absl::OkStatus();
}

}  // namespace

xla::PjRtClient* GetPjRtClient() { return g_pjrt_client->get(); }

absl::StatusOr<int> GetGlobalDeviceCount()
    ABSL_LOCKS_EXCLUDED(device_count_mutex) {
  TT_MUTEX_LOCK(lock, device_count_mutex);
  std::optional<int>& global_device_count = get_global_device_count();
  TT_RET_CHECK(global_device_count.has_value(), error::kFailedPrecondition)
      << "global device count is not set.";
  return *global_device_count;
}

absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() {
  xla::PjRtDevice* device = GetPjRtDevice();
  TT_RET_CHECK(device != nullptr, error::kFailedPrecondition)
      << "PjRtDevice is not initialized.";
  return device->GetAllocatorStats();
}

absl::StatusOr<PjRtInitializationResult> InitializePjRt(
    const PjRtInitializationOptions& options) {
  ABSL_LOG(INFO) << "InitializePjRt: device_type=" << options.device_type
                 << ", world_size=" << options.world_size;
  TT_RET_CHECK(*g_pjrt_client == nullptr, error::kInternal)
      << "PjRtClient is already initialized";

  ABSL_VLOG(1) << "Initializing PjRT";

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

  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_status =
      torch_tpu::pjrt::GetPjRtClient(options.device_type);

  TT_RETURN_IF_ERROR(client_status.status());
  *g_pjrt_client = std::move(client_status.value());

  TT_RET_CHECK(*g_pjrt_client != nullptr, error::kInternal)
      << "PjRtClient is null after initialization";

  TT_RET_CHECK(!(*g_pjrt_client)->devices().empty(), error::kNotFound)
      << "no PjRt devices found after client initialization";

  const auto& addressable_devices = (*g_pjrt_client)->addressable_devices();
  const int device_count = (*g_pjrt_client)->device_count();
  const int world_size = options.world_size;

  // Initialize the global device count. This is used by the CompilationCache
  // to determine the number of replicas of the XLA computation.
  TT_RET_CHECK(world_size == 1 || world_size == device_count,
               error::kInvalidArgument)
      << "world size must be 1 or equal to the number of global device count, "
      << "got world size: " << world_size
      << " and device count: " << device_count;

  if (addressable_devices.size() > world_size) {
    ABSL_LOG(WARNING) << "Only using " << world_size
                      << "device(s) out of all the "
                      << addressable_devices.size() << " addressable devices.";
  }
  TT_RETURN_IF_ERROR(SetGlobalDeviceCount(world_size));

  xla::PjRtDevice* device = addressable_devices[0];
  ABSL_CHECK(device != nullptr)  // CRASH_OK
      << "PjRtDevice is null.";

  SetPjRtDevice(device, device_type);
  ABSL_VLOG(1) << "PjRt initialized. Using device: " << device->DebugString();
  return PjRtInitializationResult{
      .device_count = (*g_pjrt_client)->device_count(),
      .global_device_id = device->global_device_id().value()};
}

}  // namespace torch_tpu
