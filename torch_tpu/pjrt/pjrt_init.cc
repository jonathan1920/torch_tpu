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
#include <functional>
#include <memory>
#include <utility>

#include "absl/base/call_once.h"
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
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"

namespace torch_tpu {
namespace {

bool IsRunningInTest() {
  return GetEnvOnce<kTestWorkspaceEnvVar>().has_value() ||
         GetEnvOnce<kTestTargetEnvVar>().has_value();
}

}  // namespace

static absl::once_flag profiler_factory_once;

absl::StatusOr<PjRtInitializationResult> InitializePjRt(
    const PjRtInitializationOptions& options) {
  ABSL_LOG(INFO) << "InitializePjRt: device_type=" << options.device_type
                 << ", world_size=" << options.world_size;

  if (IsPjRtInitialized()) {
    xla::PjRtDevice* device = GetPjRtDevice();
    return PjRtInitializationResult{
        .device_count = GetGlobalDeviceCount().value(),
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
    TT_RETURN_IF_ERROR(
        ::pjrt::InitializePjrtPlugin(kTpuPjrtName));  // LEADING_COLONS_OK

    // Initialize profiler if PJRT_Profiler_Extension is available.
    absl::StatusOr<const PJRT_Api*> api =
        ::pjrt::PjrtApi(kTpuPjrtName);  // LEADING_COLONS_OK
    if (api.ok() && *api != nullptr) {
      PJRT_Profiler_Extension* profiler_ext =
          ::pjrt::FindExtension<PJRT_Profiler_Extension>(  // LEADING_COLONS_OK
              *api, PJRT_Extension_Type::PJRT_Extension_Type_Profiler);
      if (profiler_ext != nullptr && profiler_ext->profiler_api != nullptr) {
        absl::call_once(profiler_factory_once, [&]() {
          const PLUGIN_Profiler_Api* profiler_api = profiler_ext->profiler_api;
          std::function<std::unique_ptr  // STD_FUNCTION_OK
                        <tsl::profiler::ProfilerInterface>(
                            const tensorflow::ProfileOptions&)>
              create_func =
                  [profiler_api](const tensorflow::ProfileOptions& profile_opts)
              -> std::unique_ptr<tsl::profiler::ProfilerInterface> {
            // In test environments (e.g. running on simulators with shared
            // symbols), the mock libtpu plugin is often linked statically
            // into the framework binary.
            //
            // When the framework starts tracing, it calls
            // `tsl::profiler::CreateProfilers`, which acquires a global
            // non-recursive mutex in TSL.
            //
            // 1. Framework acquires Global Mutex A.
            // 2. Calls this factory lambda to create `PluginTracer`.
            // 3. `PluginTracer` constructor calls the plugin (mock libtpu).
            // 4. The mock plugin attempts to initialize its own profilers by
            // calling
            //    `tsl::profiler::CreateProfilers` again (nested call).
            // 5. The nested call tries to acquire Global Mutex A and deadlocks!
            //
            // Since we cannot modify TSL (to make it re-entrant safe or using
            // recursive mutexes) and we cannot modify the mock plugin (to stop
            // it from calling out), the only safe solution in this
            // statically-linked environment is to SKIP creating the
            // `PluginTracer` in tests.
            //
            // We use standard Bazel environment variables (`TEST_WORKSPACE` or
            // `TEST_TARGET`) to detect if we are running in a test context.
            if (IsRunningInTest()) {
              ABSL_VLOG(1) << "Skipping PluginTracer in test environment to "
                              "avoid shared "
                              "TSL mutex deadlock.";
              return nullptr;
            }

            return std::make_unique<xla::profiler::PluginTracer>(profiler_api,
                                                                 profile_opts);
          };
          tsl::profiler::RegisterProfilerFactory(std::move(create_func));
        });
      } else {
        ABSL_LOG(WARNING) << "PJRT_Profiler_Extension not found, profiler will "
                             "not be initialized.";
      }
    } else {
      ABSL_LOG(WARNING)
          << "Failed to get PjRtApi, profiler will not be initialized.";
    }
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
                      torch_tpu::pjrt::GetPjRtClient(
                          options.device_type, options.premapped_buffer_size));

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

  int global_device_count = world_size;

  SetPjRtState(client.release(), device, device_type, global_device_count);

  ABSL_VLOG(1) << "PjRt initialized. Using device: " << device->DebugString();
  return PjRtInitializationResult{
      .device_count = global_device_count,
      .global_device_id = device->global_device_id().value()};
}

}  // namespace torch_tpu
