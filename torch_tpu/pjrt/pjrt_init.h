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

#ifndef TORCH_TPU_PJRT_INIT_H_
#define TORCH_TPU_PJRT_INIT_H_

#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

struct PjRtInitializationResult {
  // The number of devices visible to the PjRtClient. In a multi-client setting
  // they may not all be addressable.
  int device_count = -1;
  // The global ID of the current device. Globally unique across hosts.
  int global_device_id = -1;
};

struct PjRtInitializationOptions {
  // A const & field simulates a required parameter: it forces the caller to
  // provide a value for this field instead of using a default value.
  const std::string& device_type;
  const int& world_size;
};

// Initializes PjRt if it hasn't been initialized. Otherwise, returns OK without
// doing anything. To succeed, the PjRt plugin corresponding to
// `options.device_type` must have been linked in:
//   - "xla/pjrt/xla_tpu:xla_tpu_pjrt_client" for "tpu"
//   - "xla/pjrt/xla_gpu:xla_gpu_pjrt_client" for "xla_cuda"
//   - "xla/pjrt/xla_cpu:xla_cpu_pjrt_client" for "xla_cpu"
// TODO(mvoz): make RAII.
absl::StatusOr<PjRtInitializationResult> InitializePjRt(
    const PjRtInitializationOptions& options);

// Returns the PjRtClient singleton if it has been initialized. Otherwise,
// returns nullptr.
[[nodiscard]] xla::PjRtClient* absl_nullable GetPjRtClient();

// Returns the global device count. Returns an error if the global device count
// is not set. Thread-safe.
absl::StatusOr<int> GetGlobalDeviceCount();

}  // namespace torch_tpu

#endif  // TORCH_TPU_PJRT_INIT_H_
