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

#include "torch_tpu/pjrt/pjrt_state.h"

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_types.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {
namespace {

struct GlobalPjRtState {
  absl::Mutex mutex;
  xla::PjRtClient* client = nullptr;
  xla::PjRtDevice* device = nullptr;
  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
  int global_device_count = 0;
  std::string backend_name = "tpu";
};

GlobalPjRtState& GetGlobalState() {
  static absl::NoDestructor<GlobalPjRtState> state;
  return *state;
}

}  // namespace

xla::PjRtClient* GetPjRtClient() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.client;
}

xla::PjRtDevice* GetPjRtDevice() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.device;
}

void SetPjRtDevice(xla::PjRtDevice* device) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.device = device;
}

bool IsPjRtInitialized() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.client != nullptr;
}

absl::StatusOr<int> GetGlobalDeviceCount() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  if (state.client == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return state.global_device_count;
}

PjRtDeviceType GetPjRtDeviceType() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.device_type;
}

void SetPjRtBackendName(const std::string& device_type) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.backend_name = device_type;
}

std::string GetPjRtBackendName() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.backend_name;
}

void SetPjRtState(xla::PjRtClient* client, xla::PjRtDevice* device,
                  PjRtDeviceType device_type, int global_device_count) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.client = client;
  state.device = device;
  state.device_type = device_type;
  state.global_device_count = global_device_count;
}

absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() {
  xla::PjRtDevice* device = GetPjRtDevice();
  if (device == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt device is not initialized";
  }
  return device->GetAllocatorStats();
}

}  // namespace torch_tpu
