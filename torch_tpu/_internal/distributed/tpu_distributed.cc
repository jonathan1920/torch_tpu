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

#include <chrono>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep or multi_tpu_test fails
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/process_group_tpu.h"
#include "torch_tpu/eager/device.h"
#include "torch_tpu/pjrt/pjrt_init.h"
#include "pybind11/chrono.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torch_tpu {

//
// These are only intended to be called from Python.
//
namespace {

// This function will be registered as the TPU distributed backend init
// function (see `torch.distributed.Backend.register_backend`) and called by
// pytorch when the user calls `torch.distributed.init_process_group()`.
c10::intrusive_ptr<c10d::Backend> MakeProcessGroup(
    c10::intrusive_ptr<c10d::Store> store, int rank, int size,
    std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupTpu>(store, rank, size);
}

// Global device id of the (single) addressible device this process is using.
int GetGlobalDeviceId() {
  const auto* pjrtdev = GetPjRtDevice();
  TT_CHECK_THROW(pjrtdev, error::kInternal) << "PjRtDevice is not initialized.";
  return pjrtdev->global_device_id().value();
}

std::vector<int> GetAllGlobalDeviceIds() {
  const auto* pjrt_client = GetPjRtClient();
  TT_CHECK_THROW(pjrt_client, error::kInternal)
      << "PjRtClient is not initialized.";

  const auto& devices = pjrt_client->devices();
  std::vector<int> global_device_ids;

  for (const auto& device : devices) {
    ABSL_CHECK(device != nullptr) << "Got a nullptr PjRtDevice.";  // CRASH_OK
    global_device_ids.push_back(device->global_device_id().value());
  }
  return global_device_ids;
}

std::string DeviceDebugString() {
  const auto* pjrtdev = GetPjRtDevice();
  TT_CHECK_THROW(pjrtdev, error::kInternal) << "PjRtDevice is not initialized.";
  return std::string(pjrtdev->DebugString());
}

}  // namespace

PYBIND11_MODULE(tpu_distributed, m) {
  // Custom distributed backend initialization happens from the Python side,
  // so we need to expose this init function.
  m.def("create_process_group", &MakeProcessGroup);

  // We expose these as helper functions on the Python side
  m.def("global_device_id", &GetGlobalDeviceId);
  m.def("all_global_device_ids", &GetAllGlobalDeviceIds);
  m.def("device_debug_string", &DeviceDebugString);
}

}  // namespace torch_tpu
