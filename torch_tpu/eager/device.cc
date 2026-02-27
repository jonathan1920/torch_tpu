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

#include "torch_tpu/eager/device.h"

#include "absl/base/nullability.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

struct PjRtDeviceData {
  xla::PjRtDevice* absl_nullable device = nullptr;
  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
};

static PjRtDeviceData g_pjrt_device_data = PjRtDeviceData();

void SetPjRtDevice(xla::PjRtDevice* absl_nullable device,
                   const PjRtDeviceType device_type) {
  g_pjrt_device_data = PjRtDeviceData{
      .device = device,
      .device_type = device_type,
  };
}

xla::PjRtDevice* absl_nullable GetPjRtDevice() {
  return g_pjrt_device_data.device;
}

[[nodiscard]] PjRtDeviceType GetPjRtDeviceType() {
  return g_pjrt_device_data.device_type;
}
}  // namespace torch_tpu
