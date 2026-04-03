/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_COMMON_DEVICE_TYPE_H_
#define TORCH_TPU_COMMON_DEVICE_TYPE_H_

#include <string>

#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

// The actual physical device type used by PjRt.
enum class PjRtDeviceType {
  kUnknown,
  kTpu,
  kCpu,
  kCuda,
};

// Returns the device type for TPU.
[[nodiscard]] inline constexpr c10::DeviceType GetPrivateUse1DeviceType() {
  return c10::DeviceType::PrivateUse1;
}

// Returns a user-friendly name for the PrivateUse1 device.
[[nodiscard]] inline std::string GetPrivateUse1DeviceDebugName() {
  return "tpu";
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DEVICE_TYPE_H_
