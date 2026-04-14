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

#include "torch_tpu/eager/eager_mode.h"

#include <atomic>

#include "absl/log/absl_log.h"
#include "torch_tpu/common/env_vars.h"

namespace torch_tpu {

namespace {

EagerMode GetDefaultEagerMode() {
  if (GetEnvOnce<kTpuLaunchBlocking>() == "1") {
    return EagerMode::kDeferNeverAndLaunchBlocking;
  } else if (GetEnvOnce<kTpuDeferAndFuse>() == "1") {
    return EagerMode::kDeferAndFuse;
  } else {
    return EagerMode::kDeferNever;
  }
}

// Returns the defer mode for the current thread.
std::atomic<EagerMode>& GetMutableEagerMode() {
  static std::atomic<EagerMode> eager_mode = GetDefaultEagerMode();
  return eager_mode;
}

}  // namespace

EagerMode GetEagerMode() { return GetMutableEagerMode(); }

void SetEagerMode(const EagerMode mode) {
  ABSL_VLOG(1) << "SetEagerMode " << static_cast<int>(mode);
  GetMutableEagerMode() = mode;
}

}  // namespace torch_tpu
