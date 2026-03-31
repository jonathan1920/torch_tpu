// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/pjrt/pjrt_shutdown.h"

#include "absl/log/absl_log.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {

void ShutdownPjRt() {
  ABSL_VLOG(1) << "ShutdownPjRt";
  CompilationCache::ShutDown();
  auto* client = GetPjRtClient();
  if (!client) {
    ABSL_VLOG(1) << "PjRt not initialized.";
    return;
  }
  // TODO(mvoz): Teardown order issue, will be resolved w/ RAII
  // client->reset();
  SetPjRtDevice(nullptr);
  ABSL_VLOG(1) << "PjRt shut down.";
}
}  // namespace torch_tpu
