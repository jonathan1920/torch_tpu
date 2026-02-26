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

#include "torch_tpu/common/environment.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"

namespace torch_tpu {

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config) {
  SetEnv(kCloudTpuTaskIdEnvVar, absl::StrCat(config.rank));
  SetEnv(kTpuVisibleChipsEnvVar, absl::StrCat(config.local_rank));
  SetEnv(kTpuHostBoundsEnvVar, config.topology);
  SetEnv(kTpuChipsPerHostBoundsEnvVar, "1,1,1,1");

  // The free slicebuilder port of this process.
  SetEnv(kTpuProcessPortEnvVar, absl::StrCat(config.sb_port));

  // The addresses of all other workers in the slice.
  SetEnv(kTpuProcessAddressesEnvVar, config.sb_addrs);

  // Avoid multi-process libtpu lock in GCP init, see b/487769788.
  SetEnv(kAllowMultipleLibtpuLoadEnvVar, "1");

  std::string libtpu_init_args_str =
      GetEnvOnce<kLibtpuInitArgsEnvVar>().value_or("");
  if (!absl::StrContains(libtpu_init_args_str,
                         "--xla_tpu_use_enhanced_launch_barrier")) {
    // Preventing distributed hangs, see b/477673365.
    absl::StrAppend(&libtpu_init_args_str,
                    " --xla_tpu_use_enhanced_launch_barrier=false");
    SetEnv(kLibtpuInitArgsEnvVar, libtpu_init_args_str);
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
