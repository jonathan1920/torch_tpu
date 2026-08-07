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
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "torch_tpu/common/device_utils.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"

namespace torch_tpu {

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config) {
  if (config.sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  if (config.rank < 0) {
    return TT_ERROR(error::kFailedPrecondition)
           << "RANK " << config.rank << " is out of bounds (negative)";
  }
  std::vector<std::string> addresses = absl::StrSplit(config.sb_addrs, ',');
  int slice_rank = config.rank % static_cast<int>(addresses.size());
  SetEnv(kCloudTpuTaskIdEnvVar, absl::StrCat(slice_rank));

  const auto& visible_chips_env = GetEnvOnce<kTpuVisibleChipsEnvVar>();
  const auto& visible_devices_env = GetEnvOnce<kTpuVisibleDevicesEnvVar>();
  if (visible_chips_env.has_value() &&
      IsSingleDeviceSpecified(*visible_chips_env)) {
    // Preserve existing TPU_VISIBLE_CHIPS setting.
  } else if (visible_devices_env.has_value() &&
             IsSingleDeviceSpecified(*visible_devices_env)) {
    SetEnv(kTpuVisibleChipsEnvVar, *visible_devices_env);
  } else {
    SetEnv(kTpuVisibleChipsEnvVar, absl::StrCat(config.local_rank));
  }

  std::vector<std::string> topology_dims = absl::StrSplit(config.topology, ',');
  std::string chips_bounds = (topology_dims.size() == 4) ? "1,1,1,1" : "1,1,1";

  SetEnv(kTpuHostBoundsEnvVar, config.topology);
  SetEnv(kTpuChipsPerHostBoundsEnvVar, chips_bounds);
  SetEnv(kTpuProcessBoundsEnvVar, config.topology);
  SetEnv(kTpuChipsPerProcessBoundsEnvVar, chips_bounds);

  // The free slicebuilder port of this process.
  SetEnv(kTpuProcessPortEnvVar, config.sb_port);

  // The addresses of all other workers in the slice.
  SetEnv(kTpuProcessAddressesEnvVar, config.sb_addrs);

  // Avoid multi-process libtpu lock in GCP init, see b/487769788.
  SetEnv(kAllowMultipleLibtpuLoadEnvVar, "1");

  std::string libtpu_init_args_str =
      GetEnvOnce<kLibtpuInitArgsEnvVar>().value_or("");
  if (absl::StrContains(libtpu_init_args_str,
                        "--xla_tpu_use_enhanced_launch_barrier=true")) {
    ABSL_LOG(WARNING)
        << "libtpu_init_args contains "
           "--xla_tpu_use_enhanced_launch_barrier=true, "
           "this can cause distributed hangs. Rewriting environment "
           "variable to false.";
    libtpu_init_args_str =
        absl::StrReplaceAll(libtpu_init_args_str,
                            {{"--xla_tpu_use_enhanced_launch_barrier=true",
                              "--xla_tpu_use_enhanced_launch_barrier=false"}});
  } else if (!absl::StrContains(libtpu_init_args_str,
                                "--xla_tpu_use_enhanced_launch_barrier")) {
    // Preventing distributed hangs, see b/477673365.
    absl::StrAppend(&libtpu_init_args_str,
                    " --xla_tpu_use_enhanced_launch_barrier=false");
  }
  SetEnv(kLibtpuInitArgsEnvVar, libtpu_init_args_str);
  return absl::OkStatus();
}

}  // namespace torch_tpu
