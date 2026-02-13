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

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {

namespace {

absl::StatusOr<std::string> InferV7Topology(int world_size) {
  if (world_size == 8) {
    return "2,2,1,2";
  }
  return TT_ERROR(error::kInvalidArgument)
         << absl::StrCat("Unsupported TPU V7 world size: ", world_size);
}

absl::StatusOr<std::string> InferV6Topology(int world_size) {
  if (world_size == 8) {
    return "4,2,1";
  }
  if (world_size == 4) {
    return "2,2,1";
  }
  return TT_ERROR(error::kInvalidArgument)
         << absl::StrCat("Unsupported TPU V6 world size: ", world_size);
}

absl::StatusOr<std::string> InferV5Topology(int world_size) {
  if (world_size == 8) {
    return "2,4,1";
  }
  if (world_size == 4) {
    return "2,2,1";
  }
  return TT_ERROR(error::kInvalidArgument)
         << absl::StrCat("Unsupported TPU V5 world size: ", world_size);
}

absl::StatusOr<std::string> GetHostBounds(int world_size) {
  const auto& accelerator_type_env = GetEnvOnce<kAcceleratorTypeEnvVar>();
  TT_RET_CHECK(accelerator_type_env.has_value(), error::kInvalidArgument)
      << "ACCELERATOR_TYPE environment variable not set.";
  const std::string& accelerator_type = *accelerator_type_env;

  if (absl::StartsWith(accelerator_type, "v7")) {
    return InferV7Topology(world_size);
  }
  if (absl::StartsWith(accelerator_type, "v6")) {
    return InferV6Topology(world_size);
  }
  if (absl::StartsWith(accelerator_type, "v5")) {
    return InferV5Topology(world_size);
  }
  return TT_ERROR(error::kInvalidArgument)
         << absl::StrCat("Unsupported ACCELERATOR_TYPE: ", accelerator_type,
                         " for world size: ", world_size);
}

}  // namespace

absl::Status InitializeDistributedEnvironment(int rank, int world_size,
                                              int local_rank,
                                              std::string sb_addrs,
                                              int sb_port) {
  if (!GetEnvOnce<kTpuProcessAddressesEnvVar>().has_value()) {
    if (sb_addrs.empty()) {
      return absl::OkStatus();
    }
    absl::StatusOr<std::string> host_bounds = GetHostBounds(world_size);
    if (!host_bounds.ok()) {
      return TT_ERROR(error::kInternal) << absl::StrCat(
                 "Failed to get host bounds: ", host_bounds.status().message());
    }
    SetEnv(kCloudTpuTaskIdEnvVar, absl::StrCat(local_rank));
    SetEnv(kTpuVisibleChipsEnvVar, absl::StrCat(local_rank));
    SetEnv(kTpuHostBoundsEnvVar, *host_bounds);
    SetEnv(kTpuChipsPerHostBoundsEnvVar, "1,1,1");

    // The free port of this process.
    SetEnv(kTpuProcessPortEnvVar, absl::StrCat(sb_port));

    // The addresses of all other workers in the slice.
    SetEnv(kTpuProcessAddressesEnvVar, sb_addrs);
  }

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
