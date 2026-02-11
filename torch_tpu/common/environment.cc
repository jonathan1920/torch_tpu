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
  const char* accelerator_type_env = std::getenv("ACCELERATOR_TYPE");
  if (accelerator_type_env == nullptr) {
    return TT_ERROR(error::kInvalidArgument)
           << "ACCELERATOR_TYPE environment variable not set.";
  }
  std::string accelerator_type = accelerator_type_env;

  if (absl::StartsWith(accelerator_type, "v7")) {
    return InferV7Topology(world_size);
  } else if (absl::StartsWith(accelerator_type, "v6")) {
    return InferV6Topology(world_size);
  } else if (absl::StartsWith(accelerator_type, "v5")) {
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
  if (std::getenv("TPU_PROCESS_ADDRESSES") == nullptr) {
    if (sb_addrs.empty()) {
      return absl::OkStatus();
    }
    absl::StatusOr<std::string> host_bounds = GetHostBounds(world_size);
    if (!host_bounds.ok()) {
      return TT_ERROR(error::kInternal) << absl::StrCat(
                 "Failed to get host bounds: ", host_bounds.status().message());
    }
    setenv("CLOUD_TPU_TASK_ID", absl::StrCat(local_rank).c_str(), 1);
    setenv("TPU_VISIBLE_CHIPS", absl::StrCat(local_rank).c_str(), 1);
    setenv("TPU_HOST_BOUNDS", host_bounds->c_str(), 1);
    setenv("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1", 1);

    // The free port of this process.
    setenv("TPU_PROCESS_PORT", absl::StrCat(sb_port).c_str(), 1);

    // The addresses of all other workers in the slice.
    setenv("TPU_PROCESS_ADDRESSES", sb_addrs.c_str(), 1);
  }

  const char* libtpu_init_args = std::getenv("LIBTPU_INIT_ARGS");
  std::string libtpu_init_args_str =
      libtpu_init_args == nullptr ? "" : libtpu_init_args;
  if (!absl::StrContains(libtpu_init_args_str,
                         "--xla_tpu_use_enhanced_launch_barrier")) {
    // Preventing distributed hangs, see b/477673365.
    absl::StrAppend(&libtpu_init_args_str,
                    " --xla_tpu_use_enhanced_launch_barrier=false");
    setenv("LIBTPU_INIT_ARGS", libtpu_init_args_str.c_str(), 1);
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
