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

#ifndef TORCH_TPU_COMMON_ENV_VARS_H_
#define TORCH_TPU_COMMON_ENV_VARS_H_

#include <cstdlib>
#include <string>

namespace torch_tpu {

// All environment variables read or written by torch_tpu.
//
// If an env var is not introduced by TorchTPU, please document the source.
//
// Since std::getenv() takes a const char*, and conversion from std::string_view
// to const char* is expensive, we define these as C-string literals.

// go/keep-sorted start
inline constexpr char kAcceleratorTypeEnvVar[] =
    "ACCELERATOR_TYPE";  // Set by Google Cloud.
inline constexpr char kCloudTpuTaskIdEnvVar[] =
    "CLOUD_TPU_TASK_ID";  // Read by Google Cloud.
inline constexpr char kLibtpuInitArgsEnvVar[] =
    "LIBTPU_INIT_ARGS";  // Set by Google Cloud.
inline constexpr char kNprocEnvVar[] =
    "NPROC";  // Set by the build system (e.g. bazel).
inline constexpr char kTorchShowCppStacktracesEnvVar[] =
    "TORCH_SHOW_CPP_STACKTRACES";  // Shared with PyTorch.
inline constexpr char kTorchTpuEnableDebugChecksEnvVar[] =
    "TORCH_TPU_ENABLE_DEBUG_CHECKS";
inline constexpr char kTorchTpuInternalEagerCompilationModeEnvVar[] =
    "TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE";
inline constexpr char kTorchTpuInternalXlaOptionsEnvVar[] =
    "TORCH_TPU_INTERNAL_XLA_OPTIONS";
inline constexpr char kTpuChipsPerHostBoundsEnvVar[] =
    "TPU_CHIPS_PER_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuHostBoundsEnvVar[] =
    "TPU_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuProcessAddressesEnvVar[] =
    "TPU_PROCESS_ADDRESSES";  // Set by Google Cloud.
inline constexpr char kTpuProcessPortEnvVar[] =
    "TPU_PROCESS_PORT";  // Read by Google Cloud.
inline constexpr char kTpuVisibleChipsEnvVar[] =
    "TPU_VISIBLE_CHIPS";  // Read by Google Cloud.
inline constexpr char kWorldSizeEnvVar[] =
    "WORLD_SIZE";  // Set by launchers like torchrun.
// go/keep-sorted end

inline void SetEnv(const char* name, const std::string& value) {
  setenv(name, value.c_str(), /*overwrite=*/1);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_ENV_VARS_H_
