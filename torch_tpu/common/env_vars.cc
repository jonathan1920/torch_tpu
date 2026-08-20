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

#include "torch_tpu/common/env_vars.h"

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"

namespace torch_tpu {

const absl::flat_hash_set<const char*>& GetStableEnvVars() {
  static const absl::NoDestructor<absl::flat_hash_set<const char*>>
      stable_env_vars({kWorldSizeEnvVar});
  return *stable_env_vars;
}

const absl::flat_hash_set<const char*>& GetExperimentalEnvVars() {
  static const absl::NoDestructor<absl::flat_hash_set<const char*>>
      experimental_env_vars({kTorchTpuTier2CompilationCacheEnvVar,      //
                             kTorchTpuTier3CompilationCacheRootEnvVar,  //
                             kXlaFlagsEnvVar});
  return *experimental_env_vars;
}

const absl::flat_hash_map<const char*, std::string>& GetDeprecatedEnvVars() {
  static const absl::NoDestructor<absl::flat_hash_map<const char*, std::string>>
      deprecated_env_vars({});
  return *deprecated_env_vars;
}

bool GetMaterializeCollectiveTensorsEnvValue() {
  static const bool env_value = []() {
    const auto& raw_env_value =
        GetEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>();
    return !raw_env_value.has_value() ||
           (*raw_env_value != "0" &&
            !absl::EqualsIgnoreCase(*raw_env_value, "false"));
  }();
  return env_value;
}

}  // namespace torch_tpu
