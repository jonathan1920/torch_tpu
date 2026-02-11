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

#ifndef TORCH_TPU_DISTRIBUTED_SLICEBUILDER_DISCOVERY_H_
#define TORCH_TPU_DISTRIBUTED_SLICEBUILDER_DISCOVERY_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace torch_tpu {

struct DistributedWorkerConfiguration {
  int rank;
  int local_rank;
  int world_size;
  std::string master_addr;
  int master_port;
};

absl::StatusOr<std::pair<std::string, int64_t>> GetSlicebuilderConfig(
    int rank, int world_size, std::string master_addr, int master_port);

absl::Status InitializeAsDistributedWorker(
    const DistributedWorkerConfiguration& config);

absl::StatusOr<DistributedWorkerConfiguration>
GetDistributedWorkerConfiguration();

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_SLICEBUILDER_DISCOVERY_H_
