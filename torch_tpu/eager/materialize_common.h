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

#ifndef TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_
#define TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_

#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

absl::Status ExecuteMaterializationJob(
    absl::Span<const DeviceBufferRef> arguments,
    absl::Span<const DeviceBufferRef> output_buffer_refs,
    std::vector<SharedLoadedExecutableWithMetadata> executables,
    std::string_view task_name = "anonymous");

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_
