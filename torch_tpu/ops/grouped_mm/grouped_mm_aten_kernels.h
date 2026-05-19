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

#ifndef TORCH_TPU_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor AtenGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                         const std::optional<at::Tensor>& offs,
                         const std::optional<at::Tensor>& bias,
                         std::optional<at::ScalarType> out_dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_
