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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_INPUT_PREPROCESSING_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_INPUT_PREPROCESSING_ATEN_KERNELS_H_

#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

struct SparseCoreInputTorch {
  at::Tensor row_pointers;   // UNINITIALIZED_TENSOR_OK
  at::Tensor embedding_ids;  // UNINITIALIZED_TENSOR_OK
  at::Tensor sample_ids;     // UNINITIALIZED_TENSOR_OK
  at::Tensor gains;          // UNINITIALIZED_TENSOR_OK
};

// Note:
//   Currently scoped to a single-host, single-device execution baseline.
//   Will be expanded to multi-device distributed topologies in future PRs.
//   `allow_id_dropping`: Controls whether tail embedding IDs are dropped when
//   `coo_buffer_size_per_device` limit is exceeded. If false, overflows error.
SparseCoreInputTorch PreprocessToSparseCore(
    const at::Tensor& values, const at::Tensor& offsets,
    int64_t global_device_count = 1, int64_t coo_buffer_size_per_device = -1,
    int64_t num_sc_per_device = 2, bool allow_id_dropping = true);

std::tuple<at::Tensor, at::Tensor> RestoreSparseCore(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& offsets,
    int64_t global_device_count = 1, int64_t num_sc_per_device = 2);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_INPUT_PREPROCESSING_ATEN_KERNELS_H_
