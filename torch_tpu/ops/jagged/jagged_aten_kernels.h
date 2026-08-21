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

#ifndef TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"

namespace torch_tpu {

// Converts a jagged nested tensor's flat values buffer into a padded dense
// tensor on TPU.
//
// Arguments:
//   - values: 1D or ND tensor containing concatenated jagged sequence tokens
//     with shape (total_L, *trailing_feature_dims).
//   - offsets: 1D int64 tensor list (single jagged dimension) with shape
//     (batch_size + 1,) indicating sequence boundary offsets.
//   - max_lengths: SymIntArrayRef containing the target sequence length
//     [max_length] for the padded dense dimension. Sequences longer than
//     max_length are truncated, while shorter sequences are padded with
//     padding_value.
//   - padding_value: Constant scalar value used to fill unoccupied positions
//     in the padded dense output.
//
// Returns:
//   A dense tensor of shape (batch_size, max_length, *trailing_feature_dims).
at::Tensor AtenJaggedToPaddedDenseForward(const at::Tensor& values,
                                          at::TensorList offsets,
                                          c10::SymIntArrayRef max_lengths,
                                          double padding_value);

// Converts a padded dense tensor back into a flat jagged values buffer on TPU.
//
// Arguments:
//   - dense: Tensor of shape (batch_size, max_length, *trailing_feature_dims).
//   - offsets: 1D int64 tensor list with shape (batch_size + 1,) indicating
//     the destination sequence lengths and boundaries in the jagged output.
//   - total_L: Optional total number of valid elements expected in the
//     output values buffer. Defaults to offsets[-1] if not specified.
//
// Returns:
//   A flat jagged values tensor of shape (total_L, *trailing_feature_dims).
at::Tensor AtenPaddedDenseToJaggedForward(const at::Tensor& dense,
                                          at::TensorList offsets,
                                          std::optional<c10::SymInt> total_L);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_
