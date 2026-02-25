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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_DECOMPOSITION_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_DECOMPOSITION_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

// On CUDA, when using the strided layout, the relationship between an index
// tuple and the linear byte index in the 1D storage buffer is described by
// four pieces of metadata:
//   * sizes:          the allowed indexes on each axis are [0, sizes[i]).
//   * strides:        the relative offset between adjacent elements in the
//                     dimension; increasing the index by 1 in this axis
//                     increases the element index by strides[i].
//   * storage_offset: the offset, in number of elements, from the start of the
//                     1D storage buffer to the first element of the tensor.
//   * dtype:          the data type of the tensor, which implies a byte size.
//
// The ops for "torch.as_strided" and "torch.Tensor.view(dtype)" allow users
// to manually set this metadata, but complex sequences of view ops can produce
// complex striding in their own right.
//
// However, StableHLO is a high-level representation, which does not have
// support for manual striding. This makes it necessary to decompose an
// arbitrary stride pattern into a sequence of supported StableHLO primitives.
//
// This library contains an algorithm for doing this decomposition, provided
// the original contiguous base tensor shape and the final strided layout.

namespace torch_tpu {

// Decomposes a final strided view of a contiguous base tensor into a sequence
// of view primitives that would reproduce that view when applied.
//
// An N-dimensional tensor is a contiguous base if it is:
//   * non-overlapping; each element is accessed by a unique index tuple.
//                      In particular, no dimension has stride 0.
//   * row-major;       stride[i] > stride[i+1]
//   * dense;           stride[i] == size[i+1] * stride[i+1] for i in [0, N-1)
//   * base;            storage_offset == 0
//   * non-empty;       must have at least one element (scalars allowed)
//
// These properties guarantee that the 1D storage representation of the tensor
// would cover the entire byte array in a predictable order with no gaps. The
// input contiguous_base_shape describes the original tensor shape, and asserts
// that it was contiguous on creation (allowing strides and storage offset to be
// inferred).
//
// For the decomposition to be possible, the view layout must:
//   * Have at least one element (scalars allowed)
//   * Have a non-negative storage offset
//   * Have a maximum byte index that is less than the number of original
//     bytes in the contiguous base tensor.
absl::StatusOr<ViewSequence> DecomposeIntoViewSequence(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype, bool is_conj = false);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_DECOMPOSITION_H_
