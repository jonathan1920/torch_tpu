// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/unfold/unfold_aten_kernels.h"

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Optional.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

at::Tensor AtenUnfold(const at::Tensor& self, int64_t dimension, int64_t size,
                      int64_t step) {
  TT_KERNEL(
      OpName::kUnfold, _,
      (self, IgnoreInCacheKey(dimension, "Delegates to AtenAsStrided"),
       IgnoreInCacheKey(size, "Delegates to AtenAsStrided"),
       IgnoreInCacheKey(step, "Delegates to AtenAsStrided")),
      {
        const bool self_is_scalar = self.dim() == 0;
        at::Tensor self_reshaped = self_is_scalar ? self.unsqueeze(0) : self;
        int64_t dims = self_reshaped.dim();

        TT_CHECK_THROW(step > 0, error::kInvalidArgument)
            << "expected step > 0, got " << step;

        TT_CHECK_THROW(dimension >= -dims && dimension < dims,
                       error::kOutOfRange)
            << "expected dimension to be in range of [" << -dims << ", "
            << dims - 1 << "] for shape " << self.sizes() << ", got "
            << dimension;

        TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                           SafeWrapDim(dimension, dims));
        TT_CHECK_THROW(size <= self_reshaped.size(normalized_dim),
                       error::kInvalidArgument)
            << "expected size <= dimension size (shape[" << dimension
            << "]: " << self_reshaped.size(normalized_dim)
            << "), got size: " << size;

        if (self_is_scalar) {
          return size == 0 ? at::empty({0}, self.options()) : self_reshaped;
        }

        // Compute the new layout.
        const int64_t num_windows =
            (self.size(normalized_dim) - size) / step + 1;
        Dimensions new_size = CopyIntVector(self.sizes());
        Strides new_stride = CopyIntVector(self.strides());
        // The unfolded dimension goes from old_size, old_stride
        // to num_windows, old_stride * step.
        new_size[normalized_dim] = num_windows;
        new_stride[normalized_dim] *= step;
        // A new dimension is added with the given size and the same stride as
        // the pre-unfolded dimension.
        new_size.push_back(size);
        new_stride.push_back(self.stride(normalized_dim));

        c10::SymIntArrayRef new_size_sym =
            c10::fromIntArrayRefKnownNonNegative(new_size);
        c10::SymIntArrayRef new_stride_sym =
            c10::fromIntArrayRefKnownNonNegative(new_stride);
        c10::optional<c10::SymInt> storage_offset_sym_opt = c10::nullopt;

        return AtenAsStrided(self, new_size_sym, new_stride_sym,
                             storage_offset_sym_opt);
      });
}

}  // namespace torch_tpu
