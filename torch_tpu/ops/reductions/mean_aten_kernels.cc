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

#include "torch_tpu/ops/reductions/mean_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {

// Get the output scalar type and throw if it is float or complex.
// Types comes from dtype if it has a value, otherwise it is tensor's type.
absl::StatusOr<c10::ScalarType> GetOutputScalarType(
    const at::Tensor& tensor, std::optional<c10::ScalarType> dtype) {
  c10::ScalarType scalar_dtype = dtype.value_or(tensor.scalar_type());
  TT_RETURN_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
  return scalar_dtype;
}

// Divides the reduced tensor by the reduction factor (i.e. the input:output
// ratio). If dim is specified then the reduction factor is the product of
// the size along each summed dimension. Otherwise the reduction factor is
// the number of elements in the input tensor.
absl::Status DivideByReductionFactor(const at::Tensor& input_tensor,
                                     at::Tensor& reduced_tensor,
                                     c10::OptionalArrayRef<int64_t> dim) {
  TT_ASSIGN_OR_RETURN(const int64_t reduction_factor,
                      GetReductionFactor(input_tensor, dim));
  reduced_tensor.div_(reduction_factor);
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenMeanOut(const at::Tensor& self,
                        c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                        std::optional<c10::ScalarType> dtype, at::Tensor& out) {
  TT_KERNEL(
      OpName::kMeanOut, _,
      (self, IgnoreInCacheKey(dim), IgnoreInCacheKey(keep_dim),
       IgnoreInCacheKey(dtype), out),
      {
        TT_ASSIGN_OR_THROW(c10::ScalarType scalar_dtype,
                           GetOutputScalarType(out, dtype));
        if (self.numel() == 0) {
          at::full_out(out, {}, std::numeric_limits<double>::quiet_NaN());
          return out;
        }

        TT_THROW_IF_ERROR(ApplySumReductionOut(
            self, out, dim,
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims,
            scalar_dtype));
        TT_THROW_IF_ERROR(DivideByReductionFactor(self, out, dim));
        return out;
      });
}

}  // namespace torch_tpu
