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

#include "torch_tpu/ops/reductions/var_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/conj_physical.h"
#include "ATen/ops/full.h"
#include "ATen/ops/mean.h"
#include "ATen/ops/mul.h"
#include "ATen/ops/sub.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/OptionalArrayRef.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {

at::Tensor CalculateSquaredDiff(const at::Tensor& self,
                                const c10::OptionalArrayRef<int64_t> dim) {
  at::Tensor mean_tensor = at::mean(self, dim, /*keepdim=*/true);
  at::Tensor diff_tensor = at::sub(self, mean_tensor);
  // Generic calculation that works for any dtype including complex.
  return at::mul(diff_tensor, at::conj_physical(diff_tensor));
}

// Variance normalization factor is (number of elements in sum) - correction.
// If this value is <= 0 issue a warning but go ahead with division by 0.
double CalculateNormalizationFactor(const std::optional<at::Scalar>& correction,
                                    int64_t reduction_size) {
  const double correction_val =
      correction.has_value() ? correction->toDouble() : 1.0;
  const double denom = reduction_size - correction_val;
  if (denom > 0) {
    return denom;
  }
  TORCH_WARN(GetRootOpName(OpName::kVar),
             "(): degrees of freedom (i.e., reduction "
             "size - correction) should "
             "be positive, got reduction size = ",
             reduction_size, ", correction = ", correction,
             ", and degrees of freedom = ", denom);
  return 0.0;
}

}  // namespace

at::Tensor AtenVar(const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
                   const std::optional<at::Scalar>& correction, bool keep_dim) {
  TT_KERNEL(OpName::kVar, _, (self, dim, correction, keep_dim), {
    c10::ScalarType scalar_dtype = self.scalar_type();
    TT_THROW_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
    if (c10::isComplexType(scalar_dtype)) {
      scalar_dtype = c10::toRealValueType(scalar_dtype);
    }
    if (self.numel() == 0)
      return at::full({}, std::numeric_limits<double>::quiet_NaN(),
                      scalar_dtype);

    at::Tensor diff_sq_tensor = CalculateSquaredDiff(self, dim);
    TT_ASSIGN_OR_THROW(at::Tensor sum_sq_tensor,
                       ApplySumReduction(diff_sq_tensor, dim,
                                         keep_dim ? ReductionMode::kKeepDims
                                                  : ReductionMode::kDropDims,
                                         scalar_dtype));

    TT_ASSIGN_OR_THROW(int64_t reduction_size, GetReductionFactor(self, dim));
    const double denom =
        CalculateNormalizationFactor(correction, reduction_size);
    return sum_sq_tensor.div(denom);
  });
}

at::Tensor& AtenVarOut(const at::Tensor& self,
                       c10::OptionalArrayRef<int64_t> dim,
                       const std::optional<at::Scalar>& correction,
                       bool keep_dim, at::Tensor& out) {
  TT_KERNEL(OpName::kVarOut, _, (self, dim, correction, keep_dim, out), {
    c10::ScalarType scalar_dtype = out.scalar_type();
    TT_THROW_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
    if (self.numel() == 0) {
      at::full_out(out, {}, std::numeric_limits<double>::quiet_NaN());
      return out;
    }

    at::Tensor diff_sq_tensor = CalculateSquaredDiff(self, dim);
    TT_THROW_IF_ERROR(ApplySumReductionOut(
        diff_sq_tensor, out, dim,
        keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims,
        scalar_dtype));

    TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                       GetReductionFactor(self, dim));
    const double denom =
        CalculateNormalizationFactor(correction, reduction_size);
    out.div_(denom);
    return out;
  });
}

}  // namespace torch_tpu
