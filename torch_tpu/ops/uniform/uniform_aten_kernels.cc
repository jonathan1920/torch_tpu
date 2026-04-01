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

#include "torch_tpu/ops/uniform/uniform_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/uniform/uniform.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<1, 2> GetUniformFunctional(Dimensions dims,
                                             mlir::ElementType output_dtype,
                                             double from, double to) {
  return [dims, output_dtype, from, to](mlir::MlirOp rng_input_state) {
    return BuildUniformShlo(rng_input_state, from, to, dims, output_dtype);
  };
}

void CheckUniformPreconditions(const at::Tensor& self) {
  TT_CHECK_THROW(self.is_floating_point() || self.is_complex(),
                 error::kInvalidArgument)
      << "input tensor must be floating point or complex type, got "
      << ToString(self.scalar_type());
}

}  // namespace

at::Tensor& AtenUniform_(at::Tensor& self, double from, double to,
                         std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kUniform_, param_keys, (self, from, to, generator), {
    if (self.numel() == 0) {
      return self;
    }
    CheckUniformPreconditions(self);
    at::Tensor self_real = self.is_complex() ? AtenViewAsReal(self) : self;
    // Since we need to generate random bits, we query for the rng state tensor.
    TT_ASSIGN_OR_THROW(auto rng_input_state, GetRngState(generator));
    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self_real.scalar_type()));
    auto dims = CopyIntVector(self_real.sizes());
    TT_ASSIGN_OR_THROW(
        (auto [rng_output_state_buf, output_buf]),
        (DispatchOp<1, 2>(
            OpName::kUniform_,
            GetUniformFunctional(dims, output_dtype, from, to),
            {rng_input_state},
            {.out_dtypes = {mlir::ElementType::UI64, output_dtype},
             .out_dims_list = {{2}, self_real.sizes()},
             .op_param_cache_keys = std::move(param_keys)})));
    // After the state has been used (and updated) after generating random bits,
    // we give it back to the generator, so that it can be used by other ops in
    // the same graph.
    auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
    TT_THROW_IF_ERROR(UpdateRngState(generator, rng_output_state));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), self_real));
    return self;
  });
}

}  // namespace torch_tpu
