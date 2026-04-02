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

#include "torch_tpu/ops/gelu/gelu_aten_kernels.h"

#include <functional>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/gelu/gelu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

MlirUnaryOpBuilder GetGeluFunctional(c10::string_view approximate) {
  return std::bind(&BuildGeluShlo, std::placeholders::_1, approximate);
}

}  // namespace

at::Tensor& AtenGeluOut(const at::Tensor& self, c10::string_view approximate,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kGeluOut, param_keys, (self, approximate, out), {
    TT_CHECK_THROW(approximate == "none" || approximate == "tanh",
                   error::kInvalidArgument)
        << "unsupported approximate argument: " << approximate;
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, OpName::kGeluOut, GetGeluFunctional(approximate),
                   {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenGeluBackwardGradInput(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      c10::string_view approximate,
                                      at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kGeluBackwardGradInput, param_keys,
      (grad_output, self, approximate, grad_input), {
        TT_CHECK_THROW(approximate == "none" || approximate == "tanh",
                       error::kInvalidArgument)
            << "unsupported approximate argument: " << approximate;
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [approximate](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, input_op] = inputs;
          return BuildGeluBackwardGradInputShlo(grad_output_op, input_op,
                                                approximate);
        };
        auto output_shape = CopyIntVector(self.sizes());

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<2>(OpName::kGeluBackwardGradInput, std::move(op_builder),
                          {grad_output, self},
                          /*options=*/
                          {.out_dtype = out_dtype,
                           .out_dims = output_shape,
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
