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

#include "torch_tpu/ops/leaky_relu/leaky_relu_aten_kernels.h"

#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildLeakyReluShlo(
    mlir::MlirOp self_op, const at::Scalar& negative_slope,
    mlir::ElementType out_dtype, Dimensions result_shape) {
  auto& builder = self_op.getBuilder();
  mlir::MlirOp zero_op = MakeConstant(builder, 0.0, out_dtype, result_shape);
  mlir::MlirOp compare_ge_zero = mlir::stablehlo::Compare(
      self_op, zero_op, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp negative_slope_op =
      MakeConstant(builder, negative_slope.toDouble(), out_dtype, result_shape);
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(self_op, negative_slope_op);
  return mlir::stablehlo::Select(compare_ge_zero, self_op, mul_op);
}

absl::StatusOr<DeviceBufferRef> LeakyReluShlo(const at::Tensor& self,
                                              const at::Scalar& negative_slope,
                                              const at::Tensor& out) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  auto result_shape = CopyIntVector(self.sizes());
  auto op_builder = [negative_slope, out_dtype, result_shape](
                        mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildLeakyReluShlo(self_op, negative_slope, out_dtype, result_shape);
  };
  return DispatchOp<1>(OpName::kLeakyReluOut, std::move(op_builder), {self},
                       {.out_dtype = out_dtype,
                        .out_dims = result_shape,
                        .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

}  // namespace

at::Tensor& AtenLeakyReluOut(const at::Tensor& self,
                             const at::Scalar& negative_slope,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kLeakyReluOut, _,
      (self, IgnoreInCacheKey(negative_slope, "Legacy usage"), out), {
        TT_ASSIGN_OR_THROW(mlir::ElementType dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_CHECK_THROW(!IsBoolean(dtype), error::kInvalidArgument)
            << "boolean dtypes are not supported, got " << self.scalar_type();
        TT_CHECK_THROW(!IsInteger(dtype, /*includeBool=*/false),
                       error::kInvalidArgument)
            << "integer dtypes are not supported, got " << self.scalar_type();
        TT_CHECK_THROW(!IsComplex(dtype), error::kInvalidArgument)
            << "complex dtypes are not supported, got " << self.scalar_type();
        TT_ASSIGN_OR_THROW(auto result_buf,
                           LeakyReluShlo(self, negative_slope, out));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
