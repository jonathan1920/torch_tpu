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

#include "torch_tpu/ops/linalg/qr/linalg_qr_kernels.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/linalg/qr/qr_lib.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRefArray<2>> Geqrf(const at::Tensor& self,
                                              OpParamCacheKeys param_keys) {
  constexpr mlir::ElementType kCastDtypeForIntegerInput =
      mlir::ElementType::F64;
  const bool input_is_integer = IsInteger(self);
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  if (input_is_integer) {
    out_dtype = kCastDtypeForIntegerInput;
  }

  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input to have at least 2 dimensions, got " << self.dim();
  const int64_t m = self.size(self.dim() - 2);
  const int64_t n = self.size(self.dim() - 1);
  Dimensions tau_dims(self.sizes().begin(), self.sizes().end() - 2);
  tau_dims.push_back(std::min(m, n));

  auto op_builder =
      [input_is_integer](
          mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    mlir::MlirOp cast_self_op = input_is_integer
                                    ? mlir::stablehlo::ConvertElementType(
                                          self_op, kCastDtypeForIntegerInput)
                                    : self_op;
    TT_ASSIGN_OR_RETURN(const MlirOpResults<2> result_ops,
                        BuildGeqrfShlo(cast_self_op));
    return result_ops;
  };

  return DispatchOp<1, 2>(std::move(op_builder), {self},
                          {.out_dtypes = {out_dtype, out_dtype},
                           .out_dims_list = {self.sizes(), tau_dims},
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenGeqrf(const at::Tensor& self) {
  TT_KERNEL(OpName::kGeqrf, param_keys, (self), {
    TT_ASSIGN_OR_THROW(const DeviceBufferRefArray<2> result_buffers,
                       Geqrf(self, std::move(param_keys)));
    return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1])};
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenGeqrfA(const at::Tensor& self,
                                                at::Tensor& a,
                                                at::Tensor& tau) {
  TT_KERNEL(OpName::kGeqrfA, param_keys, (self, a, tau), {
    TT_ASSIGN_OR_THROW(const DeviceBufferRefArray<2> result_buffers,
                       Geqrf(self, std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], a));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], tau));
    return {a, tau};
  });
}

}  // namespace torch_tpu
