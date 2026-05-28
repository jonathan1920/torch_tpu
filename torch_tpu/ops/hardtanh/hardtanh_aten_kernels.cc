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

#include "torch_tpu/ops/hardtanh/hardtanh_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {
absl::StatusOr<mlir::MlirOp> BuildHardtanhShlo(mlir::MlirOp input,
                                               mlir::MlirOp min_val,
                                               mlir::MlirOp max_val) {
  return mlir::stablehlo::Clamp(min_val, input, max_val);
}

// Helper to dispatch the hardtanh computation on the device.
absl::StatusOr<DeviceBufferRef> AtenHardtanhImpl(const at::Tensor& self,
                                                 PromotedScalar& promoted_min,
                                                 PromotedScalar& promoted_max,
                                                 OpParamCacheKeys param_keys) {
  auto scalar_type = self.scalar_type();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected a non-complex tensor, got " << ToString(scalar_type);
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "expected a non-boolean tensor, got " << ToString(scalar_type);
  TT_RET_CHECK(c10::isSignedType(scalar_type)  // MLIR_SIGNED_INT_OK=Checking
                                               // ATen scalar type signedness
                   || (promoted_min.scalar().toInt() >= 0 &&
                       promoted_max.scalar().toInt() >= 0),
               error::kInvalidArgument)
      << "expected positive limit values when executing on an unsigned tensor, "
      << "got min_val=" << promoted_min.scalar().toInt()
      << " and max_val=" << promoted_max.scalar().toInt();

  TT_ASSIGN_OR_RETURN(at::Tensor min_tensor,
                      promoted_min.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor max_tensor,
                      promoted_max.GetTensor(scalar_type));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, min_op, max_op] = inputs;
    return BuildHardtanhShlo(self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return DispatchOp<3>(op_builder, {self, min_tensor, max_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, promoted_min, promoted_max), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenHardtanhImpl(self, promoted_min, promoted_max,
                                        std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanh_, param_keys, (self, promoted_min, promoted_max), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanhOut, param_keys, (self, promoted_min, promoted_max, out),
      {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
