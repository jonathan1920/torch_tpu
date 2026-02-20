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

#include "torch_tpu/ops/foreach/unary_foreach_aten_kernels.h"

#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/foreach/utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {
std::vector<DeviceBufferRef> ForeachUnaryOp(
    at::TensorList self, OpName op_name,
    const std::vector<mlir::ElementType>& out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&) const> tensor_transform) {
  auto op_builder =
      [out_dtypes, tensor_transform = std::move(tensor_transform)](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp casted_input,
                          CastIfNeeded(inputs[i], out_dtypes[i]));
      results.push_back(tensor_transform(casted_input));
    }
    return results;
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<torch_tpu::kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = out_dims_list,
  };
  TT_ASSIGN_OR_THROW(
      auto result_buffers,
      (DispatchOp<kDynamicSize, kDynamicSize>(op_name, std::move(op_builder),
                                              inputs, std::move(options))));
  return result_buffers;
}
}  // namespace

std::vector<DeviceBufferRef> ForeachSqrt(
    at::TensorList self, const std::vector<mlir::ElementType>& out_dtypes) {
  auto tensor_transform = [](mlir::MlirOp input) {
    return mlir::stablehlo::Sqrt(input);
  };
  return ForeachUnaryOp(self, OpName::kForeachSqrt, out_dtypes,
                        std::move(tensor_transform));
}

std::vector<at::Tensor> AtenForeachSqrt(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetFloatingOutputDtypes(self));
    return ForeachConvertToTensor(ForeachSqrt(self, out_dtypes));
  });
}

void AtenForeachSqrt_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt_, _, (self), {
    TT_THROW_IF_ERROR(EnsureNotIntegral(self));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(
        ForeachAssignToTensor(ForeachSqrt(self, out_dtypes), self));
  });
}

std::vector<DeviceBufferRef> ForeachNeg(at::TensorList self) {
  TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
  return ForeachUnaryOp(self, OpName::kForeachNeg, out_dtypes,
                        mlir::stablehlo::Neg);
}

std::vector<at::Tensor> AtenForeachNeg(at::TensorList self) {
  TT_KERNEL(OpName::kForeachNeg, _, (self),
            { return ForeachConvertToTensor(ForeachNeg(self)); });
}

void AtenForeachNeg_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachNeg, _, (self), {
    TT_THROW_IF_ERROR(ForeachAssignToTensor(ForeachNeg(self), self));
  });
}

std::vector<DeviceBufferRef> ForeachReciprocal(
    at::TensorList self, const std::vector<mlir::ElementType>& out_dtypes) {
  auto tensor_transform = [](mlir::MlirOp input) {
    mlir::MlirOp one_scalar = MakeConstantLike(input, 1.0);
    return mlir::stablehlo::Div(one_scalar, input);
  };
  return ForeachUnaryOp(self, OpName::kForeachReciprocal, out_dtypes,
                        std::move(tensor_transform));
}

std::vector<at::Tensor> AtenForeachReciprocal(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetFloatingOutputDtypes(self));
    return ForeachConvertToTensor(ForeachReciprocal(self, out_dtypes));
  });
}

void AtenForeachReciprocal_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal_, _, (self), {
    TT_THROW_IF_ERROR(EnsureNotIntegral(self));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(
        ForeachAssignToTensor(ForeachReciprocal(self, out_dtypes), self));
  });
}

void AtenForeachZero_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachZero_, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    auto tensor_transform = [](mlir::MlirOp input) {
      return MakeConstantLike(input, 0.0);
    };
    auto result_buffers = ForeachUnaryOp(
        self, OpName::kForeachZero_, out_dtypes, std::move(tensor_transform));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}
}  // namespace torch_tpu
