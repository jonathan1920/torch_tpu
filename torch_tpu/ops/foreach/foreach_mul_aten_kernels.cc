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

#include "torch_tpu/ops/foreach/foreach_mul_aten_kernels.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/cache_key.h"
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

std::vector<DeviceBufferRef> ForeachMulList(
    at::TensorList self, at::TensorList other,
    const std::vector<mlir::ElementType>& out_dtypes,
    OpParamCacheKeys param_keys) {
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  size_t num_tensors = self.size();

  auto op_builder = [num_tensors, out_dtypes](absl::Span<mlir::MlirOp> inputs,
                                              mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    return BuildForeachShlo(self_ops, other_ops, out_dtypes,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  // Dispatch the op and prepare results.
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
  };
  // Dispatch the op and prepare results.
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachMulList, std::move(op_builder), inputs,
                         std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachMulScalar(
    at::TensorList self, const at::Scalar& scalar,
    const std::vector<mlir::ElementType>& out_dtypes,
    OpParamCacheKeys param_keys) {
  auto op_builder = [scalar, out_dtypes](absl::Span<mlir::MlirOp> inputs,
                                         mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      absl::StatusOr<mlir::MlirOp> scalar_op_status =
          MakeConstant(builder, scalar, out_dtypes[i]);
      if (!scalar_op_status.ok()) {
        return scalar_op_status.status();
      }
      mlir::MlirOp scalar_op = *scalar_op_status;
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachMulScalar, std::move(op_builder),
                         inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars,
    const std::vector<mlir::ElementType>& out_dtypes,
    OpParamCacheKeys param_keys) {
  const std::vector<at::Scalar> scalars_vec(scalars.begin(), scalars.end());
  auto op_builder = [scalars_vec, out_dtypes](absl::Span<mlir::MlirOp> inputs,
                                              mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      absl::StatusOr<mlir::MlirOp> scalar_op_status =
          MakeConstant(builder, scalars_vec[i], out_dtypes[i]);
      if (!scalar_op_status.ok()) {
        return scalar_op_status.status();
      }
      mlir::MlirOp scalar_op = *scalar_op_status;
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachMulScalarList, std::move(op_builder),
                         inputs, std::move(options))));
  return result_buffers;
}

}  // namespace

std::vector<at::Tensor> AtenForeachMulList(at::TensorList self,
                                           at::TensorList other) {
  TT_KERNEL(OpName::kForeachMulList, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self, other));
    return ForeachConvertToTensor(
        ForeachMulList(self, other, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachMulScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachMulScalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self, scalar));
    return ForeachConvertToTensor(
        ForeachMulScalar(self, scalar, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachMulScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self, scalars));
    return ForeachConvertToTensor(
        ForeachMulScalarList(self, scalars, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachMulTensor(at::TensorList self,
                                             const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachMulTensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    return AtenForeachMulList(self, other_list);
  });
}

void AtenForeachMul_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMul_List, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulList(self, other, out_dtypes, std::move(param_keys)), self));
  });
}

void AtenForeachMul_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachMul_Scalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalar));
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                                        self[i].scalar_type(), scalar.type()));
    }

    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulScalar(self, scalar, out_dtypes, std::move(param_keys)),
        self));
  });
}

void AtenForeachMul_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachMul_ScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(const auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalars));
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                                        self[i].scalar_type(),
                                        scalars[i].type()));
    }

    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulScalarList(self, scalars, out_dtypes, std::move(param_keys)),
        self));
  });
}

void AtenForeachMul_Tensor(at::TensorList self, const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachMul_Tensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    AtenForeachMul_List(self, other_list);
  });
}

}  // namespace torch_tpu
