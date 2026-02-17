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

#include "torch_tpu/ops/foreach/foreach_add_aten_kernels.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
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

std::vector<DeviceBufferRef> ForeachAddList(
    at::TensorList self, at::TensorList other, const at::Scalar& alpha,
    const std::vector<mlir::ElementType>& out_dtypes,
    OpParamCacheKeys param_keys) {
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  size_t num_tensors = self.size();

  // Check for invalid input types.
  for (size_t i = 0; i < num_tensors; ++i) {
    TT_CHECK_THROW(!(c10::isIntegralType(self[i].scalar_type(), true) &&
                     c10::isIntegralType(other[i].scalar_type(), true) &&
                     !c10::isIntegralType(alpha.type(), true)),
                   error::kInvalidArgument)
        << "expected alpha to be integral for integral input tensors, got "
        << ToString(alpha.type());
    TT_CHECK_THROW(
        !alpha.isBoolean() || (self[i].scalar_type() == at::ScalarType::Bool &&
                               other[i].scalar_type() == at::ScalarType::Bool),
        error::kInvalidArgument)
        << "expected input tensor dtypes to be bool when alpha dtype is "
           "bool, got "
        << ToString(self[i].scalar_type()) << " and "
        << ToString(other[i].scalar_type());
  }
  // The op builder.
  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  auto op_builder = [alpha, num_tensors, out_dtypes](
                        absl::Span<mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    std::vector<mlir::MlirOp> alpha_ops;
    alpha_ops.reserve(num_tensors);
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(num_tensors);

    // If alpha is 1.0, do a simple addition without multiplying by alpha.
    if ((alpha.isIntegral(true) && alpha.to<int64_t>() == 1) ||
        (alpha.isFloatingPoint() && alpha.to<double>() == 1.0)) {
      return BuildForeachShlo(self_ops, other_ops, out_dtypes,
                              mlir::stablehlo::Add, builder);
    }
    for (int i = 0; i < num_tensors; ++i) {
      TT_ASSIGN_OR_RETURN(auto current_alpha_op,
                          MakeConstant(builder, alpha, out_dtypes[i]));
      alpha_ops.push_back(current_alpha_op);
    }
    TT_ASSIGN_OR_RETURN(
        auto new_other_ops,
        BuildForeachShlo(other_ops, absl::MakeSpan(alpha_ops), out_dtypes,
                         mlir::stablehlo::Mul, builder));
    return BuildForeachShlo(self_ops, absl::MakeSpan(new_other_ops), out_dtypes,
                            mlir::stablehlo::Add, builder);
  };

  // Dispatch the op and prepare results.
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachAddList, std::move(op_builder), inputs,
                         std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddScalar(
    at::TensorList self, const at::Scalar& scalar,
    const std::vector<mlir::ElementType>& out_dtypes,
    OpParamCacheKeys param_keys) {
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  auto op_builder = [scalar, out_dtypes](absl::Span<mlir::MlirOp> inputs,
                                         mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp scalar_op,
                          MakeConstant(builder, scalar, out_dtypes[i]));
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes,
                            mlir::stablehlo::Add, builder);
  };
  std::vector<at::Tensor> inputs(self.begin(), self.end());
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachAddScalar, std::move(op_builder),
                         inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddScalarList(
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
      TT_ASSIGN_OR_RETURN(auto scalar_op,
                          MakeConstant(builder, scalars_vec[i], out_dtypes[i]));
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes,
                            mlir::stablehlo::Add, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = absl::MakeConstSpan(out_dtypes),
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachAddScalarList, std::move(op_builder),
                         inputs, std::move(options))));
  return result_buffers;
}

}  // namespace

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAddList, param_keys, (self, other, alpha), {
    const auto out_dtypes = GetOutputDtypes(self, other);
    return ForeachConvertToTensor(
        ForeachAddList(self, other, alpha, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachAddScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachAddScalar, param_keys, (self, scalar), {
    const auto out_dtypes = GetOutputDtypes(self, scalar);
    return ForeachConvertToTensor(
        ForeachAddScalar(self, scalar, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachAddScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachAddScalarList, param_keys, (self, scalars), {
    const auto out_dtypes = GetOutputDtypes(self, scalars);
    return ForeachConvertToTensor(
        ForeachAddScalarList(self, scalars, out_dtypes, std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachAddTensor(at::TensorList self,
                                             const at::Tensor& other,
                                             const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAddTensor, _, (self, other, alpha), {
    std::vector<at::Tensor> other_list(self.size(), other);
    return AtenForeachAddList(self, other_list, alpha);
  });
}

void AtenForeachAdd_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAdd_List, param_keys, (self, other, alpha), {
    const auto out_dtypes = GetOutputDtypes(self);
    ForeachAssignToTensor(
        ForeachAddList(self, other, alpha, out_dtypes, std::move(param_keys)),
        self);
  });
}

void AtenForeachAdd_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachAdd_Scalar, param_keys, (self, scalar), {
    const auto out_dtypes = GetOutputDtypes(self);
    const auto result_out_dtypes = GetOutputDtypes(self, scalar);
    for (size_t i = 0; i < self.size(); ++i) {
      CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                      self[i].scalar_type(), scalar.type());
    }

    ForeachAssignToTensor(
        ForeachAddScalar(self, scalar, out_dtypes, std::move(param_keys)),
        self);
  });
}
void AtenForeachAdd_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachAdd_ScalarList, param_keys, (self, scalars), {
    const auto out_dtypes = GetOutputDtypes(self);
    const auto result_out_dtypes = GetOutputDtypes(self, scalars);
    for (size_t i = 0; i < self.size(); ++i) {
      CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                      self[i].scalar_type(), scalars[i].type());
    }
    ForeachAssignToTensor(
        ForeachAddScalarList(self, scalars, out_dtypes, std::move(param_keys)),
        self);
  });
}

void AtenForeachAdd_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAdd_Tensor, _, (self, other, alpha), {
    std::vector<at::Tensor> other_list(self.size(), other);
    AtenForeachAdd_List(self, other_list, alpha);
  });
}

}  // namespace torch_tpu
