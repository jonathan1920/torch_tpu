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

#include "torch_tpu/ops/foreach_aten_kernels.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

c10::ScalarType GetOutputDtypeFromTensorAndScalar(const at::Tensor& tensor,
                                                  const at::Scalar& scalar) {
  if ((IsFloatingPoint(tensor) && scalar.isFloatingPoint()) ||
      (IsInteger(tensor) && scalar.isIntegral(/*includeBool=*/true))) {
    // If both tensor and scalar are floating point or integral, then use the
    // tensor's dtype, unless the tensor is boolean (see below).
    return tensor.scalar_type();
  } else if (IsIntegral(tensor) && scalar.isFloatingPoint()) {
    // If tensor is integral and scalar is floating point, then use the
    // default floating point dtype.
    return c10::get_default_dtype_as_scalartype();
  } else {
    // If the tensor is boolean and the scalar is integer, just normally
    // promote the types to the scalar's integer type.
    return c10::promoteTypes(tensor.scalar_type(), scalar.type());
  }
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&, mlir::MlirOp&)>
        tensor_transform,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_other = other[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_other,
                        CastIfNeeded(current_other, out_dtypes[i]));

    std::array<mlir::MlirOp, 2> broadcasted_ops;
    TT_ASSIGN_OR_RETURN(broadcasted_ops,
                        ApplyBroadcastIfNeeded(current_self, current_other));
    current_self = broadcasted_ops[0];
    current_other = broadcasted_ops[1];

    mlir::MlirOp result = tensor_transform(current_self, current_other);
    results.push_back(result);
  }
  return results;
}

std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers) {
  std::vector<at::Tensor> result;
  result.reserve(result_buffers.size());
  for (auto& result_buffer : result_buffers) {
    result.push_back(MakeTensor(std::move(result_buffer)));
  }
  return result;
}

absl::Status ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                                   at::TensorList self) {
  for (int i = 0; i < result_buffers.size(); ++i) {
    TT_RETURN_IF_ERROR(
        AssignBufferToAtTensor(std::move(result_buffers[i]), self[i]));
  }
  return absl::OkStatus();
}

std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list) {
  std::vector<absl::Span<const int64_t>> dims_list;
  dims_list.reserve(tensor_list.size());
  for (int i = 0; i < tensor_list.size(); ++i) {
    dims_list.push_back(tensor_list[i].sizes());
  }
  return dims_list;
}

using DtypeVec = std::vector<mlir::ElementType>;
using UniqueDtypeVec = absl_nonnull std::unique_ptr<DtypeVec>;
using DtypeSpan = absl::Span<const mlir::ElementType>;

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(at::TensorList self) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    TT_ASSIGN_OR_RETURN(auto output_element_type,
                        ConvertTo<mlir::ElementType>(self[i].scalar_type()));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(at::TensorList self,
                                               at::TensorList other) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(at::TensorList self,
                                               const at::Scalar& scalar) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalar);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetFloatingOutputDtypes(at::TensorList self) {
  const at::ScalarType default_dtype = c10::get_default_dtype_as_scalartype();
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    c10::ScalarType tensor_type;
    if (IsIntegral(self[i])) {
      tensor_type = default_dtype;
    } else {
      tensor_type = self[i].scalar_type();
    }
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(tensor_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalars[i]);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

// Helper functions for input validation.

absl::Status CheckScalarType(mlir::ElementType out_dtype,
                             mlir::ElementType compute_dtype,
                             at::ScalarType tensor_type,
                             at::ScalarType scalar_type) {
  TT_RET_CHECK(out_dtype == compute_dtype, error::kInvalidArgument)
      << "expected the scalar dtype to be castable to the tensor dtype "
         "(e.g. bool to int or int to float), got "
      << ToString(scalar_type) << " and " << ToString(tensor_type);
  return absl::OkStatus();
}

absl::Status EnsureNotIntegral(at::TensorList self) {
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RET_CHECK(!IsIntegral(self[i]), error::kInvalidArgument)
        << "expected input tensor dtype to be non-integral, got "
        << ToString(self[i].scalar_type());
  }
  return absl::OkStatus();
}

inline absl::Status EnsureNotBool(const at::Scalar& scalar) {
  TT_RET_CHECK(!IsBool(scalar), error::kInvalidArgument)
      << "bool dtype is not supported";
  return absl::OkStatus();
}

inline absl::Status EnsureNotBool(const at::Tensor& tensor) {
  TT_RET_CHECK(!IsBool(tensor), error::kInvalidArgument)
      << "bool dtype is not supported";
  return absl::OkStatus();
}

absl::Status EnsureNotBool(at::TensorList self) {
  for (const auto& tensor : self) {
    TT_RETURN_IF_ERROR(EnsureNotBool(tensor));
  }
  return absl::OkStatus();
}

absl::Status EnsureNotBool(at::ArrayRef<at::Scalar> scalars) {
  for (const auto& scalar : scalars) {
    TT_RETURN_IF_ERROR(EnsureNotBool(scalar));
  }
  return absl::OkStatus();
}

std::vector<DeviceBufferRef> ForeachUnaryOp(
    at::TensorList self, OpName op_name, UniqueDtypeVec out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&) const> tensor_transform) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  auto op_builder = [out_dtypes = std::move(out_dtypes), out_dtypes_span,
                     tensor_transform = std::move(tensor_transform)](
                        absl::Span<mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp casted_input,
                          CastIfNeeded(inputs[i], out_dtypes_span[i]));
      results.push_back(tensor_transform(casted_input));
    }
    return results;
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<torch_tpu::kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = out_dims_list,
  };

  TT_ASSIGN_OR_THROW(
      auto result_buffers,
      (DispatchOp<kDynamicSize, kDynamicSize>(op_name, std::move(op_builder),
                                              inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddList(at::TensorList self,
                                            at::TensorList other,
                                            const at::Scalar& alpha,
                                            UniqueDtypeVec out_dtypes,
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
  const DtypeSpan out_dtypes_span = *out_dtypes;
  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  auto op_builder = [alpha, num_tensors, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
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
      return BuildForeachShlo(self_ops, other_ops, out_dtypes_span,
                              mlir::stablehlo::Add, builder);
    }
    for (int i = 0; i < num_tensors; ++i) {
      TT_ASSIGN_OR_RETURN(auto current_alpha_op,
                          MakeConstant(builder, alpha, out_dtypes_span[i]));
      alpha_ops.push_back(current_alpha_op);
    }
    TT_ASSIGN_OR_RETURN(
        auto new_other_ops,
        BuildForeachShlo(other_ops, absl::MakeSpan(alpha_ops), out_dtypes_span,
                         mlir::stablehlo::Mul, builder));
    return BuildForeachShlo(self_ops, absl::MakeSpan(new_other_ops),
                            out_dtypes_span, mlir::stablehlo::Add, builder);
  };

  // Dispatch the op and prepare results.
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachAddList, std::move(op_builder), inputs,
                         std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddScalar(at::TensorList self,
                                              const at::Scalar& scalar,
                                              UniqueDtypeVec out_dtypes,
                                              OpParamCacheKeys param_keys) {
  const auto out_dims_list = GetDimsList(self);
  const DtypeSpan out_dtypes_span = *out_dtypes;
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  auto op_builder = [scalar, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp scalar_op,
                          MakeConstant(builder, scalar, out_dtypes_span[i]));
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes_span,
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
    UniqueDtypeVec out_dtypes, OpParamCacheKeys param_keys) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  const std::vector<at::Scalar> scalars_vec(scalars.begin(), scalars.end());
  auto op_builder = [scalars_vec, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto scalar_op, MakeConstant(builder, scalars_vec[i],
                                                       out_dtypes_span[i]));
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes_span,
                            mlir::stablehlo::Add, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         OpName::kForeachAddScalarList, std::move(op_builder),
                         inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachMulList(at::TensorList self,
                                            at::TensorList other,
                                            UniqueDtypeVec out_dtypes,
                                            OpParamCacheKeys param_keys) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  const size_t num_tensors = self.size();
  auto op_builder = [num_tensors, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    return BuildForeachShlo(self_ops, other_ops, out_dtypes_span,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
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

std::vector<DeviceBufferRef> ForeachMulScalar(at::TensorList self,
                                              const at::Scalar& scalar,
                                              UniqueDtypeVec out_dtypes,
                                              OpParamCacheKeys param_keys) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  auto op_builder = [scalar, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      absl::StatusOr<mlir::MlirOp> scalar_op_status =
          MakeConstant(builder, scalar, out_dtypes_span[i]);
      if (!scalar_op_status.ok()) {
        return scalar_op_status.status();
      }
      mlir::MlirOp scalar_op = *scalar_op_status;
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes_span,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
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
    UniqueDtypeVec out_dtypes, OpParamCacheKeys param_keys) {
  DtypeSpan out_dtypes_span = *out_dtypes;
  const std::vector<at::Scalar> scalars_vec(scalars.begin(), scalars.end());
  auto op_builder = [scalars_vec, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    std::vector<mlir::MlirOp> scalar_ops;
    scalar_ops.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      absl::StatusOr<mlir::MlirOp> scalar_op_status =
          MakeConstant(builder, scalars_vec[i], out_dtypes_span[i]);
      if (!scalar_op_status.ok()) {
        return scalar_op_status.status();
      }
      mlir::MlirOp scalar_op = *scalar_op_status;
      scalar_ops.push_back(scalar_op);
    }
    return BuildForeachShlo(inputs, absl::MakeSpan(scalar_ops), out_dtypes_span,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
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

std::vector<DeviceBufferRef> ForeachSqrt(at::TensorList self,
                                         UniqueDtypeVec out_dtypes) {
  auto tensor_transform = [](mlir::MlirOp input) {
    return mlir::stablehlo::Sqrt(input);
  };
  return ForeachUnaryOp(self, OpName::kForeachSqrt, std::move(out_dtypes),
                        std::move(tensor_transform));
}

std::vector<at::Tensor> AtenForeachSqrt(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetFloatingOutputDtypes(self));
    return ForeachConvertToTensor(ForeachSqrt(self, std::move(out_dtypes)));
  });
}

void AtenForeachSqrt_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt_, _, (self), {
    TT_THROW_IF_ERROR(EnsureNotIntegral(self));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(
        ForeachAssignToTensor(ForeachSqrt(self, std::move(out_dtypes)), self));
  });
}

std::vector<DeviceBufferRef> ForeachNeg(at::TensorList self) {
  TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
  return ForeachUnaryOp(self, OpName::kForeachNeg, std::move(out_dtypes),
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

std::vector<DeviceBufferRef> ForeachReciprocal(at::TensorList self,
                                               UniqueDtypeVec out_dtypes) {
  auto tensor_transform = [](mlir::MlirOp input) {
    mlir::MlirOp one_scalar = MakeConstantLike(input, 1.0);
    return mlir::stablehlo::Div(one_scalar, input);
  };
  return ForeachUnaryOp(self, OpName::kForeachReciprocal, std::move(out_dtypes),
                        std::move(tensor_transform));
}

std::vector<at::Tensor> AtenForeachReciprocal(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetFloatingOutputDtypes(self));
    return ForeachConvertToTensor(
        ForeachReciprocal(self, std::move(out_dtypes)));
  });
}

void AtenForeachReciprocal_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal_, _, (self), {
    TT_THROW_IF_ERROR(EnsureNotIntegral(self));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachReciprocal(self, std::move(out_dtypes)), self));
  });
}

void AtenForeachZero_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachZero_, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    auto tensor_transform = [](mlir::MlirOp input) {
      return MakeConstantLike(input, 0.0);
    };
    auto result_buffers =
        ForeachUnaryOp(self, OpName::kForeachZero_, std::move(out_dtypes),
                       std::move(tensor_transform));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAddList, param_keys, (self, other, alpha), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));
    return ForeachConvertToTensor(ForeachAddList(
        self, other, alpha, std::move(out_dtypes), std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachAddScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachAddScalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    return ForeachConvertToTensor(ForeachAddScalar(
        self, scalar, std::move(out_dtypes), std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachAddScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachAddScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    return ForeachConvertToTensor(ForeachAddScalarList(
        self, scalars, std::move(out_dtypes), std::move(param_keys)));
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
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddList(self, other, alpha, std::move(out_dtypes),
                       std::move(param_keys)),
        self));
  });
}

void AtenForeachAdd_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachAdd_Scalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalar));
    const DtypeSpan out_dtypes_span = *out_dtypes;
    const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(CheckScalarType(out_dtypes_span[i],
                                        result_out_dtypes_span[i],
                                        self[i].scalar_type(), scalar.type()));
    }

    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddScalar(self, scalar, std::move(out_dtypes),
                         std::move(param_keys)),
        self));
  });
}

void AtenForeachAdd_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachAdd_ScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalars));
    const DtypeSpan out_dtypes_span = *out_dtypes;
    const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(
          CheckScalarType(out_dtypes_span[i], result_out_dtypes_span[i],
                          self[i].scalar_type(), scalars[i].type()));
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddScalarList(self, scalars, std::move(out_dtypes),
                             std::move(param_keys)),
        self));
  });
}

void AtenForeachAdd_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAdd_Tensor, _, (self, other, alpha), {
    std::vector<at::Tensor> other_list(self.size(), other);
    AtenForeachAdd_List(self, other_list, alpha);
  });
}

std::vector<at::Tensor> AtenForeachDivList(at::TensorList self,
                                           at::TensorList other) {
  TT_KERNEL(OpName::kForeachDivList, _, (self, other),
            { return AtenForeachMulList(self, AtenForeachReciprocal(other)); });
}

std::vector<at::Tensor> AtenForeachDivScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachDivScalar, _, (self, scalar),
            { return AtenForeachMulScalar(self, 1.0 / scalar.to<double>()); });
}

std::vector<at::Tensor> AtenForeachDivScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachDivScalarList, _, (self, scalars), {
    std::vector<at::Scalar> reciprocal_scalars;
    reciprocal_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
      reciprocal_scalars.push_back(1.0 / scalar.to<double>());
    }
    return AtenForeachMulScalarList(self, reciprocal_scalars);
  });
}

std::vector<at::Tensor> AtenForeachDivTensor(at::TensorList self,
                                             const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachDivTensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    return AtenForeachDivList(self, other_list);
  });
}

void AtenForeachDiv_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachDiv_List, _, (self, other),
            { AtenForeachMul_List(self, AtenForeachReciprocal(other)); });
}

void AtenForeachDiv_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachDiv_Scalar, _, (self, scalar),
            { AtenForeachMul_Scalar(self, 1.0 / scalar.to<double>()); });
}

void AtenForeachDiv_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachDiv_ScalarList, _, (self, scalars), {
    std::vector<at::Scalar> reciprocal_scalars;
    reciprocal_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
      reciprocal_scalars.push_back(1.0 / scalar.to<double>());
    }
    AtenForeachMul_ScalarList(self, reciprocal_scalars);
  });
}

void AtenForeachDiv_Tensor(at::TensorList self, const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachDiv_Tensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    AtenForeachDiv_List(self, other_list);
  });
}

std::vector<at::Tensor> AtenForeachMulList(at::TensorList self,
                                           at::TensorList other) {
  TT_KERNEL(OpName::kForeachMulList, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));
    return ForeachConvertToTensor(ForeachMulList(
        self, other, std::move(out_dtypes), std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachMulScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachMulScalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    return ForeachConvertToTensor(ForeachMulScalar(
        self, scalar, std::move(out_dtypes), std::move(param_keys)));
  });
}

std::vector<at::Tensor> AtenForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachMulScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    return ForeachConvertToTensor(ForeachMulScalarList(
        self, scalars, std::move(out_dtypes), std::move(param_keys)));
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
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(
        ForeachAssignToTensor(ForeachMulList(self, other, std::move(out_dtypes),
                                             std::move(param_keys)),
                              self));
  });
}

void AtenForeachMul_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachMul_Scalar, param_keys, (self, scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalar));
    const DtypeSpan out_dtypes_span = *out_dtypes;
    const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(CheckScalarType(out_dtypes_span[i],
                                        result_out_dtypes_span[i],
                                        self[i].scalar_type(), scalar.type()));
    }

    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulScalar(self, scalar, std::move(out_dtypes),
                         std::move(param_keys)),
        self));
  });
}

void AtenForeachMul_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachMul_ScalarList, param_keys, (self, scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(const auto result_out_dtypes,
                       GetOutputDtypes(self, scalars));
    const DtypeSpan out_dtypes_span = *out_dtypes;
    const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
    for (size_t i = 0; i < self.size(); ++i) {
      TT_THROW_IF_ERROR(
          CheckScalarType(out_dtypes_span[i], result_out_dtypes_span[i],
                          self[i].scalar_type(), scalars[i].type()));
    }

    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulScalarList(self, scalars, std::move(out_dtypes),
                             std::move(param_keys)),
        self));
  });
}

void AtenForeachMul_Tensor(at::TensorList self, const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachMul_Tensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    AtenForeachMul_List(self, other_list);
  });
}

std::vector<at::Tensor> AtenForeachSubList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachSubList, _, (self, other, alpha), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(alpha));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    TT_THROW_IF_ERROR(EnsureNotBool(other));
    return AtenForeachAddList(self, other, -alpha);
  });
}

void AtenForeachSub_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachSub_List, _, (self, other, alpha), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(alpha));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    TT_THROW_IF_ERROR(EnsureNotBool(other));
    AtenForeachAdd_List(self, other, -alpha);
  });
}

std::vector<at::Tensor> AtenForeachSubScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachSubScalar, _, (self, scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(scalar));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    return AtenForeachAddScalar(self, -scalar);
  });
}

void AtenForeachSub_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachSub_Scalar, _, (self, scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(scalar));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    AtenForeachAdd_Scalar(self, -scalar);
  });
}

std::vector<at::Tensor> AtenForeachSubScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachSubScalarList, _, (self, scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(scalars));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    std::vector<at::Scalar> neg_scalars;
    neg_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
      neg_scalars.push_back(-scalar);
    }
    return AtenForeachAddScalarList(self, neg_scalars);
  });
}

void AtenForeachSub_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachSub_ScalarList, _, (self, scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(EnsureNotBool(scalars));
    TT_THROW_IF_ERROR(EnsureNotBool(self));
    std::vector<at::Scalar> neg_scalars;
    neg_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
      neg_scalars.push_back(-scalar);
    }
    AtenForeachAdd_ScalarList(self, neg_scalars);
  });
}

}  // namespace torch_tpu
