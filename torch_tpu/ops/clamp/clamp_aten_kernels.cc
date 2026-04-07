// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/clamp/clamp_aten_kernels.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/promote_types.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/clamp/clamp.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// Returns a comma-separated, string representation of the dtypes of the input
// arguments.
//
// For `self`, only its dtype is printed. For the optional parameters, their
// dtype is prefixed with their parameter name.
//
// In this context, `T` should be either of:
//
//   - at::Scalar
//   - at::Tensor
template <typename T>
std::string GetInputsTypeStr(const at::Tensor& self,
                             const c10::optional<T>& min,
                             const c10::optional<T>& max) {
  return absl::StrCat(
      ToString(self.scalar_type()),
      min ? absl::StrCat(", min: ", ToString(GetScalarType(*min))) : "",
      max ? absl::StrCat(", max: ", ToString(GetScalarType(*max))) : "");
}

// Checks whether it's possible to cast `computation_scalar_type` into
// `output_scalar_type`.
//
// If the check turns out `false`, the first 3 parameters are used for adding
// useful information into the error message.
//
// For more information on `at::canCast()`, see:
// https://docs.pytorch.org/docs/stable/tensor_attributes.html
//
// In this context, `T` should be either of:
//
//   - at::Scalar
//   - at::Tensor
template <typename T>
absl::Status CheckCanCastComputationToOutput(
    const at::Tensor& self, const c10::optional<T>& min,
    const c10::optional<T>& max, at::ScalarType computation_scalar_type,
    at::ScalarType output_scalar_type) {
  TT_RET_CHECK(at::canCast(computation_scalar_type, output_scalar_type),
               error::kInvalidArgument)
      << "unable to cast " << ToString(computation_scalar_type)
      << ", the promotion of the dtypes of the inputs ("
      << GetInputsTypeStr(self, min, max) << "), to the output dtype "
      << ToString(output_scalar_type);
  return absl::OkStatus();
}

// Returns the computation `at::ScalarType`, given the inputs.
//
// The computation `at::ScalarType` consists on the promotion of the
// `at::ScalarType` of all the inputs.
template <typename T>
at::ScalarType GetComputationType(const at::Tensor& self,
                                  const c10::optional<T>& min,
                                  const c10::optional<T>& max) {
  at::ScalarType computation_scalar_type = self.scalar_type();

  if (min) {
    computation_scalar_type =
        at::promote_types(computation_scalar_type, GetScalarType(*min));
  }

  if (max) {
    computation_scalar_type =
        at::promote_types(computation_scalar_type, GetScalarType(*max));
  }

  return computation_scalar_type;
}

absl::StatusOr<Dimensions> ComputeOutputShape(const at::Tensor& self,
                                              std::optional<at::Tensor> min,
                                              std::optional<at::Tensor> max) {
  if (min) {
    return InferSize(self.sizes(), min->sizes());
  }
  if (max) {
    return InferSize(self.sizes(), max->sizes());
  }
  return CopyIntVector(self.sizes());
}

absl::StatusOr<DeviceBufferRef> AtenClampHelper(
    const at::Tensor& self, const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max, at::ScalarType output_scalar_type,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(output_scalar_type));
  auto output_dims = self.sizes();

  at::ScalarType computation_scalar_type = GetComputationType(self, min, max);
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));
  TT_RETURN_IF_ERROR(CheckCanCastComputationToOutput(
      self, min, max, computation_scalar_type, output_scalar_type));

  if (min && max) {
    auto op_builder = [min = *min, max = *max, computation_dtype](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      // TODO: Deprecate `MakeConstant`, add `MakeConstantLike` with at::Scalar
      // support
      auto& builder = input.getBuilder();
      TT_ASSIGN_OR_RETURN(mlir::MlirOp min_op,
                          MakeConstant(builder, min, computation_dtype));
      TT_ASSIGN_OR_RETURN(mlir::MlirOp max_op,
                          MakeConstant(builder, max, computation_dtype));
      return BuildClampShlo(input, min_op, max_op);
    };

    return DispatchOp<1>(std::move(op_builder), self,
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else if (min) {
    auto op_builder = [min = *min, computation_dtype](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      auto& builder = input.getBuilder();
      TT_ASSIGN_OR_RETURN(mlir::MlirOp min_op,
                          MakeConstant(builder, min, computation_dtype));
      return BuildClampShlo(input, min_op, std::nullopt);
    };

    return DispatchOp<1>(std::move(op_builder), self,
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else if (max) {
    auto op_builder = [max = *max, computation_dtype](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      auto& builder = input.getBuilder();
      TT_ASSIGN_OR_RETURN(mlir::MlirOp max_op,
                          MakeConstant(builder, max, computation_dtype));
      return BuildClampShlo(input, std::nullopt, max_op);
    };

    return DispatchOp<1>(std::move(op_builder), self,
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else {
    auto op_builder = [](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildClampShlo(input, std::nullopt, std::nullopt);
    };

    return DispatchOp<1>(std::move(op_builder), self,
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});
  }
}

absl::StatusOr<DeviceBufferRef> AtenClampTensorHelper(
    const at::Tensor& self, const c10::optional<at::Tensor>& min_,
    const c10::optional<at::Tensor>& max_, at::ScalarType output_scalar_type,
    OpParamCacheKeys param_keys) {
  auto min = SanitizeOptionalTensor(min_);
  auto max = SanitizeOptionalTensor(max_);

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(output_scalar_type));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ComputeOutputShape(self, min, max));

  at::ScalarType computation_scalar_type = GetComputationType(self, min, max);
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));
  TT_RETURN_IF_ERROR(CheckCanCastComputationToOutput(
      self, min, max, computation_scalar_type, output_scalar_type));

  if (min && max) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      auto& [self, min, max] = inputs;
      return BuildClampShlo(self, min, max);
    };

    return DispatchOp<3>(std::move(op_builder), {self, *min, *max},
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else if (min) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self, min] = inputs;
      return BuildClampShlo(self, min, std::nullopt);
    };
    return DispatchOp<2>(std::move(op_builder), {self, *min},
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else if (max) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [input, max] = inputs;
      return BuildClampShlo(input, std::nullopt, max);
    };
    return DispatchOp<2>(std::move(op_builder), {self, *max},
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});

  } else {
    auto op_builder = [](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildClampShlo(input, std::nullopt, std::nullopt);
    };

    return DispatchOp<1>(std::move(op_builder), self,
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .computation_dtype = computation_dtype,
                          .op_param_cache_keys = std::move(param_keys)});
  }
}

}  // namespace

at::Tensor& AtenClampOut(const at::Tensor& self,
                         const c10::optional<at::Scalar>& min,
                         const c10::optional<at::Scalar>& max,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kClampOut, param_keys, (self, min, max, out), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenClampHelper(self, min, max, out.scalar_type(),
                                       std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenClampMinOut(const at::Tensor& self, const at::Scalar& min,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kClampMinOut, param_keys, (self, min, out), {
    TT_ASSIGN_OR_THROW(auto result_buf, AtenClampHelper(self, min, std::nullopt,
                                                        out.scalar_type(),
                                                        std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenClampMaxOut(const at::Tensor& self, const at::Scalar& max,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kClampMaxOut, param_keys, (self, max, out), {
    TT_ASSIGN_OR_THROW(auto result_buf, AtenClampHelper(self, std::nullopt, max,
                                                        out.scalar_type(),
                                                        std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenClampTensorOut(const at::Tensor& self,
                               const c10::optional<at::Tensor>& min,
                               const c10::optional<at::Tensor>& max,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kClampTensorOut, param_keys, (self, min, max, out), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenClampTensorHelper(self, min, max, out.scalar_type(),
                                             std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenClampMinTensorOut(const at::Tensor& self, const at::Tensor& min,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kClampMinTensorOut, param_keys, (self, min, out), {
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenClampTensorHelper(self, min, std::nullopt, out.scalar_type(),
                              std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenClampMaxTensorOut(const at::Tensor& self, const at::Tensor& max,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kClampMaxTensorOut, param_keys, (self, max, out), {
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenClampTensorHelper(self, std::nullopt, max, out.scalar_type(),
                              std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
