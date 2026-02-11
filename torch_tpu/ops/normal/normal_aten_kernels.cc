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

#include "torch_tpu/ops/normal/normal_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/scalar_tensor.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildNormalShlo(mlir::MlirBuilder& builder,
                                             double mean, double std,
                                             llvm::ArrayRef<int64_t> sizes,
                                             mlir::ElementType mlir_type) {
  mlir::MlirOp a = MakeScalarConstant(builder, mean, mlir_type);
  mlir::MlirOp b = MakeScalarConstant(builder, std, mlir_type);
  mlir::MlirOp shape = mlir::stablehlo::Constant(
      builder,
      makeConstant(sizes, makeTensorType(builder.getContext(),
                                         {static_cast<int64_t>(sizes.size())},
                                         mlir::ElementType::I64)));
  return mlir::stablehlo::Rng(a, b, shape,
                              mlir::stablehlo::RngDistribution::NORMAL);
}

absl::StatusOr<MlirNullaryOpBuilder> GetNormalFunctional(
    double mean, double std, llvm::ArrayRef<int64_t> sizes,
    at::ScalarType aten_dtype) {
  TT_ASSIGN_OR_RETURN(const auto mlir_type,
                      ConvertTo<mlir::ElementType>(aten_dtype));
  return [mean, std, sizes = CopyIntVector(sizes), mlir_type](
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    return BuildNormalShlo(builder, mean, std, sizes, mlir_type);
  };
}

absl::StatusOr<mlir::MlirOp> BuildBinaryNormalShlo(mlir::MlirBuilder& builder,
                                                   mlir::MlirOp mean_op,
                                                   mlir::MlirOp std_op) {
  TT_ASSIGN_OR_RETURN(const Dimensions bcast_dims,
                      InferSize(GetTensorTypeOrDie(mean_op).getShape(),
                                GetTensorTypeOrDie(std_op).getShape()));
  llvm::SmallVector<int64_t> bcast_dims_vec(bcast_dims.begin(),
                                            bcast_dims.end());
  auto shape_op = mlir::stablehlo::Constant(
      builder,
      makeConstant(llvm::ArrayRef<int64_t>(bcast_dims_vec),
                   makeTensorType(builder.getContext(),
                                  {static_cast<int64_t>(bcast_dims.size())},
                                  mlir::ElementType::I64)));
  return mlir::stablehlo::Rng(mean_op, std_op, shape_op,
                              mlir::stablehlo::RngDistribution::NORMAL);
}

absl::Status CheckNormalPreconditions(const at::Tensor& self,
                                      std::optional<at::Generator> generator) {
  // TODO(b/437527594): Support RNG on-host vs RNG on-device.
  TT_RET_CHECK(!generator.has_value(), error::kUnimplemented)
      << "normal: generator is not yet supported.";
  TT_RET_CHECK(!self.is_complex(), error::kUnimplemented)
      << "normal: input tensor must not be complex type. XLA doesn't "
      << "support complex types for this op.";
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "normal: input tensor must be floating point type but got "
      << ToDTypeName(dtype);
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenNormal_(at::Tensor& self, double mean, double std,
                        std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormal_, param_keys, (self, mean, std, generator), {
    TT_THROW_IF_ERROR(CheckNormalPreconditions(self, generator));
    TT_ASSIGN_OR_THROW(
        auto builder,
        GetNormalFunctional(mean, std, self.sizes(), self.scalar_type()));
    TT_THROW_IF_ERROR(ApplyNullaryOpOut(
        self, OpName::kNormal_, std::move(builder), self.scalar_type(),
        self.sizes(), std::move(param_keys), OpSplitMode::kSplitAfter));
    return self;
  });
}

at::Tensor AtenNormalFloatTensor(double mean, const at::Tensor& std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalFloatTensor, _, (mean, std, generator), {
    TT_THROW_IF_ERROR(CheckNormalPreconditions(std, generator));
    at::Tensor mean_tensor = at::scalar_tensor(mean, std.options());
    return AtenNormalTensorTensor(mean_tensor, std, generator);
  });
}

at::Tensor& AtenNormalFloatTensorOut(double mean, const at::Tensor& std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(OpName::kNormalFloatTensorOut, _, (mean, std, generator, out), {
    TT_THROW_IF_ERROR(CheckNormalPreconditions(std, generator));
    TT_THROW_IF_ERROR(CheckNormalPreconditions(out, std::nullopt));
    out = AtenNormalFloatTensor(mean, std, generator);
    return out;
  });
}

at::Tensor AtenNormalTensorFloat(const at::Tensor& mean, double std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorFloat, _, (mean, std, generator), {
    TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, generator));
    at::Tensor std_tensor = at::scalar_tensor(std, mean.options());
    return AtenNormalTensorTensor(mean, std_tensor, generator);
  });
}

at::Tensor& AtenNormalTensorFloatOut(const at::Tensor& mean, double std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(OpName::kNormalTensorFloatOut, _, (mean, std, generator, out), {
    out = AtenNormalTensorFloat(mean, std, generator);
    return out;
  });
}

at::Tensor AtenNormalTensorTensor(const at::Tensor& mean, const at::Tensor& std,
                                  std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorTensor, _, (mean, std, generator), {
    TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, generator));
    TT_THROW_IF_ERROR(CheckNormalPreconditions(std, generator));
    TT_ASSIGN_OR_THROW(
        auto result,
        BinaryOp(OpName::kNormalTensorTensor, mean, std,
                 [](mlir::MlirOp mean_op,
                    mlir::MlirOp std_op) -> absl::StatusOr<mlir::MlirOp> {
                   return BuildBinaryNormalShlo(mean_op.getBuilder(), mean_op,
                                                std_op);
                 },
                 {.split_mode = OpSplitMode::kSplitAfter}));
    return result;
  });
}

at::Tensor& AtenNormalTensorTensorOut(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator,
                                      at::Tensor& out) {
  TT_KERNEL(OpName::kNormalTensorTensorOut, _, (mean, std, generator, out), {
    out = AtenNormalTensorTensor(mean, std, generator);
    return out;
  });
}

}  // namespace torch_tpu
