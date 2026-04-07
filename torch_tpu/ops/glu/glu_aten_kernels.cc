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
#include "torch_tpu/ops/glu/glu_aten_kernels.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildGluShlo(int64_t dim, mlir::MlirOp input) {
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto shape = input_type.getShape();
  const int64_t rank = shape.size();

  Indices start_indices(rank, 0);
  Indices limit_indices(shape.begin(), shape.end());
  Strides strides(rank, 1);

  const int64_t split_index = shape[dim] / 2;

  // First half
  limit_indices[dim] = split_index;
  mlir::MlirOp first_half =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  // Second half
  start_indices[dim] = split_index;
  limit_indices[dim] = shape[dim];
  mlir::MlirOp second_half =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  mlir::MlirOp neg = mlir::stablehlo::Neg(second_half);
  mlir::MlirOp exp = mlir::stablehlo::Exp(neg);

  mlir::MlirOp one = MakeConstantLike(second_half, 1.0);
  mlir::MlirOp add = mlir::stablehlo::Add(one, exp);
  mlir::MlirOp div = mlir::stablehlo::Div(one, add);

  mlir::MlirOp result = mlir::stablehlo::Mul(first_half, div);
  return result;
}

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenGluOut(const at::Tensor& self, int64_t dim, at::Tensor& out) {
  TT_KERNEL(OpName::kGluOut, params_key, (self, dim, out), {
    TT_ASSIGN_OR_THROW(dim, SafeWrapDim(dim, self.sizes().size()));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(self, /* name= */ "self"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(out, /* name= */ "out"));

    const auto& shape = self.sizes();
    TT_CHECK_THROW(!shape.empty(), error::kInvalidArgument)
        << "expected input tensor to have at least 1 dimension, got 0 "
           "dimensions";
    TT_CHECK_THROW(self.sizes()[dim] % 2 == 0, error::kInvalidArgument)
        << "expected the size of dimension " << dim << " to be even, got "
        << shape[dim];

    auto out_dims = CopyIntVector(shape);
    out_dims[dim] = out_dims[dim] / 2;

    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out, absl::bind_front(&BuildGluShlo, dim),
        {.op_param_cache_keys = std::move(params_key), .out_dims = out_dims}));
    return out;
  });
}

}  // namespace torch_tpu
