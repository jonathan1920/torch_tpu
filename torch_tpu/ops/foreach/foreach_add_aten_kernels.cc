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

#include <array>
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

static absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachAddListShlo(
    absl::Span<mlir::MlirOp> self, absl::Span<mlir::MlirOp> other,
    absl::Span<const mlir::MlirOp> alphas, mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());

  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_alpha = alphas[i];
    mlir::ElementType out_dtype = GetElementTypeOrDie(current_alpha);
    mlir::MlirOp current_self = CastIfNeeded(self[i], out_dtype).value();
    mlir::MlirOp current_other = CastIfNeeded(other[i], out_dtype).value();

    // Broadcast alpha (a scalar) to the same shape as other[i].
    std::array<mlir::MlirOp, 2> broadcasted_ops;
    TT_ASSIGN_OR_RETURN(broadcasted_ops,
                        ApplyBroadcastIfNeeded(current_other, current_alpha));
    current_other = broadcasted_ops[0];
    current_alpha = broadcasted_ops[1];

    mlir::MlirOp product = mlir::stablehlo::Mul(current_other, current_alpha);
    mlir::MlirOp sum = mlir::stablehlo::Add(current_self, product);

    results.push_back(sum);
  }
  return results;
}

static absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachAddListShlo(
    absl::Span<mlir::MlirOp> self, absl::Span<mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());

  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = CastIfNeeded(self[i], out_dtypes[i]).value();
    mlir::MlirOp current_other = CastIfNeeded(other[i], out_dtypes[i]).value();
    mlir::MlirOp sum = mlir::stablehlo::Add(current_self, current_other);
    results.push_back(sum);
  }
  return results;
}

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  TT_KERNEL(OpName::kForeachAddList, param_keys, (self, other, alpha), {
    // self and other are guaranteed to have the same size.
    // The error is handled by the upstream torch.
    size_t num_tensors = self.size();

    // Prepare the output specs.
    std::vector<mlir::ElementType> out_dtypes;
    out_dtypes.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      TT_CHECK_THROW(!(c10::isIntegralType(self[i].scalar_type(), true) &&
                       c10::isIntegralType(other[i].scalar_type(), true) &&
                       !c10::isIntegralType(alpha.type(), true)),
                     error::kInvalidArgument)
          << "expected alpha to be integral for integral input tensors, got "
          << ToString(alpha.type());
      TT_CHECK_THROW(!alpha.isBoolean() ||
                         (self[i].scalar_type() == at::ScalarType::Bool &&
                          other[i].scalar_type() == at::ScalarType::Bool),
                     error::kInvalidArgument)
          << "expected input tensor dtypes to be bool when alpha dtype is "
             "bool, got "
          << ToString(self[i].scalar_type()) << " and "
          << ToString(other[i].scalar_type());
      at::ScalarType output_scalar_type =
          c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
      TT_ASSIGN_OR_THROW(mlir::ElementType output_element_type,
                         ConvertTo<mlir::ElementType>(output_scalar_type));
      out_dtypes.push_back(output_element_type);
    }
    std::vector<absl::Span<const int64_t>> out_dims_list;
    out_dims_list.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      out_dims_list.push_back(self[i].sizes());
    }

    DispatchOpOptions<kDynamicSize> options = {
        .out_dtypes = absl::MakeConstSpan(out_dtypes),
        .out_dims_list = out_dims_list,
        .op_param_cache_keys = std::move(param_keys)};

    // The op builder.
    std::vector<at::Tensor> inputs;
    inputs.reserve(self.size() + other.size());
    inputs.insert(inputs.end(), self.begin(), self.end());
    inputs.insert(inputs.end(), other.begin(), other.end());
    auto op_builder = [alpha, num_tensors, out_dtypes](
                          absl::Span<mlir::MlirOp> inputs,
                          mlir::MlirBuilder& builder)
        -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
      absl::Span<mlir::MlirOp> self_op = inputs.subspan(0, num_tensors);
      absl::Span<mlir::MlirOp> other_op =
          inputs.subspan(num_tensors, num_tensors * 2);

      // If alpha is 1.0, do a simple addition without multiplying by alpha.
      if ((alpha.isIntegral(true) && alpha.to<int64_t>() == 1) ||
          (alpha.isFloatingPoint() && alpha.to<double>() == 1.0)) {
        return BuildForeachAddListShlo(self_op, other_op, out_dtypes, builder);
      }

      std::vector<mlir::MlirOp> alpha_ops;
      alpha_ops.reserve(num_tensors);
      for (int i = 0; i < num_tensors; ++i) {
        TT_ASSIGN_OR_RETURN(mlir::MlirOp current_alpha_op,
                            MakeConstant(builder, alpha, out_dtypes[i]));
        alpha_ops.push_back(current_alpha_op);
      }
      return BuildForeachAddListShlo(self_op, other_op, alpha_ops, builder);
    };

    // Dispatch the op and prepare results.
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       (DispatchOp<kDynamicSize, kDynamicSize>(
                           OpName::kForeachAddList, std::move(op_builder),
                           inputs, std::move(options))));
    std::vector<at::Tensor> result;
    result.reserve(result_buffers.size());
    for (auto& result_buffer : result_buffers) {
      result.push_back(MakeTensor(std::move(result_buffer)));
    }
    return result;
  });
}

}  // namespace torch_tpu
