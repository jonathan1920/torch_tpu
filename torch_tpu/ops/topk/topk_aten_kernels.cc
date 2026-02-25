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

#include "torch_tpu/ops/topk/topk_aten_kernels.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ATen/core/TensorBase.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/topk/topk.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
std::tuple<at::Tensor&, at::Tensor&> AtenTopKValues(
    const at::Tensor& self, const int64_t k, const int64_t dim,
    const bool largest, const bool sorted, at::Tensor& values,
    at::Tensor& indices) {
  TT_KERNEL(OpName::kTopkValues, param_keys, (self, k, dim, largest, sorted), {
    if (self.dim() == 0) {
      values = self.clone();
      indices = torch::tensor(0);
      return {values, indices};
    }
    const int64_t topk_dim = dim >= 0 ? dim : self.dim() + dim;
    TT_CHECK_THROW(topk_dim >= 0, error::kInvalidArgument) << absl::StrCat(
        "invalid argument dim: ", dim, " Input tensor rank: ", self.dim());
    // TODO(b/435537869): Support sorted = false.
    TT_CHECK_THROW(sorted, error::kInvalidArgument)
        << absl::StrCat("sorted must be true. Got sorted = ", sorted);

    const auto self_sizes = self.sizes();
    TT_ASSIGN_OR_THROW(const auto elem_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    Dimensions output_dims = CopyIntVector(self_sizes);
    output_dims[topk_dim] = k;
    auto op_builder =
        [topk_dim, k,
         largest](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      TT_ASSIGN_OR_RETURN(
          auto topk_outputs,
          BuildTopKShlo(input, k, topk_dim,
                        largest ? TopKMode::kLargest : TopKMode::kSmallest,
                        TopKStableMode::kUnstable));
      return {{topk_outputs.values, topk_outputs.indices}};
    };
    TT_ASSIGN_OR_THROW(
        (auto [values_buf, indices_buf]),
        (DispatchOp<1, 2>(OpName::kTopkValues, std::move(op_builder), self,
                          {.out_dtypes = {elem_type, mlir::ElementType::I64},
                           .out_dims_list = {output_dims, output_dims},
                           .op_param_cache_keys = std::move(param_keys)})));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(values_buf), values));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(indices_buf), indices));

    return {values, indices};
  });
}

}  // namespace torch_tpu
