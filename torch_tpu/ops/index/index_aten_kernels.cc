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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/index/index.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<Dimensions> GetOutputDims(
    const at::Tensor& self, Indices indexed_dims,
    absl::Span<const at::Tensor> indices_list) {
  TT_RET_CHECK(indices_list.size() == indexed_dims.size(),
               error::kInvalidArgument)
      << "length of indexing tensors list must be the same as the number of "
         "indexed dimensions";
  TT_RET_CHECK(!indices_list.empty(), error::kInvalidArgument)
      << "at least one index tensor must be defined";
  // TODO(unda): Add support for bool index tensors.
  for (const auto& t : indices_list) {
    TT_RET_CHECK(t.scalar_type() != at::ScalarType::Bool,
                 error::kInvalidArgument)
        << "bool index tensors are not supported";
  }
  // The shape of the output tensor is the combination of the shape of the
  // (broadcasted) index tensors and the shape of the unindexed dimensions of
  // the input tensor. Unless the indexed dimensions are consecutive, they will
  // be mapped to the first dimension of the output. See
  // https://numpy.org/devdocs/user/basics.indexing.html#combining-advanced-and-basic-indexing.
  Dimensions broadcasted_index_shape;
  Dimensions unindexed_dims_shape;
  bool indexed_dimensions_consecutive = true;
  for (int i = 0, j = 0; i < self.dim(); ++i) {
    if (j < indexed_dims.size() && i == indexed_dims[j]) {
      if (j > 0 && indexed_dimensions_consecutive &&
          indexed_dims[j - 1] != i - 1) {
        indexed_dimensions_consecutive = false;
      }
      TT_ASSIGN_OR_RETURN(
          broadcasted_index_shape,
          InferSize(broadcasted_index_shape, indices_list[j].sizes()));
      j++;
    } else {
      unindexed_dims_shape.push_back(self.size(i));
    }
  }
  Dimensions output_dims;
  output_dims.reserve(broadcasted_index_shape.size() +
                      unindexed_dims_shape.size());
  size_t insertion_index = indexed_dimensions_consecutive ? indexed_dims[0] : 0;
  output_dims.insert(output_dims.end(), unindexed_dims_shape.begin(),
                     unindexed_dims_shape.begin() + insertion_index);
  output_dims.insert(output_dims.end(), broadcasted_index_shape.begin(),
                     broadcasted_index_shape.end());
  output_dims.insert(output_dims.end(),
                     unindexed_dims_shape.begin() + insertion_index,
                     unindexed_dims_shape.end());
  return output_dims;
}

}  // namespace

at::Tensor& AtenIndexTensorOut(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_list_opt,
    at::Tensor& out) {
  TT_KERNEL(OpName::kIndexTensorOut, _, (self, indices_list_opt, out), {
    TT_CHECK_THROW(indices_list_opt.size() <= self.dim(),
                   error::kInvalidArgument)
        << "number of indexing tensors must be at most the number of "
           "dimensions";

    Indices indexed_dims;
    std::vector<at::Tensor> indices_list;
    for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
      auto maybe_tensor = indices_list_opt[i];
      if (maybe_tensor.has_value() && maybe_tensor.value().defined()) {
        indexed_dims.push_back(i);
        indices_list.push_back(maybe_tensor.value());
      }
    }
    TT_ASSIGN_OR_THROW(Dimensions output_dims,
                       GetOutputDims(self, indexed_dims, indices_list));
    // The indices_list_opt gets ignored in the cache key, but we still
    // want to record which dimensions are indexed.
    TT_ASSIGN_OR_THROW(auto param_keys,
                       TT_MAKE_OP_PARAM_CACHE_KEYS(indexed_dims));

    ABSL_VLOG(2) << "[AtenIndexTensorOut] self: " << ToString(self);
    for (const auto& t : indices_list) {
      ABSL_VLOG(2) << "[AtenIndexTensorOut] indices_list: " << ToString(t);
    }
    ABSL_VLOG(2) << "[AtenIndexTensorOut] indexed_dims: "
                 << absl::StrJoin(indexed_dims, ",");
    ABSL_VLOG(2) << "[AtenIndexTensorOut] output_dims: "
                 << absl::StrJoin(output_dims, ",");
    if (self.dim() == 0) {
      TT_CHECK_THROW(indexed_dims.size() == 1 && indexed_dims[0] == 0,
                     error::kInvalidArgument)
          << "dims must be [0] for a scalar tensor.";
      output_dims = {};
    }

    std::vector<at::Tensor> all_tensors;
    all_tensors.reserve(indices_list.size() + 1);
    all_tensors.push_back(self);
    for (int i = 0; i < indices_list.size(); ++i) {
      all_tensors.push_back(indices_list[i]);
    }

    TT_ASSIGN_OR_THROW(const auto computation_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    auto index_op_builder = [indexed_dims = std::move(indexed_dims)](
                                absl::Span<mlir::MlirOp> inputs,
                                mlir::MlirBuilder& builder) {
      return BuildIndexShlo(inputs, indexed_dims);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<kDynamicSize>(
            OpName::kIndexTensorOut, std::move(index_op_builder), all_tensors,
            {.out_dtype = computation_dtype,
             .out_dims = absl::Span<const int64_t>(output_dims),
             .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
