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

#include "torch_tpu/ops/mm/mm_aten_kernels.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/mm/mm.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

absl::Status CheckIsMatrix(const at::Tensor& tensor,
                           const std::string_view arg_name) {
  TT_RET_CHECK(tensor.dim() == 2, error::kInvalidArgument)
      << "expected the " << arg_name
      << " argument to be a 2D tensor (matrix), got " << tensor.dim()
      << "D of shape " << ToString(tensor.sizes());
  return absl::OkStatus();
}

absl::Status CheckMmOutInputs(const at::Tensor& lhs, const at::Tensor& rhs,
                              at::Tensor& out) {
  // DType checks.
  TT_RET_CHECK(lhs.scalar_type() == rhs.scalar_type(), error::kInvalidArgument)
      << "expected the two arguments to have the same dtype, got "
      << ToString(lhs.scalar_type()) << " vs " << ToString(rhs.scalar_type());
  TT_RET_CHECK(lhs.scalar_type() == out.scalar_type(), error::kInvalidArgument)
      << "expected the inputs and the output to have the same dtype, got "
      << ToString(lhs.scalar_type()) << " vs " << ToString(out.scalar_type());

  // Dimension checks.
  TT_RETURN_IF_ERROR(CheckIsMatrix(lhs, /* arg_name= */ "first"));
  TT_RETURN_IF_ERROR(CheckIsMatrix(rhs, /* arg_name= */ "second"));
  TT_RET_CHECK(lhs.size(1) == rhs.size(0), error::kInvalidArgument)
      << "expected the column size of the first matrix to match the row size "
         "of the second matrix, got shape "
      << ToString(lhs.sizes()) << " vs " << ToString(rhs.sizes()) << " where "
      << lhs.size(1) << " != " << rhs.size(0);

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> Mm(const at::Tensor& lhs, const at::Tensor& rhs,
                                   at::Tensor& out,
                                   OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckMmOutInputs(lhs, rhs, out));
  int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
  mlir::ElementType dtype =
      ConvertTo<mlir::ElementType>(lhs.scalar_type()).value();

  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(param_keys,
                      *OpParamCacheKeys::Builder(std::move(param_keys))
                           .SetParam("precision", current_precision));
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    return BuildMmShlo(inputs, current_precision);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<2>(OpName::kMmOut, std::move(op_builder), {lhs, rhs},
                    {.out_dtype = dtype,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return result_buf;
}

}  // namespace

at::Tensor& AtenMmOut(const at::Tensor& lhs, const at::Tensor& rhs,
                      at::Tensor& out) {
  TT_KERNEL(OpName::kMmOut, param_keys, (lhs, rhs, out), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       Mm(lhs, rhs, out, std::move(param_keys)));
    int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
    at::native::resize_output(out, output_dims);
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
