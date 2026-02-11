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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/mm/mm.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::Status CheckMmPreconditions(const at::Tensor& lhs,
                                  const at::Tensor& rhs) {
  TT_RET_CHECK(lhs.dim() == 2, error::kInvalidArgument)
      << "the first argument must be a 2D tensor, got a " << lhs.dim()
      << "D tensor with size [" << absl::StrJoin(lhs.sizes(), ", ") << "]";
  TT_RET_CHECK(rhs.dim() == 2, error::kInvalidArgument)
      << "the second argument must be a 2D tensor, got a " << rhs.dim()
      << "D tensor with size [" << absl::StrJoin(rhs.sizes(), ", ") << "]";
  TT_RET_CHECK(lhs.size(1) == rhs.size(0), error::kInvalidArgument)
      << "the column size of the first argument must equal the row size "
         "of the second argument, got "
      << lhs.size(1) << " and " << rhs.size(0);
  TT_ASSIGN_OR_RETURN(const auto lhs_dtype,
                      ConvertTo<mlir::ElementType>(lhs.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto rhs_dtype,
                      ConvertTo<mlir::ElementType>(rhs.scalar_type()));
  TT_RET_CHECK(lhs_dtype == rhs_dtype, error::kInvalidArgument)
      << "the dtypes of the first and second arguments must be the same, "
         "got "
      << ToDTypeName(lhs_dtype) << " and " << ToDTypeName(rhs_dtype);
  return absl::OkStatus();
}

}  // namespace


at::Tensor& AtenMmOut(const at::Tensor& lhs, const at::Tensor& rhs,
                      at::Tensor& out) {
  TT_KERNEL(OpName::kMmOut, _, (lhs, rhs, out), {
    TT_THROW_IF_ERROR(CheckMmPreconditions(lhs, rhs));
    TT_CHECK_THROW(lhs.dtype() == out.dtype(), error::kInvalidArgument)
        << "out must have the same dtype as inputs, got input type "
        << lhs.dtype() << " and output type" << out.dtype();
    int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
    mlir::ElementType dtype =
        ConvertTo<mlir::ElementType>(lhs.scalar_type()).value();
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<2>(OpName::kMmOut, BuildMmShlo, {lhs, rhs},
                      {.out_dtype = dtype, .out_dims = output_dims}));
    at::native::resize_output(out, output_dims);
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
