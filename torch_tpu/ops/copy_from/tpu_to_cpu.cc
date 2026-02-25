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

#include "torch_tpu/ops/copy_from/tpu_to_cpu.h"

#include <cstdint>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

// Translates a low-level XLA OOM error to an error that is easier to understand
// by a pytorch user. If the status is not an XLA OOM error, returns it as is.
//
// `expression` is the pytorch tensor expression that caused the XLA OOM error.
// `elem_type` is the element type of the tensor that caused the XLA OOM error.
// `dims` is the dimensions of the tensor that caused the XLA OOM error.
static absl::Status TranslateXlaTensorOomError(const absl::Status& status,
                                               std::string_view expression,
                                               at::ScalarType elem_type,
                                               absl::Span<const int64_t> dims) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=All PyTorch dtypes are supported
                        // by mlir::ElementType.
      const auto dtype, ConvertTo<mlir::ElementType>(elem_type));
  TT_RET_CHECK(!IsXlaOomError(status), error::kResourceExhausted)
      << "in " << expression << ", the tensor shape " << ToDTypeName(dtype)
      << "[" << absl::StrJoin(dims, ", ") << "] is too large to fit in memory";
  return status;
}

absl::Status CopyTpuToCpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking) {
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> CPU copy path for "
               << ToString(src, "src");

  // Materialize if deferred, no-op if already materialized.
  // If src is a view, this will materialize both the base buffer and the
  // ephemeral view buffer.
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef materialized_src_buf,
                      GetMaterialized(src));

  if (materialized_src_buf.state() == DeviceBufferRefState::kZeroSize) {
    // Shortcut: no data to move, just set dest's storage to an empty
    // c10::StorageImpl.
    at::Tensor empty_dest = at::empty(dest.sizes(), dest.options());
    dest.unsafeGetTensorImpl()->set_storage_keep_dtype(empty_dest.storage());
    return absl::OkStatus();
  }
  ABSL_CHECK_EQ(materialized_src_buf.state(),  // CRASH_OK
                DeviceBufferRefState::kMaterialized)
      << "expected materialized buffer after GetMaterialized";

  TT_ASSIGN_OR_RETURN(at::Tensor cpu_tensor_receiver,
                      TpuMemcpyDtoH(materialized_src_buf),
                      TranslateXlaTensorOomError(
                          _, "_copy_from", dest.scalar_type(), src.sizes()));

  // Redispatch using copy_() on the CPU to do any type or layout conversions
  // needed.
  dest.copy_(cpu_tensor_receiver, non_blocking);
  return absl::OkStatus();
}

}  // namespace torch_tpu
