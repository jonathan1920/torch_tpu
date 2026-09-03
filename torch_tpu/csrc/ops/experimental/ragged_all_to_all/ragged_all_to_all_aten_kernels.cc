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

#include "torch_tpu/csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_aten_kernels.h"

#include <string>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "c10/util/intrusive_ptr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/distributed/process_group_tpu.h"
#include "torch_tpu/csrc/distributed/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_builder.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<ProcessGroupTpu* absl_nonnull> GetProcessGroupTpu(
    std::string_view process_group_name) {
  std::string process_group_name_str(process_group_name);
  c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(process_group_name_str);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=process group resolution
      pg != nullptr, error::kInternal)
      << "failed to resolve given process group";
  c10::intrusive_ptr<c10d::Backend> backend =
      pg->getBackend(c10::DeviceType::PrivateUse1);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=backend resolution
      backend != nullptr, error::kInternal)
      << "failed to get backend for tpu device";

  auto* process_group_tpu = dynamic_cast<ProcessGroupTpu*>(backend.get());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=backend dynamic cast
      process_group_tpu != nullptr, error::kInternal)
      << "failed to cast c10d::Backend to ProcessGroupTpu";
  return process_group_tpu;
}

static absl::StatusOr<DeviceBufferRef> RaggedAllToAllCommon(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name, OpParamCacheKeys& param_keys) {
  at::ScalarType out_scalar_type = output.scalar_type();
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));

  std::string process_group_name_str(process_group_name);

  auto op_builder =
      [process_group_name_str](FixedSizeSpan<mlir::MlirOp, 6> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [operand, output, input_offsets, send_sizes, output_offsets,
           recv_sizes] = inputs;
    auto& builder = operand.getBuilder();
    TT_ASSIGN_OR_RETURN(ProcessGroupTpu * pg,
                        GetProcessGroupTpu(process_group_name_str));
    mlir::DenseIntElementsAttr replica_groups_attr =
        BuildReplicaGroupsAttr(builder, pg->GetSubgroupDeviceIds());

    return BuildRaggedAllToAllShlo(operand, output, input_offsets, send_sizes,
                                   output_offsets, recv_sizes,
                                   replica_groups_attr);
  };

  return DispatchOp<6>(
      std::move(op_builder),
      {operand, output, input_offsets, send_sizes, output_offsets, recv_sizes},
      {.out_dtype = out_dtype,
       .out_dims = CopyIntVector(output.sizes()),
       .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenRaggedAllToAll(const at::Tensor& operand,
                              const at::Tensor& output,
                              const at::Tensor& input_offsets,
                              const at::Tensor& send_sizes,
                              const at::Tensor& output_offsets,
                              const at::Tensor& recv_sizes,
                              std::string_view process_group_name) {
  TT_KERNEL(OpName::kRaggedAllToAll, param_keys,
            (operand, output, input_offsets, send_sizes, output_offsets,
             recv_sizes, process_group_name),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  RaggedAllToAllCommon(operand, output, input_offsets,
                                       send_sizes, output_offsets, recv_sizes,
                                       process_group_name, param_keys));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenRaggedAllToAllOut(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name, at::Tensor& out) {
  TT_KERNEL(OpName::kRaggedAllToAll, param_keys,
            (operand, output, input_offsets, send_sizes, output_offsets,
             recv_sizes, process_group_name, out),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  RaggedAllToAllCommon(operand, output, input_offsets,
                                       send_sizes, output_offsets, recv_sizes,
                                       process_group_name, param_keys));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

}  // namespace torch_tpu
