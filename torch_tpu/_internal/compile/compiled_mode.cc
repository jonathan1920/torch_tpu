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

#include "torch_tpu/_internal/compile/compiled_mode.h"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/TensorImpl.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<at::Tensor> MakePlaceholder(absl::Span<const int64_t> sizes,
                                           at::ScalarType dtype,
                                           bool requires_grad) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType tensor_element_type,
                      ConvertTo<mlir::ElementType>(dtype));
  TT_ASSIGN_OR_RETURN(DeviceBufferRef placeholder,
                      DeviceBufferList::MakePlaceholder(CopyIntVector(sizes),
                                                        tensor_element_type));
  auto new_tensor = MakeTensor(std::move(placeholder));
  new_tensor.set_requires_grad(requires_grad);
  return new_tensor;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractMlirFromGraph(
    mlir::MLIRContext& mlir_context, const std::vector<at::Tensor>& arg_tensors,
    const std::vector<at::Tensor>& result_tensors) {
  // Use artificial python context for compiled mode to that modules have better
  // names when dumped. Currently this will always point to `_export_to_fx`, and
  // can be improved in the future, but it is useful for distinguishing torch
  // compiled graphs from eager graphs.
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());

  TT_RET_CHECK(!result_tensors.empty(), error::kInvalidArgument)
      << "no result tensors provided";

  std::vector<DeviceBufferRef> argument_refs;
  argument_refs.reserve(arg_tensors.size());
  for (const at::Tensor& tensor : arg_tensors) {
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buffer_ref, GetBufferFromAtTensor(tensor),
        _.SetPrepend() << "failed to get buffer from argument tensor: "
                       << ToString(tensor));
    TT_RET_CHECK(buffer_ref.state() != DeviceBufferRefState::kDeferred,
                 error::kInternal)
        << "argument tensor has deferred ops: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] arg_tensor: "
                 << buffer_ref.DebugString();
    argument_refs.push_back(std::move(buffer_ref));
  }

  std::vector<DeviceBufferRef> result_refs;
  result_refs.reserve(result_tensors.size());
  for (const at::Tensor& tensor : result_tensors) {
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buffer_ref, GetBufferFromAtTensor(tensor),
        _.SetPrepend() << "failed to get buffer from result tensor: "
                       << ToString(tensor));
    TT_RET_CHECK(buffer_ref.state() != DeviceBufferRefState::kMaterialized,
                 error::kInternal)
        << "result tensor is already materialized: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] result_tensor: "
                 << buffer_ref.DebugString();
    result_refs.push_back(std::move(buffer_ref));
  }

  TT_ASSIGN_OR_RETURN(Traversal traversal,
                      Traversal::Create(std::move(result_refs)),
                      _.SetPrepend() << "failed to traverse graph: ");
  TT_RETURN_IF_ERROR(
      traversal.ValidateAndReorderInputs(std::move(argument_refs)))
          .SetPrepend()
      << "failed to validate and reorder inputs: ";
  TT_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                      traversal.BuildMlirModule(mlir_context));

  return mlir_module;
}

absl::StatusOr<SharedLoadedExecutable> CompileMlirExecutable(
    const std::string_view mlir_module_bytecode,
    const CompilationMode compilation_mode) {
  MlirComputationBuilder computation_builder =
      [&](mlir::MLIRContext& mlir_context) {
        return mlir::stablehlo::deserializePortableArtifact(
            {mlir_module_bytecode.data(), mlir_module_bytecode.size()},
            &mlir_context);
      };

  TT_ASSIGN_OR_RETURN(UniqueCompileOptions compile_options,
                      MakeCompilerOptions(compilation_mode));
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized";
  TT_ASSIGN_OR_RETURN(
      LoadedExecutableBuilder executable_builder,
      MlirComputationBuilderToExecutableBuilder(computation_builder));
  return Compile(*client, std::move(executable_builder),
                 std::move(compile_options));
}

// This is bound to Python function tpu_torch_compile.execute(), and thus
// allowed to throw exceptions.
std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutable& executable,
    absl::Span<const at::Tensor> argument_tensors,
    absl::Span<const Shape> output_shapes) {
  // Get the materialized buffers for the argument tensors.
  TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> argument_buffer_refs,
                     GetMaterialized(argument_tensors));

  TT_ASSIGN_OR_THROW(
      std::vector<DeviceBufferRef> result_buffer_refs,
      EnqueueExecutable(executable, std::move(argument_buffer_refs),
                        output_shapes));

  auto num_outputs = result_buffer_refs.size();
  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(num_outputs);
  for (auto& result_buffer : result_buffer_refs) {
    output_tensors.push_back(MakeTensor(std::move(result_buffer)));
  }

  return output_tensors;
}

}  // namespace torch_tpu
