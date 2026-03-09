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

#include "torch_tpu/common/dynamism_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

mlir::MlirOp PadInputOp(mlir::MlirBuilder& builder, mlir::MlirOp input_op,
                        int64_t dimension, int64_t upper_bound) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  auto padding_value =
      MakeScalarConstant(builder, 0, input_type.getElementType());
  Dimensions edge_padding_low(input_type.getRank(), 0);
  Dimensions edge_padding_high(input_type.getRank(), 0);
  edge_padding_high[dimension] = upper_bound - input_type.getDimSize(dimension);
  if (edge_padding_high[dimension] == 0) {
    return input_op;
  }
  auto pad_op = mlir::stablehlo::Pad(input_op, padding_value, edge_padding_low,
                                     edge_padding_high, edge_padding_low);
  return pad_op;
}

absl::StatusOr<mlir::MlirOp> ConvertToBounded(mlir::MlirBuilder& builder,
                                              mlir::MlirOp input_op,
                                              const DeviceBufferRef& input) {
  if (input.dynamic_dimensions().empty()) {
    return input_op;
  }
  ABSL_CHECK(  // CRASH_OK=only one dynamic dimension per tensor is supported.
      input.dynamic_dimensions().size() == 1)
      << "only one dynamic dimension per tensor is supported, but got "
      << input.dynamic_dimensions().size();
  const int64_t dimension_index = input.dynamic_dimensions()[0].dimension;
  const int64_t upper_bound = input.dynamic_dimensions()[0].upper_bound;
  TT_ASSIGN_OR_RETURN(auto upper_bound_op,
                      MakeConstant(builder, input.dimensions()[dimension_index],
                                   mlir::ElementType::I32));
  ABSL_VLOG(3) << "[ConvertToBounded] upper bound op: "
               << mlir::debugString(upper_bound_op.getType());
  mlir::MlirOp padded_op =
      PadInputOp(builder, input_op, dimension_index, upper_bound);
  ABSL_VLOG(3) << "[ConvertToBounded] padded op: "
               << mlir::debugString(padded_op.getType());
  mlir::MlirOp set_dimension_size_op = mlir::stablehlo::SetDimensionSize(
      padded_op, upper_bound_op, dimension_index);
  ABSL_VLOG(3) << "[ConvertToBounded] set dimension size op: "
               << mlir::debugString(set_dimension_size_op.getType());
  return set_dimension_size_op;
}
}  // namespace

absl::StatusOr<std::vector<DeviceRefDimensions>> GetTraversalOutputDimensions(
    mlir::MLIRContext& mlir_context,
    const PythonContext* absl_nullable python_context,
    absl::Span<const DeviceBufferRef> inputs,
    absl::Span<const DeviceBufferRef> outputs,
    absl::Span<const std::shared_ptr<DeviceBufferList>> execution_order) {
  // Initialize the module builder and main function builder.
  std::string module_name =
      BuildModuleNameFromPyContext(mlir_context, python_context);

  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");

  absl::flat_hash_map<DeviceBufferRef, mlir::MlirOp> ref_to_op_map;
  // Create Mlir Ops for all inputs.
  for (const DeviceBufferRef& input : inputs) {
    auto type =
        makeTensorType(mlir_context, input.dimensions(), input.element_type());
    mlir::MlirOp mlir_op = mlir::func::Argument(fb, type);
    TT_ASSIGN_OR_RETURN(mlir_op, ConvertToBounded(fb, mlir_op, input));
    ABSL_VLOG(3) << "[GetTraversalOutputDimensions] input op: "
                 << mlir::debugString(mlir_op.getType());
    ref_to_op_map[input] = mlir_op;
  }

  // Create Mlir Ops for all deferred ops.
  for (const SharedDeviceBufferList& node : execution_order) {
    // Get the deferred op we need to build.
    const DeferredOp* absl_nullable maybe_deferred_op = node->deferred_op();
    ABSL_CHECK(maybe_deferred_op != nullptr)  // CRASH_OK=DeviceBufferList
                                              // should have a
                                              // deferred op.
        << "DeviceBufferList in execution_order has no deferred op";
    const DeferredOp& deferred_op = *maybe_deferred_op;

    // Get the MlirOps for all inputs.
    std::vector<mlir::MlirOp> deferred_inputs;
    for (const DeviceBufferRef& input : deferred_op.inputs()) {
      ABSL_CHECK(ref_to_op_map.contains(input))  // CRASH_OK=input buffer should
                                                 // always be in ref_to_op_map.
          << "DeviceBufferRef not found in ref_to_op_map: "
          << input.DebugString();
      deferred_inputs.push_back(ref_to_op_map.at(input));
    }

    // Build the deferred op's MlirOp.
    ScopedPythonContextProvider provider(deferred_op.op_context().Copy(), &fb);
    TT_ASSIGN_OR_RETURN(
        DynamicMlirOpResults results,
        deferred_op.op_builder()(fb, absl::MakeSpan(deferred_inputs)));
    ABSL_CHECK_EQ(results.size(), node->size())  // CRASH_OK=deferred op should
                                                 // return the correct number of
                                                 // results.
        << "Deferred op " << deferred_op.op_name() << " returned "
        << results.size() << " results, but node size is " << node->size();
    for (int64_t index = 0; index < node->size(); ++index) {
      mlir::MlirOp deferred_op_mlir_op = results[index];
      TT_ASSIGN_OR_RETURN(DeviceBufferRef ref,
                          DeviceBufferRef::Create(node, index));
      ABSL_VLOG(3) << "[GetTraversalOutputDimensions] built output " << index
                   << " of deferred op " << deferred_op.op_name() << ": "
                   << mlir::debugString(deferred_op_mlir_op.getType());
      ref_to_op_map[std::move(ref)] = std::move(deferred_op_mlir_op);
    }
  }

  std::vector<DeviceRefDimensions> results;
  results.reserve(outputs.size());
  for (const DeviceBufferRef& output : outputs) {
    ABSL_CHECK(ref_to_op_map.contains(output))  // CRASH_OK=output buffer should
                                                // always be in ref_to_op_map.
        << "DeviceBufferRef not found in ref_to_op_map: "
        << output.DebugString();
    auto mlir_op = ref_to_op_map.at(output);
    auto dims = GetDimensions(mlir_op);
    results.push_back({output, std::move(dims)});
  }

  return results;
}

}  // namespace torch_tpu
