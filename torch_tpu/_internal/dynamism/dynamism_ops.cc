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

#include "torch_tpu/_internal/dynamism/dynamism_ops.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::StatusOr<MlirUnaryOpBuilder> GetPaddingOpBuilder(
    Indices dimension_indices, Dimensions upper_bounds) {
  TT_RET_CHECK(dimension_indices.size() == upper_bounds.size(),
               error::kInvalidArgument)
      << "dimensions and upper_bounds must have the same size.";
  return [dimension_indices = std::move(dimension_indices),
          upper_bounds = std::move(upper_bounds)](
             mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
    TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
        << "input type must have static shape, but has dynamic shape.";

    Dimensions edge_padding_low(input_type.getRank(), 0);
    Dimensions edge_padding_high(input_type.getRank(), 0);
    bool was_padded = false;
    for (int i = 0; i < dimension_indices.size(); ++i) {
      int64_t dimension_index = dimension_indices[i];
      int64_t upper_bound = upper_bounds[i];
      TT_RET_CHECK(
          dimension_index >= 0 && dimension_index < input_type.getRank(),
          error::kInvalidArgument)
          << "dimension index " << dimension_index
          << " is out of bounds for tensor " << input_type.getRank()
          << " dimensions";
      int padding_size = upper_bound - input_type.getDimSize(dimension_index);
      TT_RET_CHECK(padding_size >= 0, error::kInvalidArgument)
          << "padding size must be non-negative, but is " << padding_size;
      if (padding_size > 0) {
        was_padded = true;
      }
      edge_padding_high[dimension_index] = padding_size;
    }
    if (!was_padded) {
      return input_op;
    }
    auto padding_value = MakeScalarConstant(input_op.getBuilder(), 0,
                                            input_type.getElementType());
    auto pad_op =
        mlir::stablehlo::Pad(input_op, padding_value, edge_padding_low,
                             edge_padding_high, edge_padding_low);
    return pad_op;
  };
}

NAryMlirOpBuilder<2, 1> GetSetDimensionSizeOpBuilder(int64_t dimension) {
  return [dimension](FixedSizeSpan<mlir::MlirOp, 2> input_ops)
             -> absl::StatusOr<mlir::MlirOp> {
    auto [input_op, dimension_size_op] = input_ops;
    auto set_dimension_size_op = mlir::stablehlo::SetDimensionSize(
        input_op, dimension_size_op, dimension);
    return set_dimension_size_op;
  };
}

}  // namespace

absl::StatusOr<DeviceBufferRef> PadDynamicDimension(DeviceBufferRef input,
                                                    int64_t dimension_index,
                                                    int64_t upper_bound) {
  ScopedPythonContextCapturer capturer(OpName::kSetDimensionSize);
  TT_RET_CHECK(
      dimension_index >= 0 && dimension_index < input.dimensions().size(),
      error::kInvalidArgument)
      << "dimension index " << dimension_index
      << " is out of bounds for tensor " << input.dimensions().size()
      << " dimensions";
  TT_RET_CHECK(upper_bound >= input.dimensions()[dimension_index],
               error::kInvalidArgument)
      << "upper bound must be at least the current dimension size, but is "
      << upper_bound;
  TT_ASSIGN_OR_RETURN(auto op_builder,
                      GetPaddingOpBuilder({dimension_index}, {upper_bound}));
  Shape original_shape = input.device_buffer_list()->shapes()[input.index()];
  Shape padded_shape = original_shape;
  padded_shape.dimensions[dimension_index] = upper_bound;
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> padded_inputs,
                      DeviceBufferList::CreateDeferred(
                          OpName::kPadUninitialized_,
                          ToMlirOpBuilder<1, 1>(std::move(op_builder)), {input},
                          {}, {padded_shape}));
  TT_RET_CHECK(padded_inputs.size() == 1, error::kInternal)
      << "outputs must have exactly one buffer.";
  return padded_inputs[0];
}

absl::StatusOr<std::array<DeviceBufferRef, 2>> SetDynamicDimensionSize(
    DeviceBufferRef input, int64_t dimension_index,
    int64_t original_dimension_size) {
  ScopedPythonContextCapturer capturer(OpName::kSetDimensionSize);
  TT_RET_CHECK(
      dimension_index >= 0 && dimension_index < input.dimensions().size(),
      error::kInvalidArgument)
      << "dimension index " << dimension_index
      << " is out of bounds for tensor " << input.dimensions().size()
      << " dimensions";
  int32_t dimension_size = static_cast<int32_t>(original_dimension_size);
  // Using TpuMallocAndMemcpyHtoD to avoid a circular dependency when using
  // MakeBuffer.
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef size_buffer_ref,
      TpuMallocAndMemcpyHtoD(&dimension_size, mlir::ElementType::I32, {}));
  // Create a deferred op that sets the dynamic dimension size.
  auto set_dimension_size_op_builder =
      GetSetDimensionSizeOpBuilder(dimension_index);
  Shape original_shape = input.device_buffer_list()->shapes()[input.index()];
  original_shape.dimensions[dimension_index] = original_dimension_size;
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> set_dimension_size_buffer_refs,
      DeviceBufferList::CreateDeferred(
          OpName::kSetDimensionSize,
          ToMlirOpBuilder<2, 1>(std::move(set_dimension_size_op_builder)),
          {input, size_buffer_ref}, {}, {original_shape}));
  TT_RET_CHECK(set_dimension_size_buffer_refs.size() == 1, error::kInternal)
      << "outputs must have exactly one buffer.";
  return std::array<DeviceBufferRef, 2>{set_dimension_size_buffer_refs[0],
                                        size_buffer_ref};
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetPadModule(
    mlir::MLIRContext& mlir_context, absl::Span<const Shape> shapes) {
  ABSL_VLOG(2) << "[GetPadModule] Creating a padding module for dynamism with "
               << shapes.size() << " shapes.";
  TT_RET_CHECK(!shapes.empty(), error::kInvalidArgument)
      << "PadModule requires at least one shape.";
  std::string module_name = "pad_module";
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");
  DynamicMlirOpResults results;
  results.reserve(shapes.size());
  // Add a function parameter for each input shape.
  for (int i = 0; i < shapes.size(); ++i) {
    const Shape& shape = shapes[i];
    auto type = makeTensorType(mlir_context, shape.dimensions, shape.dtype);
    auto input_op = mlir::func::Argument(fb, type);
    if (shape.dynamic_dimensions.empty()) {
      results.push_back(input_op);
      continue;
    }
    Indices dimension_indices;
    Dimensions upper_bounds;
    for (auto dynamic_dimension : shape.dynamic_dimensions) {
      dimension_indices.push_back(dynamic_dimension.dimension);
      upper_bounds.push_back(dynamic_dimension.upper_bound);
    }
    ABSL_VLOG(2) << "[GetPadModule] Input " << i << " has "
                 << dimension_indices.size() << " dynamic dimensions";
    TT_ASSIGN_OR_RETURN(auto op_builder,
                        GetPaddingOpBuilder(std::move(dimension_indices),
                                            std::move(upper_bounds)));
    TT_ASSIGN_OR_RETURN(auto padded_op, op_builder(input_op));
    results.push_back(padded_op);
    for (auto dynamic_dimension : shape.dynamic_dimensions) {
      results.push_back(mlir::stablehlo::GetDimensionSize(
          input_op, dynamic_dimension.dimension));
    }
  }
  mlir::func::Return(fb, results);
  auto module = mb.build();
  return module;
}

}  // namespace torch_tpu
