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
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

MlirUnaryOpBuilder GetPaddingOpBuilder(int64_t dimension, int64_t upper_bound) {
  return [dimension,
          upper_bound](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
    auto padding_value = MakeScalarConstant(input_op.getBuilder(), 0,
                                            input_type.getElementType());
    Dimensions edge_padding_low(input_type.getRank(), 0);
    Dimensions edge_padding_high(input_type.getRank(), 0);
    edge_padding_high[dimension] =
        upper_bound - input_type.getDimSize(dimension);
    if (edge_padding_high[dimension] == 0) {
      return input_op;
    }
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
  MlirUnaryOpBuilder op_builder =
      GetPaddingOpBuilder(dimension_index, upper_bound);
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

}  // namespace torch_tpu
