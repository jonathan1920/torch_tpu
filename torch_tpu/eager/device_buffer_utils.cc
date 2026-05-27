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

#include "torch_tpu/eager/device_buffer_utils.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRef> CreateConstantDeviceBufferRef(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type) {
  // Create the components of the DeferredOp.
  auto op_name = OpName::kTorchTpuInternalConstant;
  ScopedPythonContextCapturer capturer(op_name);

  // Create the cache keys for the op parameters.
  // Since a change in data causes recompilation, we include the hash of the
  // tensor data as part of the cache key.
  auto op_param_cache_keys = OpParamCacheKeys::Empty();
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("data", absl::HashOf(cpu_tensor_data)));
  TT_RETURN_IF_ERROR(op_param_cache_keys.SetParam("dimensions", dimensions));
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("element_type", element_type));

  // The op returns a single tensor with the given shape and dtype.
  std::vector<Shape> output_shapes;
  output_shapes.push_back(Shape(dimensions, element_type));  // intentional copy

  auto op_builder = [cpu_tensor_data = std::move(cpu_tensor_data), element_type,
                     dimensions = std::move(dimensions)](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "unexpected input to constant op";

    auto ranked_tensor_type =
        mlir::makeTensorType(builder.getContext(), dimensions, element_type);

    if (element_type == mlir::ElementType::PRED) {
      // Special case for boolean tensors.
      //
      // PyTorch stores booleans as one-per-byte on CPU, but XLA uses packed
      // 1-bit booleans. So we just make a constant byte tensor and use
      // stablehlo.convert, which will do the packing.
      //
      // DenseIntElementsAttr::get has an assertion that checks if the
      // signedness of the data, as indicated by
      // std::numeric_limits<char>::is_signed, matches the signedness of the
      // element type. However, whether or not char is signed is implementation
      // defined, which can cause assertion failures in some environments.
      // So we match the signedness of the byte tensor (using I8 or UI8) to the
      // signedness of char according to the current implementation.
      const auto byte_element_type = std::numeric_limits<char>::is_signed
                                         ? mlir::ElementType::I8
                                         : mlir::ElementType::UI8;

      auto shaped_byte_tensor_type = mlir::makeTensorType(
          builder.getContext(), dimensions, byte_element_type);
      auto shaped_byte_constant = mlir::stablehlo::Constant(
          builder, mlir::DenseIntElementsAttr::get(shaped_byte_tensor_type,
                                                   cpu_tensor_data));
      return DynamicMlirOpResults{
          mlir::stablehlo::Convert(ranked_tensor_type, shaped_byte_constant)};
    }

    auto dense_elements_attr = mlir::DenseElementsAttr::getFromRawBuffer(
        ranked_tensor_type, cpu_tensor_data);
    return DynamicMlirOpResults{
        mlir::stablehlo::Constant(builder, dense_elements_attr)};
  };

  // OpSplitMode is kNone; we don't need to split around a constant.
  // No device inputs, so no aliased inputs.
  TT_ASSIGN_OR_RETURN(
      auto results,
      DeviceBufferList::CreateDeferred(
          op_name, std::move(op_builder), /*inputs=*/{},
          std::move(op_param_cache_keys), std::move(output_shapes)));
  TT_RET_CHECK(results.size() == 1, error::kInternal)
      << "CreateConstantDeviceBufferRef should return exactly one output";
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> CreateEmptyDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  auto op_builder = [dimensions =
                         CopyIntVector(absl::MakeConstSpan(dimensions)),
                     element_type](mlir::MlirBuilder& builder,
                                   absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "CreateEmptyDeviceBufferRef should not be called with any inputs";
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, element_type, dimensions)};
  };
  Shape output_shape(std::move(dimensions), element_type);
  TT_ASSIGN_OR_RETURN(auto results,
                      DeviceBufferList::CreateDeferred(
                          OpName::kEmpty, std::move(op_builder), /*inputs=*/{},
                          OpParamCacheKeys::Empty(), {std::move(output_shape)},
                          OpSplitMode::kNone,
                          /*donated_indices=*/{}));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> CreateZeroSizeDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type) {
  bool is_zero_sized = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_sized = true;
      break;
    }
  }
  TT_RET_CHECK(is_zero_sized, error::kInvalidArgument)
      << "CreateZeroSizeDeviceBufferRef requires a zero-sized tensor, but got: "
      << ToString(dimensions);
  return CreateConstantDeviceBufferRef({}, std::move(dimensions), element_type);
}

}  // namespace torch_tpu
