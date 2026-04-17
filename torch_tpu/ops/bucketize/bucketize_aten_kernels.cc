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

#include "torch_tpu/ops/bucketize/bucketize_aten_kernels.h"

#include <cstdint>
#include <numeric>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::Status ValidateInputAndBoundaries(const at::Tensor& self,
                                        const at::Tensor& boundaries) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(!IsComplex(dtype), error::kInvalidArgument)
      << "self must not be complex, got '" << self.scalar_type() << "'";

  TT_ASSIGN_OR_RETURN(const mlir::ElementType boundaries_dtype,
                      ConvertTo<mlir::ElementType>(boundaries.scalar_type()));
  TT_RET_CHECK(!IsComplex(boundaries_dtype), error::kInvalidArgument)
      << "boundaries must not be complex, got '" << boundaries.scalar_type()
      << "'";
  TT_RET_CHECK(boundaries.dim() == 1, error::kInvalidArgument)
      << "boundaries tensor must be 1 dimension, got dim(" << boundaries.dim()
      << ")";

  return absl::OkStatus();
}

struct BroadcastResult {
  mlir::MlirOp self_bcast_op;
  mlir::MlirOp boundaries_bcast_op;
};

absl::StatusOr<BroadcastResult> BroadcastArgs(mlir::MlirOp self_op,
                                              mlir::MlirOp boundaries_op) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType boundaries_type =
      GetTensorTypeOrDie(boundaries_op);
  // TODO(b/499034385): Support dynamic dimensions.
  TT_RET_CHECK(self_type.hasStaticShape(), error::kInvalidArgument)
      << "expected 0 dynamic dimensions in self, got "
      << self_type.getNumDynamicDims();
  TT_RET_CHECK(boundaries_type.hasStaticShape(), error::kInvalidArgument)
      << "expected 0 dynamic dimensions in boundaries, got "
      << boundaries_type.getNumDynamicDims();
  const int64_t num_boundaries = boundaries_type.getShape()[0];

  // Add the boundaries dimension to the input shape to get the target broadcast
  // shape.
  Dimensions target_shape(self_type.getShape().begin(),
                          self_type.getShape().end());
  target_shape.push_back(num_boundaries);

  // Broadcast the input (self_op).
  const int64_t input_rank = self_type.getRank();
  Dimensions self_bcast_dims(input_rank);
  std::iota(self_bcast_dims.begin(), self_bcast_dims.end(), 0);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp self_bcast_op,
                      Broadcast(self_op, target_shape, self_bcast_dims));

  // Broadcast the boundaries (boundaries_op).
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp boundaries_bcast_op,
                      Broadcast(boundaries_op, target_shape, {input_rank}));

  return BroadcastResult{self_bcast_op, boundaries_bcast_op};
}

absl::StatusOr<mlir::MlirOp> BuildCompareAllShlo(mlir::MlirOp self_op,
                                                 mlir::MlirOp boundaries_op,
                                                 mlir::ElementType out_dtype,
                                                 bool right) {
  // Broadcast the input and boundaries for element-wise comparison.
  TT_ASSIGN_OR_RETURN(const BroadcastResult broadcast_result,
                      BroadcastArgs(self_op, boundaries_op));
  mlir::MlirOp self_bcast_op = broadcast_result.self_bcast_op;
  mlir::MlirOp boundaries_bcast_op = broadcast_result.boundaries_bcast_op;

  // PyTorch bucketize logic:
  // right=true  -> count boundaries b such that b <= input.
  // right=false -> count boundaries b such that b < input.
  const auto direction = right ? mlir::stablehlo::ComparisonDirection::LE
                               : mlir::stablehlo::ComparisonDirection::LT;

  // Compare the broadcasted input with the broadcasted boundaries, and convert
  // the boolean results to the desired output type (I32/I64) for summation.
  const mlir::MlirOp compare_op = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Compare(boundaries_bcast_op, self_bcast_op, direction),
      out_dtype);

  // Reduce the compared results along the last dimension (boundaries
  // dimension).
  const int64_t reduction_dim = GetTensorTypeOrDie(self_op).getRank();
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::SmallVector<mlir::MlirOp> reduce_results =
      mlir::stablehlo::Reduce(
          builder, {compare_op}, {MakeScalarConstant(builder, 0, out_dtype)},
          [out_dtype](mlir::RegionBuilder& region_builder) {
            mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
                mlir::getElementType(region_builder.getContext(), out_dtype),
                region_builder.getRegion(), region_builder.getOpBuilder());
          },
          /*dimensions=*/{reduction_dim});

  TT_RET_CHECK(reduce_results.size() == 1, error::kInternal)
      << "expected 1 result from reduce op, got " << reduce_results.size();
  return reduce_results[0];
}

absl::StatusOr<mlir::MlirOp> BuildBucketizeShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp boundaries_op,
                                                mlir::ElementType out_dtype,
                                                bool right) {
  // TODO(b/499034385): Current, only the "compare all" algorithm is supported.
  // In the future, we would like to implement search-based bucketing as well,
  // similar to jax.numpy.searchsorted.
  return BuildCompareAllShlo(self_op, boundaries_op, out_dtype, right);
}

absl::StatusOr<DeviceBufferRef> Bucketize(const at::Tensor& self,
                                          const at::Tensor& boundaries,
                                          bool out_int32, bool right,
                                          OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateInputAndBoundaries(self, boundaries));

  TT_ASSIGN_OR_RETURN(
      const auto common_dtype,
      ConvertTo<mlir::ElementType>(at::result_type(self, boundaries)));
  const auto out_dtype =
      out_int32 ? mlir::ElementType::I32 : mlir::ElementType::I64;
  auto op_builder = [common_dtype, out_dtype,
                     right](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, boundaries_op] = inputs;

    // Convert both input and boundaries to a common dtype as a consistent type
    // is required for comparison operations.
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_self_op,
                        CastIfNeeded(self_op, common_dtype));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_boundaries_op,
                        CastIfNeeded(boundaries_op, common_dtype));

    TT_ASSIGN_OR_RETURN(
        const mlir::MlirOp result_op,
        BuildBucketizeShlo(cast_self_op, cast_boundaries_op, out_dtype, right));
    return result_op;
  };

  return DispatchOp<2>(std::move(op_builder), {self, boundaries},
                       {.out_dtype = out_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenBucketizeScalar(const at::Scalar& self,
                               const at::Tensor& boundaries, bool out_int32,
                               bool right) {
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(OpName::kBucketizeScalar, param_keys,
            (promoted_self, boundaries, out_int32, right), {
              TT_ASSIGN_OR_THROW(at::Tensor self_tensor,
                                 promoted_self.GetTensor());
              TT_ASSIGN_OR_THROW(DeviceBufferRef result_buffer,
                                 Bucketize(self_tensor, boundaries, out_int32,
                                           right, std::move(param_keys)));
              return MakeTensor(result_buffer);
            });
}

at::Tensor AtenBucketizeTensor(const at::Tensor& self,
                               const at::Tensor& boundaries, bool out_int32,
                               bool right) {
  TT_KERNEL(OpName::kBucketizeTensor, param_keys,
            (self, boundaries, out_int32, right), {
              TT_ASSIGN_OR_THROW(DeviceBufferRef result_buffer,
                                 Bucketize(self, boundaries, out_int32, right,
                                           std::move(param_keys)));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenBucketizeTensorOut(const at::Tensor& self,
                                   const at::Tensor& boundaries, bool out_int32,
                                   bool right, at::Tensor& out) {
  TT_KERNEL(OpName::kBucketizeTensorOut, param_keys,
            (self, boundaries, out_int32, right, out), {
              TT_ASSIGN_OR_THROW(DeviceBufferRef result_buffer,
                                 Bucketize(self, boundaries, out_int32, right,
                                           std::move(param_keys)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

}  // namespace torch_tpu
