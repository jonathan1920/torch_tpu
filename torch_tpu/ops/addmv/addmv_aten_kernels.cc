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

#include "torch_tpu/ops/addmv/addmv_aten_kernels.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Matrix-vector dot product helper.
mlir::MlirOp MatVecDot(mlir::MlirOp mat_op, mlir::MlirOp vec_op, double alpha,
                       mlir::ElementType out_dtype) {
  const mlir::RankedTensorType mat_type = GetTensorTypeOrDie(mat_op);
  const mlir::RankedTensorType vec_type = GetTensorTypeOrDie(vec_op);

  // TODO: XLA doesn't support matmuls with i64, so we convert them to f64.
  const bool is_any_i64 = mat_type.getElementType().isInteger(64) ||
                          vec_type.getElementType().isInteger(64);
  if (is_any_i64) {
    mat_op = stablehlo::ConvertElementType(mat_op, mlir::ElementType::F64);
    vec_op = stablehlo::ConvertElementType(vec_op, mlir::ElementType::F64);
  }

  // After matmul, we want the result back in the original type, so we specify
  // the preferred element type to be the output type.
  mlir::MlirOp mv_op = stablehlo::DotGeneral(
      mat_op, vec_op,
      /*dotDimsAttr=*/
      stablehlo::DotDimensionNumbersAttr::get(
          &mat_op.getContext(), /*lhs_batch_dimensions=*/{},
          /*rhs_batch_dimensions=*/{}, /*lhs_contracting_dimensions=*/{1},
          /*rhs_contracting_dimensions=*/{0}),
      /*precisionConfig=*/
      stablehlo::PrecisionConfigAttr::get(
          &mat_op.getContext(),
          {stablehlo::Precision::DEFAULT, stablehlo::Precision::DEFAULT}));

  // Cast the matmul result back to the original type right away. CPU behavior
  // expects the computation to be lossy not precise.
  if (is_any_i64) {
    mv_op = stablehlo::ConvertElementType(mv_op, out_dtype);
  }
  return mv_op;
}

absl::StatusOr<mlir::MlirOp> BuildAddmvShloBetaZero(mlir::MlirOp mat_op,
                                                    mlir::MlirOp vec_op,
                                                    double alpha,
                                                    mlir::ElementType out_dtype,
                                                    Dimensions result_shape) {
  auto mv_op = MatVecDot(mat_op, vec_op, alpha, out_dtype);
  mlir::MlirOp alpha_op = MakeConstantLike(mv_op, alpha);
  return stablehlo::Mul(alpha_op, mv_op);
}

absl::StatusOr<mlir::MlirOp> BuildAddmvShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat_op, mlir::MlirOp vec_op, double beta,
    double alpha, mlir::ElementType out_dtype, Dimensions result_shape) {
  auto mv_op = MatVecDot(mat_op, vec_op, alpha, out_dtype);

  mlir::MlirOp alpha_op = MakeConstantLike(mv_op, alpha);
  mlir::MlirOp beta_op = MakeConstantLike(self_op, beta);
  auto bias_op = stablehlo::Mul(beta_op, self_op);
  TT_ASSIGN_OR_RETURN(auto bias_op_bcast,
                      ApplyBroadcastIfNeeded(bias_op, mv_op));
  bias_op = bias_op_bcast[0];
  mv_op = stablehlo::Mul(alpha_op, mv_op);
  return stablehlo::Add(bias_op, mv_op);
}

absl::StatusOr<DeviceBufferRef> Addmv(const at::Tensor& self,
                                      const at::Tensor& mat,
                                      const at::Tensor& vec,
                                      const at::Scalar& beta,
                                      const at::Scalar& alpha, at::Tensor& out,
                                      OpParamCacheKeys& param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  Dimensions result_shape = {mat.size(0)};
  DispatchOpOptions<1> options = {.out_dtype = out_dtype,
                                  .out_dims = result_shape,
                                  .op_param_cache_keys = std::move(param_keys)};

  // If beta is zero, the content of self is not used, so we can skip it.
  // Because beta is zero, PyTorch will pass an uninitialized tensor to self as
  // an optimization because anything multiplied by zero is zero. We don't allow
  // reading from uninitialized tensors (at::empty) at the moment, so instead of
  // dispatching to ternary op, we dispatch to binary op without self.
  if (beta.toDouble() == 0.0) {
    auto op_builder = [alpha, out_dtype,
                       result_shape](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [mat_op, vec_op] = inputs;
      return BuildAddmvShloBetaZero(mat_op, vec_op, alpha.toDouble(), out_dtype,
                                    result_shape);
    };
    TT_ASSIGN_OR_RETURN(auto result_buf,
                        DispatchOp<2>(OpName::kAddmvOut, std::move(op_builder),
                                      {mat, vec}, std::move(options)));
    return result_buf;
  }

  auto op_builder = [beta, alpha, out_dtype,
                     result_shape](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, mat_op, vec_op] = inputs;
    return BuildAddmvShlo(self_op, mat_op, vec_op, beta.toDouble(),
                          alpha.toDouble(), out_dtype, result_shape);
  };
  TT_ASSIGN_OR_RETURN(auto result_buf,
                      DispatchOp<3>(OpName::kAddmvOut, std::move(op_builder),
                                    {self, mat, vec}, std::move(options)));
  return result_buf;
}

absl::Status CheckAddmvInputs(const at::Tensor& self, const at::Tensor& mat,
                              const at::Tensor& vec, const at::Scalar& beta,
                              const at::Scalar& alpha) {
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "the dtype of the first argument cannot be bool";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch implementation promotes dtype
                 // before dtype check.
      !IsBool(mat), error::kInvalidArgument)
      << "the dtype of the second argument cannot be bool";
  TT_RET_CHECK(mat.dim() == 2, error::kInvalidArgument)
      << "expected the second argument to be a matrix (2D tensor), got "
      << mat.dim() << "D tensor";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch implementation promotes dtype
                 // before dtype check.
      !IsBool(vec), error::kInvalidArgument)
      << "the dtype of the third argument cannot be bool";
  TT_RET_CHECK(vec.dim() == 1, error::kInvalidArgument)
      << "expected the third argument to be a vector (1D tensor), got "
      << vec.dim() << "D tensor";

  TT_RET_CHECK(!IsBool(alpha) && !IsComplex(alpha), error::kInvalidArgument)
      << "expected the dtype of alpha to be neither complex nor bool, got "
      << ToString(alpha.type());

  TT_RET_CHECK(!IsBool(beta) && !IsComplex(beta), error::kInvalidArgument)
      << "expected the dtype of beta to be neither complex nor bool, got "
      << ToString(beta.type());

  TT_RET_CHECK(mat.size(1) == vec.size(0), error::kInvalidArgument)
      << "expected the last dimension of the second argument (matrix of size ["
      << absl::StrJoin(mat.sizes(), /* separator= */ ", ")
      << "]) to match the first dimension of the third argument (vector of "
         "size ["
      << absl::StrJoin(vec.sizes(), /* separator= */ ", ") << "]), got "
      << mat.size(1) << " vs " << vec.size(0);

  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenAddmvOut(const at::Tensor& self, const at::Tensor& mat,
                         const at::Tensor& vec, const at::Scalar& beta,
                         const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(OpName::kAddmvOut, param_keys, (self, mat, vec, beta, alpha, out), {
    TT_THROW_IF_ERROR(CheckAddmvInputs(self, mat, vec, beta, alpha));
    TT_ASSIGN_OR_THROW(auto result_buf,
                       Addmv(self, mat, vec, beta, alpha, out, param_keys));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
