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

#include "torch_tpu/ops/reductions/var_aten_kernels.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/OptionalArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<2, 1> CreateVarBuilder(Dimensions dim_vec,
                                         ReductionMode reduction_mode,
                                         int64_t reduction_size,
                                         mlir::ElementType scalar_dtype_mlir) {
  return [dim_vec = std::move(dim_vec), reduction_mode, reduction_size,
          scalar_dtype_mlir](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, correction_op] = inputs;
    mlir::MlirBuilder& builder = self_op.getBuilder();
    mlir::Type type = GetTensorTypeOrDie(self_op).getElementType();

    // sum_tensor = ReduceSum(self, dim, keep_dims=true)
    TT_ASSIGN_OR_RETURN(
        auto sum_op, BuildSumShlo(self_op, dim_vec, ReductionMode::kKeepDims));

    // mean = sum_tensor / reduction_size
    mlir::MlirOp reduction_size_const;
    if (llvm::isa<mlir::ComplexType>(type)) {
      reduction_size_const = MakeScalarConstant(
          builder,
          std::complex<double>(static_cast<double>(reduction_size), 0.0), type);
    } else {
      reduction_size_const = MakeScalarConstant(
          builder, static_cast<double>(reduction_size), type);
    }
    TT_ASSIGN_OR_RETURN(auto mean, BuildDivShlo(sum_op, reduction_size_const));

    // diff = self - mean
    TT_ASSIGN_OR_RETURN(auto diff, BuildSubShlo(self_op, mean));

    // conj_diff = conj(diff)
    TT_ASSIGN_OR_RETURN(auto conj_diff, BuildConjPhysicalShlo(diff));

    // squared_diff = diff * conj_diff
    TT_ASSIGN_OR_RETURN(auto squared_diff, BuildMulShlo(diff, conj_diff));

    // sum_sq_tensor = ReduceSum(squared_diff, dim, keep_dims=keep_dim)
    TT_ASSIGN_OR_RETURN(
        auto sum_sq_op,
        BuildSumShlo(squared_diff, dim_vec, reduction_mode, scalar_dtype_mlir));

    // denom = max(reduction_size - correction, 0)
    mlir::Type corr_type = GetTensorTypeOrDie(correction_op).getElementType();
    auto reduction_size_const_corr = MakeScalarConstant(
        builder, static_cast<double>(reduction_size), corr_type);
    TT_ASSIGN_OR_RETURN(auto denom_diff,
                        BuildSubShlo(reduction_size_const_corr, correction_op));
    auto zero = MakeScalarConstant(builder, 0.0, corr_type);
    TT_ASSIGN_OR_RETURN(auto denom, BuildMaximumShlo(denom_diff, zero));

    // res = sum_sq_tensor / denom
    return BuildDivShlo(sum_sq_op, denom);
  };
}

void WarnIfInvalidCorrection(OpName op_name, double correction_val,
                             int64_t reduction_size) {
  const double denom = reduction_size - correction_val;
  if (denom <= 0) {
    TORCH_WARN(GetRootOpName(op_name),
               "(): degrees of freedom (i.e., reduction "
               "size - correction) should "
               "be positive, got reduction size = ",
               reduction_size, ", correction = ", correction_val,
               ", and degrees of freedom = ", denom);
  }
}

}  // namespace

at::Tensor AtenVar(const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
                   const std::optional<at::Scalar>& correction, bool keep_dim) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kVar, param_keys, (self, dim, promoted_correction, keep_dim), {
        c10::ScalarType scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
        if (c10::isComplexType(scalar_dtype)) {
          scalar_dtype = c10::toRealValueType(scalar_dtype);
        }
        if (self.numel() == 0) {
          return at::full({}, std::numeric_limits<double>::quiet_NaN(),
                          self.options().dtype(scalar_dtype));
        }

        TT_ASSIGN_OR_THROW(at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(int64_t reduction_size,
                           GetReductionFactor(self, dim));
        double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kVar, correction_val, reduction_size);

        TT_ASSIGN_OR_THROW(mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);
        auto var_builder =
            CreateVarBuilder(std::move(canonicalized_dims), reduction_mode,
                             reduction_size, scalar_dtype_mlir);

        TT_ASSIGN_OR_THROW(
            auto result_bufs,
            (DispatchOp<2, 1>(std::move(var_builder), {self, correction_tensor},
                              {.out_dtype = scalar_dtype_mlir,
                               .out_dims = output_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(result_bufs));
      });
}

at::Tensor& AtenVarOut(const at::Tensor& self,
                       c10::OptionalArrayRef<int64_t> dim,
                       const std::optional<at::Scalar>& correction,
                       bool keep_dim, at::Tensor& out) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kVarOut, param_keys,
      (self, dim, promoted_correction, keep_dim, out), {
        c10::ScalarType scalar_dtype = out.scalar_type();
        TT_THROW_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
        if (self.numel() == 0) {
          at::full_out(out, {}, std::numeric_limits<double>::quiet_NaN());
          return out;
        }

        TT_ASSIGN_OR_THROW(at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kVarOut, correction_val,
                                reduction_size);

        TT_ASSIGN_OR_THROW(mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);
        auto var_builder =
            CreateVarBuilder(std::move(canonicalized_dims), reduction_mode,
                             reduction_size, scalar_dtype_mlir);

        TT_ASSIGN_OR_THROW(
            auto result_bufs,
            (DispatchOp<2, 1>(std::move(var_builder), {self, correction_tensor},
                              {.out_dtype = scalar_dtype_mlir,
                               .out_dims = output_dims,
                               .op_param_cache_keys = std::move(param_keys)})));

        bool is_inplace =
            out.is_alias_of(self) || out.is_alias_of(correction_tensor);
        OutputOpMode mode =
            is_inplace ? OutputOpMode::kInPlace : OutputOpMode::kOutOfPlace;
        TT_THROW_IF_ERROR(FinalizeOpOutput(out, std::move(result_bufs), mode));
        return out;
      });
}

}  // namespace torch_tpu
