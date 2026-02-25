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

#include "torch_tpu/ops/unary_aten_kernels.h"

#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"  // IWYU pragma: keep
#include "torch_tpu/eager/device.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"  // IWYU pragma: keep
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

// A "Simple Unary" op is an Aten op that:
// - takes 1 input and produces 1 output
// - the dtype and shape of the output is the same as the input
// - doesn't have any parameters
// TODO(b/442660800): minimize the number of kernels implemented by macros
#define TT_DEFINE_ATEN_UNARY_OUT(op_name, func_name, op_builder,        \
                                 the_out_dtype)                         \
  at::Tensor& func_name##Out(const at::Tensor& self, at::Tensor& out) { \
    const std::optional<c10::ScalarType> out_dtype = (the_out_dtype);   \
    TT_KERNEL(op_name, _, (self, out, out_dtype), {                     \
      TT_THROW_IF_ERROR(                                                \
          ::torch_tpu::UnaryOpOut(self, out, op_name, op_builder));     \
      return out;                                                       \
    });                                                                 \
  }                                                                     \
  TT_REQUIRE_SEMICOLON_

// A floating-point (FP) only op is a simple Aten op (see above), with the
// additional restriction that it will cast its integer arguments to the default
// PyTorch data type. This captures the default dtype at **dispatch** time, not
// **compile** time, to mirror eager CUDA behavior.
// This means the default dtype also needs to be included as a parameter key.
// TODO(b/440585584): parameterize by explicit output dtype, not by default
// dtype, and remove the cache to avoid unnecessary cache misses when unused.
// TODO(b/442660800): minimize the number of kernels implemented by macros
#define TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(op_name, func_name, op_builder, \
                                         out_dtype_val)                  \
  at::Tensor& func_name##Out(const at::Tensor& self, at::Tensor& out) {  \
    const at::ScalarType default_dtype =                                 \
        c10::get_default_dtype_as_scalartype();                          \
    TT_KERNEL(op_name, param_keys, (self, out, default_dtype), {         \
      TT_ASSIGN_OR_THROW(                                                \
          const auto default_mlir_type,                                  \
          ::torch_tpu::ConvertTo<mlir::ElementType>(default_dtype));     \
      ::torch_tpu::MlirUnaryOpBuilder op_builder_with_default_dtype =    \
          [default_mlir_type](                                           \
              mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {      \
        return op_builder(input, default_mlir_type);                     \
      };                                                                 \
      TT_THROW_IF_ERROR(::torch_tpu::UnaryOpOut(                         \
          self, out, op_name, std::move(op_builder_with_default_dtype),  \
          {.op_param_cache_keys = std::move(param_keys),                 \
           .out_dtype = out_dtype_val}));                                \
      return out;                                                        \
    });                                                                  \
  }                                                                      \
  TT_REQUIRE_SEMICOLON_

namespace torch_tpu {

absl::StatusOr<at::Tensor> UnaryOp(const at::Tensor& self, OpName op_name,
                                   MlirUnaryOpBuilder op_builder,
                                   UnaryOpOptions options) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(
                          options.out_dtype.value_or(self.scalar_type())));
  const at::IntArrayRef output_dims = options.out_dims.has_value()
                                          ? options.out_dims.value()
                                          : at::IntArrayRef(self.sizes());
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          op_name, std::move(op_builder), self,
          {.out_dtype = output_dtype,
           .out_dims = output_dims,
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  return MakeTensor(std::move(result_buf));
}

absl::Status UnaryOpInPlace(at::Tensor& self, OpName op_name,
                            MlirUnaryOpBuilder op_builder,
                            UnaryOpOptions options) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          op_name, std::move(op_builder), self,
          {.out_dtype = output_dtype,
           .out_dims = self.sizes(),
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  return AssignBufferToAtTensor(std::move(result_buf), self);
}

absl::Status UnaryOpOut(const at::Tensor& self, at::Tensor& out, OpName op_name,
                        MlirUnaryOpBuilder op_builder, UnaryOpOptions options) {
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kFailedPrecondition)
      << "out not on PrivateUse1";

  at::IntArrayRef shape =
      options.out_dims.has_value() ? *options.out_dims : self.sizes();
  at::ScalarType dtype =
      options.out_dtype.has_value() ? *options.out_dtype : out.scalar_type();
  TT_RET_CHECK(out.scalar_type() == dtype, error::kInvalidArgument)
      << "out tensor dtype expected to be " << dtype << ", got "
      << out.scalar_type();

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(
                          options.out_dtype.value_or(out.scalar_type())));
  const at::IntArrayRef output_dims =
      options.out_dims.has_value() ? options.out_dims.value() : self.sizes();
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          op_name, std::move(op_builder), self,
          {.out_dtype = output_dtype,
           .out_dims = output_dims,
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  at::native::resize_output(out, shape);
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

// Returns the input dtype if it is a floating-point type.
// Otherwise, returns the current default scalar dtype.
[[nodiscard]] static inline std::optional<c10::ScalarType> InferOutputDtype(
    const at::Tensor& self) {
  const c10::ScalarType input_dtype = self.scalar_type();
  if (c10::isFloatingType(input_dtype) || c10::isComplexType(input_dtype)) {
    return input_dtype;
  }
  return c10::get_default_dtype_as_scalartype();
}

// go/keep-sorted start
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAcosOut, AtenAcos, BuildAcosShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAcoshOut, AtenAcosh, BuildAcoshShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAsinOut, AtenAsin, BuildAsinShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAsinhOut, AtenAsinh, BuildAsinhShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAtanOut, AtenAtan, BuildAtanShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAtanhOut, AtenAtanh, BuildAtanhShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kCosOut, AtenCos, BuildCosShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kCoshOut, AtenCosh, BuildCoshShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kErfInvOut, AtenErfInv,
                                 BuildErfInvShlo, InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kErfOut, AtenErf, BuildErfShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kExpM1Out, AtenExpm1, BuildExpm1Shlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kExpOut, AtenExp, BuildExpShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLgammaOut, AtenLgamma,
                                 BuildLgammaShlo, InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog10Out, AtenLog10, BuildLog10Shlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog1pOut, AtenLog1p, BuildLog1pShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog2Out, AtenLog2, BuildLog2Shlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLogOut, AtenLog, BuildLogShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kReciprocalOut, AtenReciprocal,
                                 BuildReciprocalShlo, InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kRsqrOut, AtenRsqrt, BuildRsqrtShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSinOut, AtenSin, BuildSinShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSinhOut, AtenSinh, BuildSinhShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSqrOut, AtenSqrt, BuildSqrtShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kTanOut, AtenTan, BuildTanShlo,
                                 InferOutputDtype(self));
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kTanhOut, AtenTanh, BuildTanhShlo,
                                 InferOutputDtype(self));
// go/keep-sorted end

at::Tensor& AtenAbsOut(const at ::Tensor& self, at ::Tensor& out) {
  TT_KERNEL(OpName::kAbsOut, _, (self, out), {
    TT_THROW_IF_ERROR(UnaryOpOut(self, out, OpName::kAbsOut, BuildAbsShlo));
    return out;
  });
}

at::Tensor& AtenNegOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kNegOut, _, (self, out), {
    TT_THROW_IF_ERROR(UnaryOpOut(self, out, OpName::kNegOut, BuildNegShlo));
    return out;
  });
}

at::Tensor AtenRelu(const at::Tensor& self) {
  TT_KERNEL(OpName::kRelu, _, (self), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        ::torch_tpu::UnaryOp(self, OpName::kRelu, BuildReluShlo));
    return result;
  });
}

at::Tensor& AtenRelu_(at::Tensor& self) {
  TT_KERNEL(OpName::kRelu_, _, (self), {
    TT_THROW_IF_ERROR(UnaryOpInPlace(self, OpName::kRelu_, BuildReluShlo));
    return self;
  });
}

at::Tensor& AtenSignOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSignOut, _, (self, out), {
    TT_CHECK_THROW(!self.is_complex(), error::kInvalidArgument)
        << "does not support complex dtype. Please use torch.sgn() instead";
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(OpName::kSignOut, BuildSignShlo, self,
                      /*options=*/
                      {.out_dtype = out_dtype, .out_dims = self.sizes()}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenSignbitOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSignbitOut, _, (self, out), {
    const auto output_dtype = mlir::ElementType::PRED;
    auto op_builder =
        [output_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

      if (IsBooleanType(input_type) || IsUnsignedType(input_type)) {
        return MakeConstantLike(input, false, output_dtype);
      } else if (IsFloatType(input_type)) {
        auto zero_float = MakeConstantLike(input, +0.0f);
        return mlir::stablehlo::Compare(
            input, zero_float, mlir::stablehlo::ComparisonDirection::LT,
            mlir::stablehlo::ComparisonTypeAttr::get(
                &input.getContext(),
                mlir::stablehlo::ComparisonType::TOTALORDER));
      } else {
        auto zero = MakeConstantLike(input, 0);
        return mlir::stablehlo::Compare(
            input, zero, mlir::stablehlo::ComparisonDirection::LT);
      }
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(OpName::kSignbitOut, std::move(op_builder), self,
                      /*options=*/
                      {.out_dtype = output_dtype, .out_dims = out.sizes()}));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenTruncOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kTruncOut, _, (self, out), {
    TT_CHECK_THROW(self.scalar_type() != c10::ScalarType::Bool,
                   error::kInvalidArgument)
        << "does not support boolean types";
    TT_THROW_IF_ERROR(UnaryOpOut(self, out, OpName::kTruncOut, BuildTruncShlo));
    return out;
  });
}

// go/keep-sorted start
TT_DEFINE_ATEN_UNARY_OUT(OpName::kCeilOut, AtenCeil, BuildCeilShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kConjPhysicalOut, AtenConjPhysical,
                         BuildConjPhysicalShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kFloorOut, AtenFloor, BuildFloorShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kLiftFreshOut, AtenLiftFresh,
                         BuildLiftFreshShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kNotOut, AtenNot, BuildNotShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kSgnOut, AtenSgn, BuildSgnShlo,
                         /*out_dtype=*/std::nullopt);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kSiluOut, AtenSilu, BuildSiluShlo,
                         /*out_dtype=*/std::nullopt);
// go/keep-sorted end

}  // namespace torch_tpu
