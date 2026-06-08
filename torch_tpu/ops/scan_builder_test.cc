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

#include <cstdint>
#include <string>

// We disable clang-format here to force the StableHLO Reference headers to
// be included before the TorchTPU StablehloBuilder headers. This prevents
// a namespace collision where both packages attempt to define 'Tuple'.
// TODO(b/505045588): Remove this workaround after cl/927205178.
// clang-format off
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
// clang-format on

#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/scan_builder.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::HasSubstr;

class ScanBuilderTest : public testing::Test {
 protected:
  ScanBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MlirBuilder& builder() { return module_builder_; }
  mlir::ModuleBuilder& module_builder() { return module_builder_; }
  mlir::OpBuilder& op_builder() { return module_builder_.getOpBuilder(); }
  mlir::MLIRContext* context() { return &context_; }

  mlir::MlirOp CreateBoundedTensorArg(
      mlir::func::FunctionBuilder& function_builder,
      mlir::ArrayRef<int64_t> shape, mlir::ArrayRef<int64_t> bound_dims,
      mlir::Type element_type) {
    // A bounded dynamic tensor requires:
    // 1. `dim_sizes`: The base shape, where dynamic dims are marked with
    //    kDynamic.
    // 2. `bounds_vec`: The memory padding limits. Static dimensions have no
    //    bounds, so they are marked with kDynamic.
    mlir::SmallVector<int64_t> bounds_vec;
    mlir::SmallVector<int64_t> dim_sizes;
    for (int64_t i = 0; i < shape.size(); ++i) {
      if (absl::c_find(bound_dims, i) != bound_dims.end()) {
        bounds_vec.push_back(shape[i]);
        dim_sizes.push_back(mlir::ShapedType::kDynamic);
      } else {
        bounds_vec.push_back(mlir::ShapedType::kDynamic);
        dim_sizes.push_back(shape[i]);
      }
    }

    const auto bounds_attr =
        mlir::DenseI64ArrayAttr::get(context(), bounds_vec);
    const auto encoding =
        mlir::stablehlo::TypeExtensionsAttr::get(context(), bounds_attr);
    const auto type =
        mlir::RankedTensorType::get(dim_sizes, element_type, encoding);

    for (const int64_t bound_dim : bound_dims) {
      EXPECT_TRUE(type.isDynamicDim(bound_dim));
    }

    return mlir::func::Argument(function_builder, type);
  }

  // Body builder that adds the slice to the carry (cumulative sum).
  ScanBodyBuilder CreateAddBodyBuilder() {
    return [](mlir::OpBuilder& op_builder, mlir::Location loc,
              mlir::Value slice, mlir::Value index, mlir::ValueRange carries)
               -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
      return llvm::SmallVector<mlir::Value>{
          mlir::stablehlo::AddOp::create(op_builder, loc, slice, carries[0])
              .getResult()};
    };
  }

  void EvaluateAndVerifyOutputs(mlir::ModuleOp module,
                                llvm::ArrayRef<int32_t> expected_values) {
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), 1);

    const mlir::DenseElementsAttr res_attr = (*eval_result)[0];
    const llvm::SmallVector<int32_t> res_vals(res_attr.getValues<int32_t>());
    EXPECT_THAT(res_vals, testing::ElementsAreArray(expected_values));
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST_F(ScanBuilderTest, InvalidDimension) {
  // Input tensor of shape [4, 5]. Scan along an invalid dimension 2.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator. For cumulative sum, the output shape matches the input.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry tensor. The shape does not matter in this case, since an error will
  // be returned.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      BuildScanShlo(builder(), input, /*dim=*/2, {carry_init}, {output_init},
                    CreateAddBodyBuilder());
  EXPECT_THAT(result, StatusIs(error::kOutOfRange,
                               HasSubstr("dimension out of range")));
}

TEST_F(ScanBuilderTest, EmptyTensor) {
  // Input tensor of shape [0, 5], empty along the scan dimension.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Output accumulator. For cumulative sum, the output shape matches the input.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Carry tensor. The scan loop extracts a slice of the input along the
  // scan dimension (size 1) at each iteration. Therefore, the carry
  // initialization must match this sliced shape.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(builder(), input, /*dim=*/0, {carry_init}, {output_init},
                    CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  // For an empty scan dimension, the loop is skipped, so the returned
  // value matches the initial output accumulator.
  EXPECT_EQ(result[0].getValue(), output_init.getValue());
}

TEST_F(ScanBuilderTest, StaticShape1D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 1D tensor [1, 2, 3, 4] for input.
  const mlir::RankedTensorType type = mlir::RankedTensorType::get({4}, i32);
  const mlir::DenseElementsAttr input_attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [4], initialized to zeros.
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});
  // Carry initialized with shape [1] (slice along dim 0).
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {1});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/0, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 6, 10});
}

TEST_F(ScanBuilderTest, StaticShape2D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We will scan along dim 0.
  const mlir::RankedTensorType type = mlir::RankedTensorType::get({2, 2}, i32);
  const mlir::DenseElementsAttr input_attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  // Carry initialized with shape [1, 2] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/0, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 2, 4, 6});
}

TEST_F(ScanBuilderTest, StaticShape2D_ScanDim1) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We will scan along dim 1.
  const mlir::RankedTensorType type = mlir::RankedTensorType::get({2, 2}, i32);
  const mlir::DenseElementsAttr input_attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  // Carry initialized with shape [2, 1] (slice along dim 1).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {2, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/1, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 3, 7});
}

TEST_F(ScanBuilderTest, StaticShape3D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 3D tensor [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] for input. We scan
  // along dim 2.
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get({2, 2, 2}, i32);
  const mlir::DenseElementsAttr input_attr = mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef<int32_t>{1, 2, 3, 4, 5, 6, 7, 8});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 2});
  // Carry initialized with shape [2, 2, 1] (slice along dim 2).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/2, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 3, 7, 5, 11, 7, 15});
}

TEST_F(ScanBuilderTest, BodyBuilderError) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that always returns an error to simulate a failure
  // during the construction of the loop body.
  auto error_body_builder = [](mlir::OpBuilder& op_builder, mlir::Location loc,
                               mlir::Value slice, mlir::Value index,
                               mlir::ValueRange carries)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    return TT_ERROR(error::kInternal) << "Simulated body builder error";
  };

  // Verify that the error from the body builder is correctly propagated
  // up by BuildScanShlo.
  const absl::StatusOr<DynamicMlirOpResults> result =
      BuildScanShlo(builder(), input, /*dim=*/0, {carry_init}, {output_init},
                    error_body_builder);
  EXPECT_THAT(result, StatusIs(error::kInternal,
                               HasSubstr("Simulated body builder error")));
}

TEST_F(ScanBuilderTest, MismatchedInits) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0). We provide
  // two carry inits, but only one output init, causing a mismatch.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      BuildScanShlo(builder(), input, /*dim=*/0, {carry_init, carry_init},
                    {output_init}, CreateAddBodyBuilder());
  EXPECT_THAT(
      result,
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("expected the number of carry inits (2) and the number of "
                    "output inits (1) to match")));
}

TEST_F(ScanBuilderTest, MismatchedCarries) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that returns two carries instead of the expected one.
  auto wrong_carries_builder = [](mlir::OpBuilder& op_builder,
                                  mlir::Location loc, mlir::Value slice,
                                  mlir::Value index, mlir::ValueRange carries)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    return llvm::SmallVector<mlir::Value>{carries[0], carries[0]};
  };

  const absl::StatusOr<DynamicMlirOpResults> result =
      BuildScanShlo(builder(), input, /*dim=*/0, {carry_init}, {output_init},
                    wrong_carries_builder);
  EXPECT_THAT(result, StatusIs(error::kInvalidArgument,
                               HasSubstr("expected 1 new carries, got 2")));
}

TEST_F(ScanBuilderTest, DynamicShapeScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensor of bounded shape [<=4, 5], where dimension 0 is dynamic.
  mlir::MlirOp input_arg =
      CreateBoundedTensorArg(function_builder, {4, 5}, {0}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 4).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input =
      mlir::stablehlo::SetDimensionSize(input_arg, size, /*dim=*/0);

  // Output accumulator of static shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/0, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(mlir::isa<mlir::stablehlo::WhileOp>(
      result[0].getValue().getDefiningOp()));
}

TEST_F(ScanBuilderTest, DynamicShapeNonScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensor of bounded shape [4, <=5], where dimension 1 is dynamic.
  mlir::MlirOp input_arg =
      CreateBoundedTensorArg(function_builder, {4, 5}, {1}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 5).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input =
      mlir::stablehlo::SetDimensionSize(input_arg, size, /*dim=*/1);

  // Output accumulator of static shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TF_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/0, {carry_init},
                    {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(mlir::isa<mlir::stablehlo::WhileOp>(
      result[0].getValue().getDefiningOp()));
}

}  // namespace
}  // namespace torch_tpu
