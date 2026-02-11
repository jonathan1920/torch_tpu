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

#include "torch_tpu/ops/view_decomposition/decomposition.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/absl_test_shim.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

// Validates that applying the view sequence to the contiguous base shape
// produces the desired view layout. Returns an error if it does not.
absl::Status ValidateViewSequence(
    absl::Span<const ViewPrimitive> view_sequence,
    absl::Span<const int64_t> contiguous_base_shape,
    const StridedLayout& view_layout) {
  auto layout = MakeContiguousBaseLayout(contiguous_base_shape);
  for (const auto& primitive : view_sequence) {
    TT_RETURN_IF_ERROR(UpdateLayout(layout, primitive)).SetPrepend()
        << "Validation failed:\n"
        << "contiguous_base_shape: " << ToString(contiguous_base_shape)
        << "\nview_sequence: " << ToString(view_sequence)
        << "\nvalidation error: ";
  }

  TT_RET_CHECK(layout == view_layout, error::kInvalidArgument)
      << "Validation failed:\n"
      << "contiguous_base_shape: " << ToString(contiguous_base_shape)
      << "\nview_sequence: " << ToString(view_sequence)
      << "\nvalidation error: view_sequence does not produce the desired view "
         "layout. Expected: "
      << view_layout << " but got: " << layout;
  return absl::OkStatus();
}

void DecompositionTest(
    absl::Span<const int64_t> contiguous_base_shape,
    const StridedLayout& view_layout,
    const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32,
    const mlir::ElementType view_dtype = mlir::ElementType::F32,
    bool is_conj = false) {
  absl::StatusOr<ViewSequence> sequence =
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype, is_conj);

  // Un-simplified sequence should be valid.
  EXPECT_TRUE(sequence.ok());
  TT_EXPECT_OK(ValidateViewSequence(sequence.value(), contiguous_base_shape,
                                    view_layout));

  // Sequence should be simplifiable.
  TT_EXPECT_OK(Simplify(sequence.value(), contiguous_base_shape));

  // Simplified sequence should be also valid.
  TT_EXPECT_OK(ValidateViewSequence(sequence.value(), contiguous_base_shape,
                                    view_layout));
}

TEST(DecomposeIntoViewSequence, ScalarNoOp) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, TensorNoOp) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeScalarToTensor) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 1}, {.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeTensorToScalar) {
  Dimensions contiguous_base_shape = {1, 1};
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, FlattenTensor) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 24, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, FlattenAndBroadcastTensor) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 999, .stride = 0},
                       {.size = 24, .stride = 1},
                       {.size = 1, .stride = 999}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeTensorToTensor) {
  Dimensions contiguous_base_shape = {27, 2};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9}, {.size = 9, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, UnsqueezeTensor) {
  Dimensions contiguous_base_shape = {27, 2};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 999},
                       {.size = 27, .stride = 2},
                       {.size = 1, .stride = 999},
                       {.size = 2, .stride = 1},
                       {.size = 1, .stride = 999}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, PermuteTensor) {
  Dimensions contiguous_base_shape = {1, 2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 24},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1},
                       {.size = 2, .stride = 12}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ExpandTensor) {
  Dimensions contiguous_base_shape = {1024, 1, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 128},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ExpandSingleElementTensor) {
  Dimensions contiguous_base_shape = {128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 999, .stride = 0}, {.size = 1, .stride = 1}},
      .storage_offset = 64};
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastScalar) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 0},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 0}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastTensor) {
  Dimensions contiguous_base_shape = {1024, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 128},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastAndPermuteTensor) {
  Dimensions contiguous_base_shape = {1024, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 128, .stride = 1},
                       {.size = 512, .stride = 0},
                       {.size = 1024, .stride = 128}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorLow) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  // Equivalent to torch.ones(2, 3, 4)[1:, 1:, 1:]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 12},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 1}},
      .storage_offset = 9,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorHigh) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  // Equivalent to torch.ones(2, 3, 4)[:1, :2, :3]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 12},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorStridedNonContiguous) {
  Dimensions contiguous_base_shape = {7};
  // Equivalent to torch.ones(7)[1:].view(2, 3)[:, ::2]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 2, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorStridedContiguous) {
  Dimensions contiguous_base_shape = {2, 4, 4};
  // Equivalent to torch.ones(2, 4, 4)[::2, ::2, ::2]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 32},
                       {.size = 2, .stride = 8},
                       {.size = 2, .stride = 2}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, PaddingRequired) {
  Dimensions contiguous_base_shape = {5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ScalarViewAsReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorViewAsReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealHalf) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarImagHalf) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealHalf) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorImagHalf) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealToSmallerReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI16;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToSmallerReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 48},
                       {.size = 3, .stride = 16},
                       {.size = 4, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI16;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealToEqualSizedReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::I8;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToEqualSizedReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::I8;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToLargerReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI16;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI64;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, InvalidScalarRealToLargerReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI16;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI64;
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 8 bytes of data, but the base tensor "
                         "only has 2 bytes")));
}

TEST(DecomposeIntoViewSequence, TensorViewAsComplex) {
  Dimensions contiguous_base_shape = {2, 3, 4, 2};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, InvalidScalarViewAsComplex) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 8 bytes of data, but the base tensor "
                         "only has 4 bytes")));
}

TEST(DecomposeIntoViewSequence, ScalarCDoubleToCFloat) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorCDoubleToCFloat) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, InvalidScalarCFloatToCDouble) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF64;
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 16 bytes of data, but the base tensor "
                         "only has 8 bytes")));
}

TEST(DecomposeIntoViewSequence, TensorCFloatToCDouble) {
  Dimensions contiguous_base_shape = {2, 3, 4, 2};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF64;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, MultiStepAsStridedAndBitcast) {
  Dimensions contiguous_base_shape = {54};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 999, .stride = 0},
                       {.size = 2, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, OverlappingMinorDimension) {
  Dimensions contiguous_base_shape = {55};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 999, .stride = 0},
                       {.size = 3, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, OverlappingMajorDimension) {
  Dimensions contiguous_base_shape = {44};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 7},
                       {.size = 999, .stride = 0},
                       {.size = 2, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, OverlappingMajorAndMinorDimension) {
  Dimensions contiguous_base_shape = {45};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 7},
                       {.size = 999, .stride = 0},
                       {.size = 3, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, UnsupportedZeroSizeView) {
  Dimensions contiguous_base_shape = {54};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 0, .stride = 1},
                       {.size = 9, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, mlir::ElementType::F32,
                                view_layout, mlir::ElementType::F32)
          .status(),
      StatusIs(
          error::kUnimplemented,
          HasSubstr("view decomposition does not support zero-sized tensors")));
}

TEST(DecomposeIntoViewSequence, InvalidNegativeStorageOffset) {
  Dimensions contiguous_base_shape = {54};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9}, {.size = 9, .stride = 1}},
      .storage_offset = -1,
  };
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, mlir::ElementType::F32,
                                view_layout, mlir::ElementType::F32)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("storage_offset -1 is negative")));
}

TEST(DecomposeIntoViewSequence, InvalidTooManyElements) {
  Dimensions contiguous_base_shape = {54};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 5, .stride = 11}, {.size = 11, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, mlir::ElementType::F32,
                                view_layout, mlir::ElementType::F32)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 220 bytes of data, but the base tensor "
                         "only has 216 bytes")));
}

TEST(DecomposeIntoViewSequence, InvalidOutOfBounds) {
  Dimensions contiguous_base_shape = {54};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9}, {.size = 9, .stride = 1}},
      .storage_offset = 1,
  };
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, mlir::ElementType::F32,
                                view_layout, mlir::ElementType::F32)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 220 bytes of data, but the base tensor "
                         "only has 216 bytes")));
}

TEST(DecomposeIntoViewSequence, InvalidOutOfBoundsOverlapping) {
  Dimensions contiguous_base_shape = {54};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9}, {.size = 10, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_THAT(
      DecomposeIntoViewSequence(contiguous_base_shape, mlir::ElementType::F32,
                                view_layout, mlir::ElementType::F32)
          .status(),
      StatusIs(error::kInvalidArgument,
               HasSubstr("view requires 220 bytes of data, but the base tensor "
                         "only has 216 bytes")));
}

TEST(DecomposeIntoViewSequence, Conjugate_ComplexBase_ComplexView) {
  Dimensions contiguous_base_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  DecompositionTest(contiguous_base_shape, view_layout,
                    mlir::ElementType::COMPLEXF32,
                    mlir::ElementType::COMPLEXF32, /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_ComplexBase_RealView) {
  Dimensions contiguous_base_shape = {2, 3};
  // Complex -> Real (ViewAsReal) implies output shape {2, 3, 2}
  Dimensions view_shape = {2, 3, 2};
  StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  DecompositionTest(contiguous_base_shape, view_layout,
                    mlir::ElementType::COMPLEXF32, mlir::ElementType::F32,
                    /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_RealBase_ComplexView) {
  Dimensions contiguous_base_shape = {2, 3, 2};
  // Real -> Complex (ViewAsComplex) implies output shape {2, 3}
  Dimensions view_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  DecompositionTest(contiguous_base_shape, view_layout, mlir::ElementType::F32,
                    mlir::ElementType::COMPLEXF32, /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_RealBase_RealView) {
  Dimensions contiguous_base_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  DecompositionTest(contiguous_base_shape, view_layout, mlir::ElementType::F32,
                    mlir::ElementType::F32, /*is_conj=*/true);
}

}  // namespace
}  // namespace torch_tpu
