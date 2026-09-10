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

#include "csrc/eager/materialization_heuristics.h"

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& /*builder*/, absl::Span<mlir::MlirOp> /*inputs*/) {
  return DynamicMlirOpResults{};
}

TEST(MaterializeHeuristicsTest, IsFastRuntimeConvolutionCandidate) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Case 1: Non-convolution op should return false.
  {
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> add_refs,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
    ASSERT_TRUE(add_refs[0].deferred_op() != nullptr);
    EXPECT_FALSE(IsFastRuntimeConvolutionCandidate(*add_refs[0].deferred_op()));
  }

  // Case 2: Qualifying 2D convolution (N=16, C=64, H=56, W=56, out_c=128)
  // Large compute (approx 411M MACs), W % 128 != 0, batch > 1, NCHW.
  {
    Shape input_shape(Dimensions{16, 64, 56, 56}, mlir::ElementType::F32);
    Shape weight_shape(Dimensions{128, 64, 3, 3}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> input_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {input_shape}));
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> weight_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {weight_shape}));
    Shape out_shape(Dimensions{16, 128, 56, 56}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> conv_refs,
        DeviceBufferList::CreateDeferred(
            OpName::kConvolution, DummyBuilder, {input_refs[0], weight_refs[0]},
            OpParamCacheKeys::Empty(), {out_shape}));
    ASSERT_TRUE(conv_refs[0].deferred_op() != nullptr);
    EXPECT_TRUE(IsFastRuntimeConvolutionCandidate(*conv_refs[0].deferred_op()));
  }

  // Case 3: Small batch (B=1) should return false.
  {
    Shape input_shape(Dimensions{1, 64, 56, 56}, mlir::ElementType::F32);
    Shape weight_shape(Dimensions{128, 64, 3, 3}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> input_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {input_shape}));
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> weight_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {weight_shape}));
    Shape out_shape(Dimensions{1, 128, 56, 56}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> conv_refs,
        DeviceBufferList::CreateDeferred(
            OpName::kConvolution, DummyBuilder, {input_refs[0], weight_refs[0]},
            OpParamCacheKeys::Empty(), {out_shape}));
    ASSERT_TRUE(conv_refs[0].deferred_op() != nullptr);
    EXPECT_FALSE(
        IsFastRuntimeConvolutionCandidate(*conv_refs[0].deferred_op()));
  }

  // Case 4: Channels-last layout should return false.
  {
    Shape input_shape(Dimensions{16, 64, 56, 56}, mlir::ElementType::F32);
    input_shape.set_layout(CustomLayout{.minor_to_major = {1, 3, 2, 0}});
    Shape weight_shape(Dimensions{128, 64, 3, 3}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> input_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {input_shape}));
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> weight_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {weight_shape}));
    Shape out_shape(Dimensions{16, 128, 56, 56}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> conv_refs,
        DeviceBufferList::CreateDeferred(
            OpName::kConvolution, DummyBuilder, {input_refs[0], weight_refs[0]},
            OpParamCacheKeys::Empty(), {out_shape}));
    ASSERT_TRUE(conv_refs[0].deferred_op() != nullptr);
    EXPECT_FALSE(
        IsFastRuntimeConvolutionCandidate(*conv_refs[0].deferred_op()));
  }

  // Case 5: Vector-aligned width (W % 128 == 0, e.g. W=128) should return
  // false.
  {
    Shape input_shape(Dimensions{16, 64, 56, 128}, mlir::ElementType::F32);
    Shape weight_shape(Dimensions{128, 64, 3, 3}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> input_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {input_shape}));
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> weight_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {weight_shape}));
    Shape out_shape(Dimensions{16, 128, 56, 128}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> conv_refs,
        DeviceBufferList::CreateDeferred(
            OpName::kConvolution, DummyBuilder, {input_refs[0], weight_refs[0]},
            OpParamCacheKeys::Empty(), {out_shape}));
    ASSERT_TRUE(conv_refs[0].deferred_op() != nullptr);
    EXPECT_FALSE(
        IsFastRuntimeConvolutionCandidate(*conv_refs[0].deferred_op()));
  }

  // Case 6: Compute volume below threshold (< 100M MACs) should return false.
  {
    Shape input_shape(Dimensions{16, 4, 14, 14}, mlir::ElementType::F32);
    Shape weight_shape(Dimensions{4, 4, 3, 3}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> input_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {input_shape}));
    TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> weight_refs,
                            DeviceBufferList::CreateDeferred(
                                OpName::kEmpty, DummyBuilder, {},
                                OpParamCacheKeys::Empty(), {weight_shape}));
    Shape out_shape(Dimensions{16, 4, 14, 14}, mlir::ElementType::F32);
    TT_ASSERT_OK_AND_ASSIGN(
        std::vector<DeviceBufferRef> conv_refs,
        DeviceBufferList::CreateDeferred(
            OpName::kConvolution, DummyBuilder, {input_refs[0], weight_refs[0]},
            OpParamCacheKeys::Empty(), {out_shape}));
    ASSERT_TRUE(conv_refs[0].deferred_op() != nullptr);
    EXPECT_FALSE(
        IsFastRuntimeConvolutionCandidate(*conv_refs[0].deferred_op()));
  }
}

}  // namespace
}  // namespace torch_tpu
