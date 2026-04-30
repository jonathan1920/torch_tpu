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

#include "torch_tpu/ops/view_decomposition/slice_primitive.h"

#include <utility>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutSlice, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  SlicePrimitive slice = {.slice_dims = {}};
  auto modified = UpdateLayout(layout, slice);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutSlice, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  SlicePrimitive slice = {
      .slice_dims = {{.start_index = 0, .limit_index = 6, .stride = 1},
                     {.start_index = 0, .limit_index = 4, .stride = 1},
                     {.start_index = 0, .limit_index = 2, .stride = 1}}};
  auto modified = UpdateLayout(layout, slice);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutSlice, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  SlicePrimitive slice = {
      .slice_dims = {{.start_index = 1, .limit_index = 6, .stride = 2},
                     {.start_index = 1, .limit_index = 3, .stride = 2},
                     {.start_index = 1, .limit_index = 2, .stride = 1}}};
  auto modified = UpdateLayout(layout, slice);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 3, .stride = 18},
                       {.size = 1, .stride = 4},
                       {.size = 1, .stride = 1}},
      .storage_offset = 12,
  };
  EXPECT_EQ(layout, expected);
}

TEST(MergeSlice, ValidMerge) {
  Dimensions contiguous_base_shape = {6, 4, 2};
  SlicePrimitive current = {
      .slice_dims = {{.start_index = 1, .limit_index = 6, .stride = 1},
                     {.start_index = 1, .limit_index = 4, .stride = 1},
                     {.start_index = 1, .limit_index = 2, .stride = 1}}};
  SlicePrimitive to_merge = {
      .slice_dims = {{.start_index = 1, .limit_index = 4, .stride = 2},
                     {.start_index = 1, .limit_index = 2, .stride = 2},
                     {.start_index = 0, .limit_index = 2, .stride = 2}}};
  auto expected_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  EXPECT_EQ(UpdateLayout(expected_layout, current).status(), absl::OkStatus());
  EXPECT_EQ(UpdateLayout(expected_layout, to_merge).status(), absl::OkStatus());

  auto merged = Merge(std::move(current), std::move(to_merge));
  EXPECT_EQ(merged.status(), absl::OkStatus());

  auto actual_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  EXPECT_EQ(UpdateLayout(actual_layout, merged.value()).status(),
            absl::OkStatus());

  EXPECT_EQ(actual_layout, expected_layout);
}

}  // namespace
}  // namespace torch_tpu
