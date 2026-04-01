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

#include "torch_tpu/eager/device_buffer.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using testing::SizeIs;

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(SubgraphTest, SubgraphMerging) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two independent deferred nodes.
  auto refs1_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs1_or.ok());
  auto ref1 = refs1_or.value()[0];

  auto refs2_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs2_or.ok());
  auto ref2 = refs2_or.value()[0];

  EXPECT_NE(ref1.device_buffer_list()->subgraph()->Find(),
            ref2.device_buffer_list()->subgraph()->Find());

  // Create a third node that merges the first two.
  auto refs3_or =
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref1, ref2},
                                       OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs3_or.ok());
  auto ref3 = refs3_or.value()[0];

  // All three should now belong to the same representative subgraph.
  auto rep1 = ref1.device_buffer_list()->subgraph()->Find();
  auto rep2 = ref2.device_buffer_list()->subgraph()->Find();
  auto rep3 = ref3.device_buffer_list()->subgraph()->Find();

  EXPECT_EQ(rep1, rep2);
  EXPECT_EQ(rep1, rep3);
}

TEST(SubgraphTest, GetLeafNodesInvalidPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node2;
  std::shared_ptr<Subgraph> subgraph;

  {
    auto refs1_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs1_or.ok());
    SharedDeviceBufferList node1 = refs1_or.value()[0].device_buffer_list();

    auto refs2_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs2_or.ok());
    node2 = refs2_or.value()[0].device_buffer_list();

    subgraph = node1->subgraph()->Find();
    Subgraph::Merge(subgraph, node2->subgraph()->Find());
    subgraph = subgraph->Find();
    // node1 and node2 are now in the same subgraph queue.
  }
  // node1 is now out of scope and should be invalid in the queue.

  auto leaves = subgraph->GetLeafNodes();
  // node1 was invalid at the front, so it should have been popped.
  // node2 is valid, so it should be returned as a leaf.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node2);
}

TEST(SubgraphTest, GetLeafNodesStopPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node1;
  SharedDeviceBufferList node3;
  std::shared_ptr<Subgraph> subgraph;

  {
    auto refs1_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    node1 = refs1_or.value()[0].device_buffer_list();

    auto refs2_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    SharedDeviceBufferList node2 = refs2_or.value()[0].device_buffer_list();

    auto refs3_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    node3 = refs3_or.value()[0].device_buffer_list();

    subgraph = node1->subgraph()->Find();
    Subgraph::Merge(subgraph, node2->subgraph()->Find());
    Subgraph::Merge(subgraph, node3->subgraph()->Find());
    subgraph = subgraph->Find();
    // node1, node2, node3 are now in the same subgraph queue.
  }
  // node2 is now out of scope and should be invalid in the queue.
  // node1 and node3 are still valid.

  auto leaves = subgraph->GetLeafNodes();
  // node1 is valid at the front, so it should STOP popping there.
  // node2 (invalid) and node3 (valid) should remain in the queue.
  // Both node1 and node3 should be returned as leaves if they have no child
  // ops.
  EXPECT_THAT(leaves, SizeIs(2));
  EXPECT_EQ(leaves[0], node1);
  EXPECT_EQ(leaves[1], node3);

  // We can verify this by checking that node2 is still in the queue if we
  // could, but we can't easily inspect the private queue_. However, we can
  // check if it gets popped on the NEXT call if node1 is gone.
  leaves.clear();  // Clear references held by leaves.
  EXPECT_EQ(node1.use_count(), 1);
  node1.reset();
  leaves = subgraph->GetLeafNodes();
  // Now node2 is at the front, it should be popped. node3 is next and returned.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node3);
}

TEST(SubgraphTest, DeferredOpSubgraphDereference) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref = refs_or.value()[0];

  const DeferredOp* op = ref.deferred_op();
  ASSERT_NE(op, nullptr);

  // Dereference to the necessary subgraph object via the
  // subgraph pointer in the deferred op.
  EXPECT_EQ(op->subgraph(), ref.device_buffer_list()->subgraph());
}

TEST(SubgraphTest, MergeAll) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two unrelated deferred nodes.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];
  const auto* ref_a_deferred_op = ref_a.deferred_op();
  ASSERT_NE(ref_a_deferred_op, nullptr);

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];
  const auto* ref_b_deferred_op = ref_b.deferred_op();
  ASSERT_NE(ref_b_deferred_op, nullptr);

  // The subgraphs should be distinct; they haven't been merged.
  EXPECT_NE(ref_a_deferred_op->subgraph()->Find(),
            ref_b_deferred_op->subgraph()->Find());

  // Merge all subgraphs.
  auto merged_subgraph = SubgraphRegistry::GetInstance().MergeAll();

  // Both nodes should now be in this merged subgraph.
  EXPECT_EQ(ref_a_deferred_op->subgraph()->Find(), merged_subgraph);
  EXPECT_EQ(ref_b_deferred_op->subgraph()->Find(), merged_subgraph);
}

}  // namespace
}  // namespace torch_tpu
