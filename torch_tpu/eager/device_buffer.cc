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

#include "torch_tpu/eager/device_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/UniqueVoidPtr.h"
#include "c10/util/accumulate.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/shape.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/ops/stride/stride_helper.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/inversion.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {

void Subgraph::push(std::weak_ptr<DeviceBufferList> device_buffer) {
  TT_MUTEX_LOCK(lock, mu_);
  queue_.push_back(std::move(device_buffer));
}

std::shared_ptr<Subgraph> Subgraph::Find() {
  std::shared_ptr<Subgraph> root = shared_from_this();
  while (true) {
    std::shared_ptr<Subgraph> next_root;
    {
      TT_MUTEX_LOCK(lock, root->mu_);
      next_root = root->parent_;
    }
    if (!next_root) break;
    root = next_root;
  }

  // Path compression: update parents along the path to the root.
  std::shared_ptr<Subgraph> current = shared_from_this();
  while (current != root) {
    std::shared_ptr<Subgraph> next;
    {
      TT_MUTEX_LOCK(lock, current->mu_);
      next = current->parent_;
      if (next) {
        current->parent_ = root;
      }
    }
    if (!next || next == root) break;
    current = next;
  }

  return root;
}

void Subgraph::Merge(std::shared_ptr<Subgraph> s1,
                     std::shared_ptr<Subgraph> s2) {
  auto r1 = s1->Find();
  auto r2 = s2->Find();
  if (r1 == r2) return;

  std::shared_ptr<Subgraph> first = r1;
  std::shared_ptr<Subgraph> second = r2;
  if (first.get() > second.get()) std::swap(first, second);

  TT_MUTEX_LOCK(lock1, first->mu_);
  TT_MUTEX_LOCK(lock2, second->mu_);

  // Merge queues.
  r1->queue_.insert(r1->queue_.end(),
                    std::make_move_iterator(r2->queue_.begin()),
                    std::make_move_iterator(r2->queue_.end()));
  r2->queue_.clear();
  r2->parent_ = r1;
}

std::vector<SharedDeviceBufferList> Subgraph::GetLeafNodes(
    absl::Span<const SharedDeviceBufferList> device_buffer_lists) {
  TT_MUTEX_LOCK(lock, mu_);

  // Pop all invalid nodes from the front.
  while (!queue_.empty()) {
    std::shared_ptr<DeviceBufferList> node = queue_.front().lock();
    if (!node) {
      queue_.pop_front();
    } else if (!node->deferred_op()) {
      queue_.pop_front();
    } else {
      break;
    }
  }

  std::vector<SharedDeviceBufferList> leaf_nodes;
  absl::flat_hash_set<DeviceBufferList*> targets;
  targets.reserve(device_buffer_lists.size());
  for (const auto& node : device_buffer_lists) {
    targets.insert(node.get());
  }

  for (auto& weak_node : queue_) {
    std::shared_ptr<DeviceBufferList> node = weak_node.lock();
    if (!node || !node->deferred_op()) continue;

    // If we are materializing this subgraph (either materialize_all is true
    // or it's one of the targeted subgraphs), then we should include all
    // leaf nodes.
    if (targets.contains(node.get())) {
      targets.erase(node.get());
      leaf_nodes.push_back(std::move(node));
      continue;
    }

    if (node->deferred_op()->num_child_ops() == 0) {
      leaf_nodes.push_back(std::move(node));
    }
  }
  TT_CHECK_THROW(targets.empty(), error::kInternal)
      << "Some targets not found in subgraph: "
      << absl::StrJoin(targets, ", ",
                       [](std::string* out, const DeviceBufferList* node) {
                         absl::StrAppend(out, absl::StrFormat("%p", node));
                       });

  return leaf_nodes;
}

std::vector<SharedDeviceBufferList> DeferredOpQueue::GetLeafNodes(
    absl::Span<const SharedDeviceBufferList> device_buffer_lists) {
  absl::flat_hash_map<std::shared_ptr<Subgraph>,
                      std::vector<SharedDeviceBufferList>>
      subgraph_targets;
  std::vector<std::shared_ptr<Subgraph>> ordered_roots;

  if (device_buffer_lists.empty()) {
    ABSL_LOG(INFO) << "[DeferredOpQueue::GetLeafNodes] No device buffer lists "
                      "provided.";
    return {};
  }

  for (const auto& node : device_buffer_lists) {
    if (!node->deferred_op()) {
      ABSL_LOG(INFO) << absl::StrFormat(
          "[DeferredOpQueue::GetLeafNodes] Node is not deferred: %p",
          node.get());
    }

    if (node->subgraph()) {
      auto root = node->subgraph()->Find();
      if (!subgraph_targets.contains(root)) {
        subgraph_targets[root] = {};
        ordered_roots.push_back(root);
      }

      subgraph_targets[root].push_back(node);
    }
  }

  std::vector<SharedDeviceBufferList> all_leaf_nodes;
  for (auto& root : ordered_roots) {
    auto leaves = root->GetLeafNodes(subgraph_targets[root]);
    all_leaf_nodes.insert(all_leaf_nodes.end(),
                          std::make_move_iterator(leaves.begin()),
                          std::make_move_iterator(leaves.end()));
  }

  return all_leaf_nodes;
}

DeferredOpQueue& GetDeferredOpQueue() {
  static absl::NoDestructor<DeferredOpQueue> deferred_op_queue;
  return *deferred_op_queue;
}

size_t DeviceBufferList::size_bytes(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  const auto xla_type = ConvertTo<xla::PrimitiveType>(shapes_[index].dtype);
  absl::Span<const int64_t> dimensions = shapes_[index].dimensions;
  if (dimensions.empty()) {
    // Scalars are 1 element, size depends on the element type.
    return xla::ShapeUtil::ByteSizeOfPrimitiveType(xla_type);
  }
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      // Non-scalar tensors with a 0 in any dimension have no data.
      return 0;
    }
  }
  return xla::ShapeUtil::ByteSizeOf(
      xla::ShapeUtil::MakeShape(xla_type, dimensions));
}

absl::StatusOr<size_t> DeviceBufferList::pjrt_buffer_size(int64_t index) const {
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer & pjrt_buffer, buffer(index));

  TT_RET_CHECK(!pjrt_buffer.IsDeleted(), error::kFailedPrecondition)
      << "DeviceBufferRef has a PjRtBuffer, but it is deleted";
  auto on_device_size_in_bytes = pjrt_buffer.GetOnDeviceSizeInBytes();
  if (on_device_size_in_bytes.ok()) {
    return on_device_size_in_bytes.value();
  }
  ABSL_VLOG(1)
      << "[DeviceBufferRef::pjrt_buffer_size] Failed to get "
         "PjRtBuffer OnDeviceSizeInBytes: "
      << on_device_size_in_bytes.status()
      << ". Inferring from PjRtBuffer shape and DeviceBufferRef elementtype.";
  xla::Shape physical_buffer_shape_estimate = xla::ShapeUtil::MakeShape(
      ConvertTo<xla::PrimitiveType>(shapes_[index].dtype),
      pjrt_buffer.on_device_shape().dimensions());
  return xla::ShapeUtil::ByteSizeOf(physical_buffer_shape_estimate);
}

namespace {

void DebugMaterializedState(std::ostream& os,
                            const MaterializedBuffers& materialized_buffers) {
  os << "materialized";
  if (materialized_buffers.IsAvailable()) {
    os << ", ready";
  } else {
    os << ", pending";
    return;
  }
  if (!materialized_buffers.size()) {
    os << ", empty";
    return;
  }
  xla::PjRtBuffer* absl_nullable maybe_pjrt_buffer = materialized_buffers[0];
  if (maybe_pjrt_buffer == nullptr) {
    os << ", null";
    return;
  }
  const xla::PjRtBuffer* pjrt_buffer = maybe_pjrt_buffer;
  if (pjrt_buffer->IsDeleted()) {
    os << ", deleted";
    return;
  }
  os << ", on_device_shape: " << pjrt_buffer->on_device_shape().ToString(true);
}

void DebugDeferredOpState(std::ostream& os, const DeferredOp& deferred_op) {
  os << "deferred, op_name: " << deferred_op.op_name();
}

void DebugMonostate(std::ostream& os, absl::Span<const int64_t> dimensions) {
  bool is_zero_size = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_size = true;
      break;
    }
  }
  if (is_zero_size) {
    os << "zero_size";
  } else {
    os << "placeholder";
  }
}

// Logs the data variant state with one of these formats:
//   materialized, pjrt_buffer addr 0x1234567890, on_device_shape: f32[8,16]
//   materialized, pjrt_buffer addr 0x1234567890, deleted
//   deferred, op_name: add
//   zero_size
//   placeholder
void DebugDataState(std::ostream& os,
                    const DeferredOp* absl_nullable deferred_op,
                    const MaterializedBuffers* absl_nullable buffers,
                    absl::Span<const int64_t> dimensions) {
  if (deferred_op != nullptr) {
    DebugDeferredOpState(os, *deferred_op);
  } else if (buffers != nullptr) {
    DebugMaterializedState(os, *buffers);
  } else {
    DebugMonostate(os, dimensions);
  }
}

}  // namespace

std::string DeviceBufferRef::DebugString() const {
  std::stringbuf sb;
  std::ostream os(&sb);
  os << "DeviceBufferRef:"
     << "\n\tAddress of ref: " << this
     << "\n\tAddress of DeviceBufferList: " << device_buffer_list_.get()
     << "\n\tIndex: " << index_
     << "\n\tShape and type: " << ToDTypeName(element_type())
     << ToString(dimensions()) << "\n\tData state: ";
  DebugDataState(os, deferred_op(), device_buffer_list_->materialized_buffers(),
                 dimensions());
  return sb.str();
}

DeviceBufferRefState DeviceBufferList::state(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  // If a DeferredOp would produce only zero-sized tensors, we can avoid ever
  // materializing it.
  // If a DeferredOp produces a mix of zero-sized and non-zero-sized tensors, we
  // only need to execute it if the non-zero-sized tensors are required;
  // this will also produce some zero-sized PjRtBuffers that can be ignored.
  // In either case, we identify zero-sized tensors as kZeroSize (even if a
  // DeferredOp or PjRtBuffer happens to exist for it).
  for (int64_t dim : shapes_[index].dimensions) {
    if (dim == 0) {
      return DeviceBufferRefState::kZeroSize;
    }
  }
  if (std::holds_alternative<DeferredOp>(data_)) {
    return DeviceBufferRefState::kDeferred;
  }
  if (std::holds_alternative<MaterializedBuffers>(data_)) {
    return DeviceBufferRefState::kMaterialized;
  }
  return DeviceBufferRefState::kPlaceholder;
}

absl::Span<const int64_t> DeviceBufferList::dimensions(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dimensions;
}

mlir::ElementType DeviceBufferList::element_type(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dtype;
}

int64_t DeviceBufferList::num_elements(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  // Validated at construction time to not overflow.
  return c10::multiply_integers(shapes_[index].dimensions);
}

absl::StatusOr<xla::PjRtBuffer&> DeviceBufferList::buffer(int64_t index) const {
  if (std::holds_alternative<MaterializedBuffers>(data_)) {
    const auto& buffers = std::get<MaterializedBuffers>(data_);
    TT_RETURN_IF_ERROR(buffers.Await());
    TT_RET_CHECK(index >= 0 && index < buffers.size(), error::kInvalidArgument)
        << "Index " << index << " is out of bounds for buffers of size "
        << buffers.size();
    xla::PjRtBuffer* maybe_buffer = buffers[index];
    TT_RET_CHECK(maybe_buffer, error::kFailedPrecondition)
        << "MaterializedBuffers has no/null PjRtBuffer at index " << index;
    return *maybe_buffer;
  }
  return TT_ERROR(error::kFailedPrecondition)
         << "DeviceBufferList does not have a PjRtBuffer at index " << index
         << " because it is not materialized";
}

// Delegate responsibility for deleting the DeviceBufferRef to the
// c10::DataPtr.
// The DeviceBufferRef* is used as both the "data" and "context" of the
// c10::DataPtr; this mirrors the semantics of a std::unique_ptr.
static void DeleteDeviceBufferRef(void* ctx_ptr) {
  DeviceBufferRef* const ref_ptr = static_cast<DeviceBufferRef*>(ctx_ptr);
  ABSL_VLOG(3) << "[c10::DataPtr deleter] deleting "
               << (ref_ptr ? ref_ptr->DebugString() : "null");
  delete ref_ptr;
}

c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref, const int device_idx) {
  auto* absl_nonnull const raw_ref_ptr =
      new DeviceBufferRef(std::move(buffer_ref));
  return c10::DataPtr(raw_ref_ptr, raw_ref_ptr, DeleteDeviceBufferRef,
                      c10::Device(GetPrivateUse1DeviceType(), device_idx));
}

class TpuAllocator final : public c10::Allocator {
 public:
  TpuAllocator() = default;

  // This class is move-only.
  TpuAllocator(TpuAllocator&& other) = default;
  TpuAllocator& operator=(TpuAllocator&& other) = default;
  TpuAllocator(const TpuAllocator&) = delete;
  TpuAllocator& operator=(const TpuAllocator&) = delete;

  c10::DataPtr allocate(size_t nbytes) override {
    c10::DeviceIndex device_idx = 0;
    if (const auto* device = GetPjRtDevice()) {
      device_idx =
          static_cast<c10::DeviceIndex>(device->local_hardware_id().value());
    }
    // Check that the size_t does not overflow an int64_t.
    // This function is only ever called from PyTorch so safe to throw an
    // exception on failure.
    TT_CHECK_THROW(
        nbytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        error::kResourceExhausted)
        << "allocation size " << nbytes
        << " overflows as a signed 64-bit integer";

    // Allocated DeviceBufferRef will be equivalent to
    // torch.empty(nbytes, dtype=torch.uint8) with
    // fill_uninitialized_memory=True, which fills the buffer with 0xFF bytes.
    TT_ASSIGN_OR_THROW(auto buffer_ref, DeviceBufferList::CreateEmpty(
                                            {static_cast<int64_t>(nbytes)},
                                            mlir::ElementType::UI8));
    return MakeDataPtr(std::move(buffer_ref), device_idx);
  }

  void copy_data(void* dest, const void* src,
                 std::size_t count) const override {
    TT_CHECK_THROW(false, error::kUnimplemented)
        << "TpuAllocator::copy_data is not implemented and should not be "
           "called directly for this conceptual allocator type.";
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return DeleteDeviceBufferRef;
  }
};

c10::Allocator* GetTpuAllocator() {
  static absl::NoDestructor<TpuAllocator> g_tpu_allocator;
  return g_tpu_allocator.get();
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateMaterialized(
    absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer) {
  Dimensions dimensions = CopyIntVector(buffer->on_device_shape().dimensions());
  TT_ASSIGN_OR_RETURN(
      const auto element_type,
      ConvertTo<mlir::ElementType>(buffer->on_device_shape().element_type()));
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer_list = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(buffer), element_type));
  return DeviceBufferRef(std::move(device_buffer_list), 0);
}

absl::StatusOr<std::vector<DeviceBufferRef>> DeviceBufferList::CreateDeferred(
    OpName op_name, MlirOpBuilder op_builder,
    std::vector<DeviceBufferRef> inputs, OpParamCacheKeys op_param_cache_keys,
    std::vector<Shape> output_shapes, OpSplitMode split_mode) {
  // Validate that the output shapes are valid.
  for (const auto& output_shape : output_shapes) {
    TT_RETURN_IF_ERROR(
        ValidateTensorByteSize(output_shape.dimensions, output_shape.dtype));
  }
  int64_t num_outputs = output_shapes.size();

  auto& queue = GetDeferredOpQueue();
  std::shared_ptr<Subgraph> subgraph = nullptr;
  for (const auto& input : inputs) {
    if (input.state() == DeviceBufferRefState::kDeferred) {
      auto input_subgraph = input.device_buffer_list()->subgraph();
      if (!input_subgraph) continue;

      auto input_rep = input_subgraph->Find();
      if (!subgraph) {
        subgraph = input_rep;
      } else if (subgraph != input_rep) {
        queue.MergeSubgraphs(subgraph, input_rep);
        subgraph = subgraph->Find();
      }
    }
  }

  if (!subgraph) {
    subgraph = std::make_shared<Subgraph>();
    queue.RegisterSubgraph(subgraph);
  }

  // Create the DeferredOp.
  auto op = DeferredOp(op_name, std::move(op_builder), std::move(inputs),
                       std::move(op_param_cache_keys), subgraph, split_mode);

  // Wrap the DeferredOp in a DeviceBufferList.
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(op), std::move(output_shapes), subgraph));

  subgraph->push(std::weak_ptr<DeviceBufferList>(device_buffer));

  // Construct one DeviceBufferRef for each output.
  std::vector<DeviceBufferRef> device_buffer_refs;
  device_buffer_refs.reserve(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    device_buffer_refs.push_back(DeviceBufferRef(device_buffer, i));
  }
  return device_buffer_refs;
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateEmpty(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(auto byte_size,
                      ValidateTensorByteSize(dimensions, element_type));
  if (byte_size == 0) {
    return DeviceBufferList::CreateZeroSize(std::move(dimensions),
                                            element_type);
  }

  auto op_builder = [dimensions =
                         CopyIntVector(absl::MakeConstSpan(dimensions)),
                     element_type](mlir::MlirBuilder& builder,
                                   absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "CreateEmpty should not be called with any inputs";
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, element_type, dimensions)};
  };
  Shape output_shape = {.dimensions = std::move(dimensions),
                        .dtype = element_type};
  TT_ASSIGN_OR_RETURN(
      auto results, DeviceBufferList::CreateDeferred(
                        OpName::kEmpty, std::move(op_builder), /*inputs=*/{},
                        /*op_param_cache_keys=*/{}, {std::move(output_shape)}));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateZeroSize(
    Dimensions dimensions, mlir::ElementType element_type) {
  bool is_zero_sized = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_sized = true;
      break;
    }
  }
  TT_RET_CHECK(is_zero_sized, error::kInvalidArgument)
      << "CreateZeroSize requires a zero-sized tensor, but got: "
      << ToString(dimensions);
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(dimensions), element_type));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::MakePlaceholder(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(auto byte_size,
                      ValidateTensorByteSize(dimensions, element_type));
  if (byte_size == 0) {
    return DeviceBufferList::CreateZeroSize(std::move(dimensions),
                                            element_type);
  }
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(dimensions), element_type));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::Status DeviceBufferList::SetAsMaterialized() {
  ABSL_VLOG(1) << "[SetAsMaterialized] Setting as empty materialized";
  auto state_ = state(0);
  if (state_ == DeviceBufferRefState::kMaterialized) {
    return absl::OkStatus();
  }
  data_ = MaterializedBuffers();
  return absl::OkStatus();
}

namespace {

// Verify that the requested `at_shape` that a deferred op expects is valid for
// the given `buffer_shape` that is on device.
//
// For static dimensions the buffer and op shape must match exactly, for bounded
// dimensions, only check upper bounds. We could get the real on-device shape
// from `buffers[i]->logical_dimensions()`, but this is a device syncing
// operation (blocking), so we want to avoid it.
absl::Status ValidateBufferShape(const Shape& at_shape,
                                 const xla::Shape& buffer_shape) {
  // Ranks must match
  absl::Span<const int64_t> buffer_dims = buffer_shape.dimensions();
  TT_RET_CHECK(at_shape.dimensions.size() == buffer_dims.size(),
               error::kInvalidArgument)
      << "unexpected rank for buffer; expected: "
      << ToString(at_shape.dimensions.size())
      << " but got: " << ToString(buffer_dims.size());

  // Static dims must match and dynamic dims must be LTE the upper bound.
  for (int64_t d = 0; d < at_shape.dimensions.size(); ++d) {
    TT_RET_CHECK(at_shape.dimensions[d] == buffer_dims[d] ||
                     (buffer_shape.is_dynamic_dimension(d) &&
                      at_shape.dimensions[d] <= buffer_dims[d]),
                 error::kInvalidArgument)
        << "incompatible buffer shapes at dimension " << d << "; for op shape "
        << ToString(at_shape.dimensions) << " and buffer shape "
        << buffer_shape.ToString();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status DeviceBufferList::SetAsMaterialized(
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
  TT_RET_CHECK(shapes_.size() == buffers.size(), error::kInvalidArgument)
      << "unexpected number of buffers; expected: " << shapes_.size()
      << " but got: " << buffers.size();
  auto materialized_buffers = std::get_if<MaterializedBuffers>(&data_);
  TT_RET_CHECK(materialized_buffers, error::kInvalidArgument)
      << "DeviceBufferList is not in a materialized state";
  TT_RET_CHECK(!materialized_buffers->IsAvailable(), error::kInvalidArgument)
      << "DeviceBufferList was already made available";
  for (size_t i = 0; i < shapes_.size(); ++i) {
    TT_RET_CHECK(!buffers[i]->IsDeleted(), error::kInvalidArgument)
        << "buffer " << i << " is deleted";
    TT_RETURN_IF_ERROR(
        ValidateBufferShape(shapes_[i], buffers[i]->on_device_shape()))
            .SetPrepend()
        << "buffer " << i << ": ";
    TT_ASSIGN_OR_RETURN(
        mlir::ElementType actual_element_type,
        ConvertTo<mlir::ElementType>(buffers[i]->element_type()),
        _.SetOverride() << "buffer " << i
                        << " has an unsupported element type: "
                        << xla::primitive_util::LowercasePrimitiveTypeName(
                               buffers[i]->element_type()));
    TT_RET_CHECK(actual_element_type == shapes_[i].dtype,
                 error::kInvalidArgument)
        << "unexpected element type for buffer " << i
        << "; expected: " << ToDTypeName(shapes_[i].dtype)
        << " but got: " << ToDTypeName(actual_element_type);
  }
  ABSL_VLOG(1) << "[SetAsMaterialized] Setting as materialized";
  return materialized_buffers->SetAsAvailable(std::move(buffers));
}

absl::Status DeviceBufferList::MarkDynamic(int64_t index, int64_t dimension,
                                           int64_t lower_bound,
                                           int64_t upper_bound) {
  TT_RET_CHECK(lower_bound >= 2 && lower_bound <= upper_bound,
               error::kInvalidArgument)
      << "trying to mark dimension " << dimension
      << " as dynamic with invalid bounds [" << lower_bound << ", "
      << upper_bound << "]";
  TT_RET_CHECK(index >= 0 && index < shapes_.size(), error::kOutOfRange)
      << "index " << index << " is out of bounds for DeviceBufferList of size "
      << shapes_.size();
  Shape& shape = shapes_[index];
  TT_RET_CHECK(dimension >= 0 && dimension < shape.dimensions.size(),
               error::kOutOfRange)
      << "dimension " << dimension << " is out of bounds for tensor of rank "
      << shape.dimensions.size();
  TT_RET_CHECK(shape.dimensions[dimension] >= lower_bound &&
                   shape.dimensions[dimension] <= upper_bound,
               error::kOutOfRange)
      << "trying to mark dimension " << dimension << " as dynamic with bounds ["
      << lower_bound << ", " << upper_bound << "], but the dimension size is "
      << shape.dimensions[dimension];
  auto it_find = std::find_if(
      shape.dynamic_dimensions.begin(), shape.dynamic_dimensions.end(),
      [dimension](const BoundedDynamicDimension& dynamic_dimension) {
        return dynamic_dimension.dimension == dimension;
      });
  if (it_find != shape.dynamic_dimensions.end()) {
    it_find->lower_bound = lower_bound;
    it_find->upper_bound = upper_bound;
  } else {
    TT_RET_CHECK(shape.dynamic_dimensions.empty(), error::kInvalidArgument)
        << "only one dynamic dimension is supported per tensor";
    shape.dynamic_dimensions.push_back({.dimension = dimension,
                                        .lower_bound = lower_bound,
                                        .upper_bound = upper_bound});
  }
  return absl::OkStatus();
};

absl::Span<const BoundedDynamicDimension> DeviceBufferList::dynamic_dimensions(
    int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dynamic_dimensions;
}

absl::StatusOr<DeviceBufferRef> DeviceBufferRef::Create(
    SharedDeviceBufferList device_buffer_list, int64_t index) {
  // Gracefully return an error on creation, but crash hard if the bounds check
  // is violated afterwards, as that would indicate this check was bypassed.
  TT_RET_CHECK(index >= 0 && index < device_buffer_list->size(),
               error::kOutOfRange)
      << "index " << index << " is out of bounds for DeviceBufferList of size "
      << device_buffer_list->size();
  return DeviceBufferRef(std::move(device_buffer_list), index);
}

size_t DeviceBufferRef::size_bytes() const {
  return device_buffer_list_->size_bytes(index_);
}

absl::StatusOr<size_t> DeviceBufferRef::pjrt_buffer_size() const {
  return device_buffer_list_->pjrt_buffer_size(index_);
}

DeviceBufferRefState DeviceBufferRef::state() const {
  return device_buffer_list_->state(index_);
}

[[nodiscard]] absl::Span<const int64_t> DeviceBufferRef::dimensions() const {
  return device_buffer_list_->dimensions(index_);
}

[[nodiscard]] int64_t DeviceBufferRef::num_elements() const {
  return device_buffer_list_->num_elements(index_);
}

[[nodiscard]] mlir::ElementType DeviceBufferRef::element_type() const {
  return device_buffer_list_->element_type(index_);
}

[[nodiscard]] const DeferredOp* absl_nullable DeviceBufferRef::deferred_op()
    const {
  return device_buffer_list_->deferred_op();
}

absl::StatusOr<xla::PjRtBuffer&> DeviceBufferRef::buffer() const {
  return device_buffer_list_->buffer(index_);
}

absl::Status DeviceBufferRef::MarkDynamic(int64_t dimension,
                                          int64_t lower_bound,
                                          int64_t upper_bound) const {
  return device_buffer_list()->MarkDynamic(index_, dimension, lower_bound,
                                           upper_bound);
}

absl::Span<const BoundedDynamicDimension> DeviceBufferRef::dynamic_dimensions()
    const {
  return device_buffer_list()->dynamic_dimensions(index_);
}

absl::StatusOr<DeviceBufferRef> GetBaseBufferFromAtTensor(
    const c10::TensorImpl& tensor) {
  // If the user tries to call a torch_tpu kernel with tensors that are
  // on different devices, then we return an error status, which will be
  // propagated to the user as a Python exception (with context).
  // We only need to uphold invariants for tensors torch_tpu constructs.
  TT_RET_CHECK(tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "tensor is expected to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << tensor.device().str();

  // This will trigger a device mismatch in Pytorch, with a good call stack.
  TT_RET_CHECK(tensor.storage().data_ptr(), error::kUnimplemented)
      << "tensor is on the custom PrivateUse1 device. But its storage data_ptr "
         "is null. This is usually caused by FakeTensor being run on TPU ops. "
         "And it is not supported";

  // If the tensor is on the PrivateUse1 device, then we check invariants.
  // Any errors indicate serious bugs and we crash to prevent worsening the
  // application state.
  ABSL_CHECK_EQ(tensor.storage().allocator(), GetTpuAllocator())  // CRASH_OK
      << "tensor is on PrivateUse1 device, but is not allocated by "
         "g_tpu_allocator";

  return GetBaseBufferFromStorage(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> GetBaseBufferFromAtTensor(
    const at::Tensor& tensor) {
  const c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined.";
  return GetBaseBufferFromAtTensor(*tensor.unsafeGetTensorImpl());
}

absl::StatusOr<DeviceBufferRef> GetBaseBufferFromStorage(
    const c10::Storage& storage) {
  // The DeviceBufferRef in the c10::StorageImpl represents the "contiguous
  // base" buffer--this is the shape of the original tensor that was used to
  // construct the tensor, and represents all of the data available to all
  // views that share the same Storage.
  const DeviceBufferRef* base_buffer_ref =
      static_cast<const DeviceBufferRef*>(storage.data_ptr().get_context());
  ABSL_CHECK(base_buffer_ref)  // CRASH_OK
      << "tensor storage has a null DeviceBufferRef via data_ptr context";
  return *base_buffer_ref;
}

namespace {

at::ScalarType GetScalarType(const c10::TensorImpl& tensor) {
  // For some reason, c10::TensorImpl does not have a scalar_type() method;
  // we have to use "dtype()" and convert instead.
  auto type_meta = tensor.dtype();
  return c10::typeMetaToScalarType(type_meta);
}

// Returns the number of bytes that must be present in the storage for the
// tensor to be well-defined.
absl::StatusOr<int64_t> RequiredStorageBytes(const c10::TensorImpl& tensor) {
  if (tensor.numel() < 1) return 0;
  // The number of storage elements required is
  //  storage_offset + 1 + maximum element index.
  // This accounts for overlapping or non-dense views, which can span more or
  // fewer storage elements than there are logical elements in the tensor.
  int64_t required_numel = tensor.storage_offset() + 1;
  for (auto i = 0; i < tensor.dim(); ++i) {
    required_numel += (tensor.size(i) - 1) * tensor.stride(i);
  }
  // Multiply by the size of the scalar type to get the number of bytes.
  auto scalar_type = GetScalarType(tensor);

  TT_ASSIGN_OR_RETURN(const auto tensor_element_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  const int64_t element_bitwidth = TorchEquivalentBitwidth(tensor_element_type);
  if (element_bitwidth >= 8) {
    return required_numel * (element_bitwidth / 8);
  }

  // Round up to the nearest byte, in the case of sub-byte-size types
  return (required_numel * element_bitwidth + 7) / 8;
}

}  // namespace

absl::StatusOr<DeviceBufferRef> GetBufferFromAtTensor(
    const c10::TensorImpl& tensor) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref,
                      GetBaseBufferFromAtTensor(tensor));

  // Get the element types; if they're different, we need to do a bitwise cast.
  auto scalar_type = GetScalarType(tensor);
  TT_ASSIGN_OR_RETURN(const auto tensor_element_type,
                      ConvertTo<mlir::ElementType>(scalar_type));

  if (!tensor.is_conj() && tensor.is_contiguous() &&
      tensor.storage_offset() == 0 &&
      tensor.sizes() == base_buffer_ref.dimensions() &&
      tensor_element_type == base_buffer_ref.element_type()) {
    ABSL_VLOG(1) << "[GetBufferFromAtTensor] Tensor is a contiguous base, "
                    "returning base buffer.";
    // This is the base tensor, no view to apply.
    return base_buffer_ref;
  }

  // If the view is zero-sized, then we don't need a view decomposition; a view
  // of nothing is valid.
  if (tensor.numel() < 1) {
    ABSL_VLOG(1) << "[GetBufferFromAtTensor] Tensor is a zero-sized view, "
                    "returning empty buffer.";
    return DeviceBufferList::CreateEmpty(CopyIntVector(tensor.sizes()),
                                         tensor_element_type);
  }

  // We need to apply a view decomposition.
  // Check to make sure we have enough data to read the tensor.
  TT_ASSIGN_OR_RETURN(const int64_t required_bytes,
                      RequiredStorageBytes(tensor));
  TT_RET_CHECK(required_bytes <= tensor.storage().nbytes(), error::kOutOfRange)
      << "cannot read " << required_bytes << " bytes (" << tensor.numel()
      << " elements of type " << ToString(GetScalarType(tensor))
      << " with an offset of " << tensor.storage_offset()
      << " elements) from a storage buffer with " << tensor.storage().nbytes()
      << " bytes";

  // Compute the steps in the decomposition.
  StridedLayout view_layout = {.storage_offset = tensor.storage_offset()};
  view_layout.strided_dims.reserve(tensor.dim());
  for (auto i = 0; i < tensor.dim(); ++i) {
    view_layout.strided_dims.push_back(
        {.size = tensor.size(i), .stride = tensor.stride(i)});
  }
  TT_ASSIGN_OR_RETURN(
      ViewSequence view_sequence,
      DecomposeIntoViewSequence(base_buffer_ref.dimensions(),
                                base_buffer_ref.element_type(), view_layout,
                                tensor_element_type, tensor.is_conj()));
  TT_RETURN_IF_ERROR(Simplify(view_sequence, base_buffer_ref.dimensions()));
  ABSL_VLOG(1)
      << "[GetBufferFromAtTensor] Decomposed base buffer dtype and shape "
      << ToDTypeName(base_buffer_ref.element_type())
      << ToString(base_buffer_ref.dimensions()) << " into view sequence "
      << ToString(view_sequence) << " to achieve target view layout "
      << view_layout << " (is_conj=" << tensor.is_conj() << ")";

  // Build the values to construct an ephemeral DeferredOp.
  const auto op_name = OpName::kAsStrided;
  ScopedPythonContextCapturer capturer(op_name);

  auto op_builder = [view_sequence = std::move(view_sequence)](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.size() == 1, error::kInternal)
        << "expected 1 input, got " << inputs.size();
    TT_ASSIGN_OR_RETURN(auto output,
                        ViewSequenceShlo(inputs[0], view_sequence));
    return DynamicMlirOpResults{std::move(output)};
  };

  std::vector<DeviceBufferRef> inputs = {std::move(base_buffer_ref)};

  TT_ASSIGN_OR_RETURN(
      OpParamCacheKeys param_keys,
      *OpParamCacheKeys::SetParam("strides", tensor.strides())
           .SetParam("storage_offset", tensor.storage_offset()));

  std::vector<Shape> output_shapes = {
      {.dimensions = CopyIntVector(tensor.sizes()),
       .dtype = tensor_element_type}};

  // Create the deferred op.
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> deferred_refs,
                      DeviceBufferList::CreateDeferred(
                          op_name, std::move(op_builder), std::move(inputs),
                          std::move(param_keys), std::move(output_shapes)));
  ABSL_CHECK_EQ(deferred_refs.size(), 1);  // CRASH_OK
  return std::move(deferred_refs[0]);
}

absl::StatusOr<DeviceBufferRef> GetBufferFromAtTensor(
    const at::Tensor& tensor) {
  // Trying to get the buffer from an undefined tensor is a user error, not
  // a critical invariant violation.
  const c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
  return GetBufferFromAtTensor(*tensor.unsafeGetTensorImpl());
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetBuffersFromAtTensors(
    absl::Span<const at::Tensor> tensors) {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBufferFromAtTensor(tensor));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

c10::Storage MakeStorage(DeviceBufferRef buffer_ref) {
  ABSL_VLOG(1) << "[MakeStorage] Received DeviceBufferRef with dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]"
               << " and dtype: " << ToDTypeName(buffer_ref.element_type());
  const auto size = buffer_ref.size_bytes();
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size,
      MakeDataPtr(std::move(buffer_ref), /*device_idx=*/0), GetTpuAllocator(),
      /*resizable=*/true));
}

at::Tensor MakeTensor(DeviceBufferRef buffer_ref) {
  ABSL_VLOG(1) << "[MakeTensor] Creating new ATen tensor for "
               << buffer_ref.DebugString();

  const auto dtype = ConvertTo<at::ScalarType>(buffer_ref.element_type());
  auto caffe2_type_meta = c10::scalarTypeToTypeMeta(dtype);
  absl::Span<const int64_t> sizes = buffer_ref.dimensions();

  c10::Storage storage = MakeStorage(std::move(buffer_ref));
  at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      caffe2_type_meta));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  tensor.unsafeGetTensorImpl()->set_storage_offset(0);
  ABSL_VLOG(1) << "[MakeTensor] Final ATen tensor created:" << ToString(tensor)
               << "\nscalar_type: " << c10::toString(tensor.scalar_type())
               << "\nsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
               << "\nstrides: "
               << ToString(absl::MakeConstSpan(tensor.strides()))
               << (tensor.is_contiguous() ? " (contiguous)"
                                          : " (non-contiguous)")
               << "\nstorage nbytes: " << tensor.storage().nbytes()
               << "\nstorage offset: " << tensor.storage_offset();

  ABSL_CHECK_EQ(tensor.device().type(),  // CRASH_OK
                GetPrivateUse1DeviceType())
      << "Tensor created does NOT have PrivateUse1 device type.";
  return tensor;
}

namespace {

// Helper function for AssignBufferToAtTensor to switch between two cases:
//   * If the view is not a slice (so that the inversion fully overwrites the
//     original tensor), then we don't care what the original tensor's values
//     were, and the DeferredOp is unary.
//   * If the view is a slice, then some amount of its original data will be
//     retained, so we need to carry a dependency on the original base_buffer,
//     and the DeferredOp is binary.
// Returns the appropriate builder and inputs for the DeferredOp.
std::pair<MlirOpBuilder, std::vector<DeviceBufferRef>>
MakeAsStridedInverseBuilder(InverseViewOperation inverse_view_operation,
                            DeviceBufferRef base_buffer_ref,
                            DeviceBufferRef result_buf) {
  if (inverse_view_operation.stages.size() == 1) {
    ABSL_VLOG(3) << "[AssignBufferToAtTensor] View elements are 1:1 with base, "
                    "creating an inverse DeferredOp that overwrites the base.";
    // Merge the view bitcast with the inverse sequence.
    ViewSequence merged_sequence;
    merged_sequence.reserve(inverse_view_operation.bitcast_view.size() +
                            inverse_view_operation.stages[0].inverse.size());
    for (auto& primitive : inverse_view_operation.bitcast_view) {
      merged_sequence.push_back(std::move(primitive));
    }
    for (auto& inverse_primitive : inverse_view_operation.stages[0].inverse) {
      std::visit(
          [&merged_sequence](auto& prim) {
            merged_sequence.push_back(std::move(prim));
          },
          inverse_primitive);
    }

    auto op_builder = [view_sequence = std::move(merged_sequence)](
                          mlir::MlirBuilder& builder,
                          absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      TT_RET_CHECK(inputs.size() == 1, error::kInternal)
          << "expected 1 input, got " << inputs.size();
      TT_ASSIGN_OR_RETURN(auto output,
                          ViewSequenceShlo(inputs[0], view_sequence));
      return DynamicMlirOpResults{std::move(output)};
    };
    auto inputs = {std::move(result_buf)};
    return {std::move(op_builder), std::move(inputs)};
  }

  // Otherwise, apply the full operation.
  ABSL_VLOG(3) << "[AssignBufferToAtTensor] View is a slice, creating an "
                  "inverse DeferredOp that partially overwrites the base.";
  auto op_builder =
      [inverse_view_operation = std::move(inverse_view_operation)](
          mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.size() == 2, error::kInternal)
        << "expected 2 inputs, got " << inputs.size();
    TT_ASSIGN_OR_RETURN(
        auto output,
        InverseViewOperationShlo(inputs[0], inputs[1], inverse_view_operation));
    return DynamicMlirOpResults{std::move(output)};
  };
  auto inputs = {std::move(base_buffer_ref), std::move(result_buf)};
  return {std::move(op_builder), std::move(inputs)};
}

}  // namespace

absl::Status AssignBufferToAtTensor(DeviceBufferRef result_buf,
                                    const at::Tensor& tensor) {
  ABSL_VLOG(1)
      << "[AssignBufferToAtTensor] Assigning DeviceBufferRef to existing "
         "tensor\n"
      << result_buf.DebugString() << "\n"
      << "\nTensor:" << ToString(tensor)
      << "\n\tscalar_type: " << c10::toString(tensor.scalar_type())
      << "\n\tsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
      << "\n\tstrides: " << ToString(absl::MakeConstSpan(tensor.strides()))
      << (tensor.is_contiguous() ? " (contiguous)" : " (non-contiguous)")
      << "\n\tstorage nbytes: " << tensor.storage().nbytes()
      << "\n\tstorage offset: " << tensor.storage_offset();

  // It's a critical failure if we are trying to write a buffer to a tensor
  // with the wrong amount or type of data; we need a 1:1 byte mapping from
  // the logical tensor elements to the buffer elements.
  // However, we do allow at::Tensors to be backed by DeviceBufferRefs of
  // view-equivalent shapes, for example, a (10, 10) tensor may be backed by
  // a DeviceBufferRef of shape (100,), or vice-versa.
  // Shape alignments will be handled by view decomposition logic on extraction
  // during GetBufferFromAtTensor later, if and only if required.
  ABSL_CHECK_EQ(result_buf.num_elements(), tensor.numel());  // CRASH_OK
  TT_ASSIGN_OR_RETURN(const auto tensor_element_type,
                      ConvertTo<mlir::ElementType>(tensor.scalar_type()));
  ABSL_CHECK_EQ(result_buf.element_type(), tensor_element_type);  // CRASH_OK

  if (result_buf.state() == DeviceBufferRefState::kZeroSize) {
    // Writing 0 elements is a no-op.
    return absl::OkStatus();
  }

  // We must skip this optimization for Conj tensors because their storage
  // contains the *original* values, not the conjugated values. Direct storage
  // assignment would lose the conjugation bit, resulting in incorrect values.
  // By falling through to ComputeInverseViewOperation, we ensure the proper
  // view inversion (which includes conjugation) is applied during the
  // write-back.
  if (tensor.is_contiguous() && !tensor.is_conj() &&
      tensor.storage_offset() == 0 &&
      tensor.storage().nbytes() == result_buf.size_bytes()) {
    // The result_buf can be used directly as the new storage buffer; even if
    // it is a different shape, any contiguous view can be the base.
    //
    // Note: this means it's possible for an operation like
    // ```
    //   a = torch.zeros(100)
    //   b = a.view(10, 10)
    //   b.add_(1.0)
    // ```
    // to change the shape of the base DeviceBufferRef from one contiguous shape
    // to another equivalent shape, in this case (100,) -> (10, 10). In some
    // sense, this changes from `b` being a view of `a`, to `a` being a view
    // of `b`, in a "last write wins" ordering.
    ABSL_VLOG(2) << "[AssignBufferToAtTensor] Directly assigning contiguous "
                    "base buffer of "
                 << result_buf.size_bytes() << " bytes.";

    // Change the c10::DataPtr in storage to use the new DeviceBufferRef.
    // This ensures that all active views on this same contiguous base tensor
    // will be updated by the write.
    tensor.storage().set_data_ptr(
        MakeDataPtr(std::move(result_buf), tensor.device().index()));
  } else {
    TT_RET_CHECK(!IsOverlapping(tensor.sizes(), tensor.strides()),
                 error::kFailedPrecondition)
        << "inplace writes to overlapping views are undefined behavior and are "
           "not supported.\n"
        << "Because multiple logical tensor indices point to the same buffer "
           "elements, writes from multiple indices may overwrite each other.\n"
        << "Please use clone() or contiguous() to copy the tensor before "
           "writing";

    ABSL_VLOG(2)
        << "[AssignBufferToAtTensor] Assigning to non-contiguous base.";

    TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref,
                        GetBaseBufferFromAtTensor(tensor));

    // Compute the operations needed to invert the view sequence.
    StridedLayout view_layout{.storage_offset = tensor.storage_offset()};
    view_layout.strided_dims.reserve(tensor.dim());
    for (auto i = 0; i < tensor.dim(); ++i) {
      view_layout.strided_dims.push_back(
          {.size = tensor.size(i), .stride = tensor.stride(i)});
    }
    TT_ASSIGN_OR_RETURN(
        InverseViewOperation inverse_view_operation,
        ComputeInverseViewOperation(base_buffer_ref.dimensions(),
                                    base_buffer_ref.element_type(), view_layout,
                                    tensor_element_type, tensor.is_conj()));

    // Create a new DeferredOp representing the updated base value
    // Build the values to construct an ephemeral DeferredOp.
    const auto op_name = OpName::kAsStridedInverse;
    ScopedPythonContextCapturer capturer(op_name);

    std::vector<Shape> output_shapes = {inverse_view_operation.final_shape};

    auto [op_builder, inputs] = MakeAsStridedInverseBuilder(
        std::move(inverse_view_operation), std::move(base_buffer_ref),
        std::move(result_buf));

    TT_ASSIGN_OR_RETURN(
        OpParamCacheKeys param_keys,
        *OpParamCacheKeys::SetParam("strides", tensor.strides())
             .SetParam("storage_offset", tensor.storage_offset()));

    // Create the deferred DeviceBufferRef.
    TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> deferred_refs,
                        DeviceBufferList::CreateDeferred(
                            op_name, std::move(op_builder), std::move(inputs),
                            std::move(param_keys), std::move(output_shapes)));
    ABSL_CHECK_EQ(deferred_refs.size(), 1);  // CRASH_OK

    // Assign the new deferred DeviceBufferRef to the c10::DataPtr
    // All reads that happened before the write will be unaffected, but reads
    // to all extant views will reflect the new value on later accesses.
    tensor.storage().set_data_ptr(
        MakeDataPtr(std::move(deferred_refs[0]), tensor.device().index()));
  }

  // Bump the version to tell PyTorch how to handle functionalization
  tensor.unsafeGetTensorImpl()->bump_version();

  ABSL_VLOG(1) << "[AssignBufferToAtTensor] Final tensor after assignment:"
               << ToString(tensor)
               << "\nscalar_type: " << c10::toString(tensor.scalar_type())
               << "\nsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
               << "\nstrides: "
               << ToString(absl::MakeConstSpan(tensor.strides()))
               << (tensor.is_contiguous() ? " (contiguous)"
                                          : " (non-contiguous)")
               << "\nstorage nbytes: " << tensor.storage().nbytes()
               << "\nstorage offset: " << tensor.storage_offset();
  return absl::OkStatus();
}

}  // namespace torch_tpu
