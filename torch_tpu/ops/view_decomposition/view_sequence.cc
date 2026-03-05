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

#include "torch_tpu/ops/view_decomposition/view_sequence.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "c10/core/TensorImpl.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/unary.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<bool> RemoveNoOps(
    ViewSequence& sequence, absl::Span<const int64_t> contiguous_base_shape) {
  if (sequence.empty()) {
    return false;
  }
  auto layout = MakeContiguousBaseLayout(contiguous_base_shape);
  size_t write_index = 0;
  bool any_no_ops = false;
  for (size_t read_index = 0; read_index < sequence.size(); ++read_index) {
    TT_ASSIGN_OR_RETURN(const bool updated,
                        UpdateLayout(layout, sequence[read_index]));
    if (updated) {
      if (write_index != read_index) {
        // Don't overwrite an operation onto itself;
        // if there is a prefix of meaningful operations, leave them as-is.
        sequence[write_index] = std::move(sequence[read_index]);
      }
      ++write_index;
    } else {
      ABSL_VLOG(3) << "[RemoveNoOps] Removing no-op: " << sequence[read_index];
      any_no_ops = true;
    }
  }
  if (any_no_ops) {
    sequence.resize(write_index);
  }
  return any_no_ops;
}

// Returns true if the second primitive was merged into the first.
// If the merge occurred, the second primitive is invalidated.
template <typename T>
absl::StatusOr<bool> MaybeMergeSequential(T& first, ViewPrimitive& second) {
  if (std::holds_alternative<T>(second)) {
    ABSL_VLOG(3) << "[MergeAllSequential] Merging " << first << " and "
                 << second;
    TT_ASSIGN_OR_RETURN(
        first, Merge(std::move(first), std::move(std::get<T>(second))));
    ABSL_VLOG(3) << "[MergeAllSequential] Merged into " << first;
    return true;
  }
  return false;
}
// The decomposition sequence will only produce at most one broadcast, one
// transpose, one pad, and one bitcast each of complex-to-real, real-to-real,
// and real-to-complex, so we don't need to worry about merging them.
// Multiple unfolds may be necessary, but they cannot be merged (as they
// can only do one concatenation per primitive).
// Only reshapes and slices need to be merged.
template <>
absl::StatusOr<bool> MaybeMergeSequential(BroadcastPrimitive& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(TransposePrimitive& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(PadPrimitive& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(RealToRealBitcast& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(ComplexToRealBitcast& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(ViewAsComplex& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(UnfoldPrimitive& first,
                                          ViewPrimitive& second) {
  return false;
}
template <>
absl::StatusOr<bool> MaybeMergeSequential(ConjPrimitive& first,
                                          ViewPrimitive& second) {
  if (auto* second_conj = std::get_if<ConjPrimitive>(&second)) {
    // If they are exactly the same, they cancel out.
    // (active + active) -> inactive
    // (inactive + inactive) -> inactive
    // Note: RemoveNoOps runs before this, so we generally only see active ones.
    if (first == *second_conj) {
      first.is_set = false;
      return true;
    }
  }
  return false;
}

template <>
absl::StatusOr<bool> MaybeMergeSequential(ViewPrimitive& first,
                                          ViewPrimitive& second) {
  return std::visit(
      [&second](auto& first) { return MaybeMergeSequential(first, second); },
      first);
}

absl::StatusOr<bool> MergeAllSequential(ViewSequence& sequence) {
  if (sequence.size() < 2) {
    return false;
  }
  bool any_merges = false;
  size_t read_index = 0;  // Index of the last non-merged primitive so far.
  for (size_t to_merge_index = 1; to_merge_index < sequence.size();
       ++to_merge_index) {
    TT_ASSIGN_OR_RETURN(
        const bool merged,
        MaybeMergeSequential(sequence[read_index], sequence[to_merge_index]));
    if (merged) {
      any_merges = true;
      continue;
    }
    ++read_index;
    if (read_index == to_merge_index) {
      // Do nothing for the non-mergeable prefix of the list
      continue;
    }
    sequence[read_index] = std::move(sequence[to_merge_index]);
  }
  if (any_merges) {
    sequence.resize(read_index + 1);
  }
  return any_merges;
}

}  // namespace

bool operator==(const ViewPrimitive& lhs, const ViewPrimitive& rhs) {
  if (lhs.index() != rhs.index()) {
    return false;
  }
  if (auto* lhs_reshape = std::get_if<ReshapePrimitive>(&lhs)) {
    const auto& rhs_reshape = std::get<ReshapePrimitive>(rhs);
    return lhs_reshape->base_sizes == rhs_reshape.base_sizes &&
           lhs_reshape->new_sizes == rhs_reshape.new_sizes;
  }
  if (auto* lhs_transpose = std::get_if<TransposePrimitive>(&lhs)) {
    return lhs_transpose->permutation ==
           std::get<TransposePrimitive>(rhs).permutation;
  }
  if (auto* lhs_broadcast = std::get_if<BroadcastPrimitive>(&lhs)) {
    const auto& rhs_broadcast = std::get<BroadcastPrimitive>(rhs);
    return lhs_broadcast->new_sizes == rhs_broadcast.new_sizes &&
           lhs_broadcast->broadcast_dimensions ==
               rhs_broadcast.broadcast_dimensions;
  }
  if (auto* lhs_slice = std::get_if<SlicePrimitive>(&lhs)) {
    return lhs_slice->slice_dims == std::get<SlicePrimitive>(rhs).slice_dims;
  }
  if (auto* lhs_pad = std::get_if<PadPrimitive>(&lhs)) {
    return lhs_pad->pad_dims == std::get<PadPrimitive>(rhs).pad_dims;
  }
  if (auto* lhs_real_to_real = std::get_if<RealToRealBitcast>(&lhs)) {
    const auto& rhs_real_to_real = std::get<RealToRealBitcast>(rhs);
    return lhs_real_to_real->from_type == rhs_real_to_real.from_type &&
           lhs_real_to_real->to_type == rhs_real_to_real.to_type;
  }
  if (auto* lhs_complex_to_real = std::get_if<ComplexToRealBitcast>(&lhs)) {
    const auto& rhs_complex_to_real = std::get<ComplexToRealBitcast>(rhs);
    return lhs_complex_to_real->complex_element_type ==
               rhs_complex_to_real.complex_element_type &&
           lhs_complex_to_real->bitcast_type ==
               rhs_complex_to_real.bitcast_type;
  }
  if (auto* lhs_view_as_complex = std::get_if<ViewAsComplex>(&lhs)) {
    return lhs_view_as_complex->complex_element_type ==
           std::get<ViewAsComplex>(rhs).complex_element_type;
  }
  if (auto* lhs_unfold = std::get_if<UnfoldPrimitive>(&lhs)) {
    const auto& rhs_unfold = std::get<UnfoldPrimitive>(rhs);
    return lhs_unfold->start_index == rhs_unfold.start_index &&
           lhs_unfold->limit_index == rhs_unfold.limit_index &&
           lhs_unfold->window_stride == rhs_unfold.window_stride &&
           lhs_unfold->window_size == rhs_unfold.window_size;
  }
  if (std::holds_alternative<ConjPrimitive>(lhs)) {
    const auto& lhs_conj = std::get<ConjPrimitive>(lhs);
    const auto& rhs_conj = std::get<ConjPrimitive>(rhs);
    return lhs_conj.is_set == rhs_conj.is_set;
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

bool operator!=(const ViewPrimitive& lhs, const ViewPrimitive& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ViewPrimitive& primitive) {
  // For some reason, std::visit doesn't work here.
  if (auto* reshape = std::get_if<ReshapePrimitive>(&primitive)) {
    return os << *reshape;
  }
  if (auto* transpose = std::get_if<TransposePrimitive>(&primitive)) {
    return os << *transpose;
  }
  if (auto* broadcast = std::get_if<BroadcastPrimitive>(&primitive)) {
    return os << *broadcast;
  }
  if (auto* slice = std::get_if<SlicePrimitive>(&primitive)) {
    return os << *slice;
  }
  if (auto* pad = std::get_if<PadPrimitive>(&primitive)) {
    return os << *pad;
  }
  if (auto* real_to_real = std::get_if<RealToRealBitcast>(&primitive)) {
    return os << *real_to_real;
  }
  if (auto* complex_to_real = std::get_if<ComplexToRealBitcast>(&primitive)) {
    return os << *complex_to_real;
  }
  if (auto* view_as_complex = std::get_if<ViewAsComplex>(&primitive)) {
    return os << *view_as_complex;
  }
  if (auto* unfold = std::get_if<UnfoldPrimitive>(&primitive)) {
    return os << *unfold;
  }
  if (std::holds_alternative<ConjPrimitive>(primitive)) {
    const auto& conj = std::get<ConjPrimitive>(primitive);
    if (conj.is_set) {
      return os << "conj()";
    } else {
      return os << "conj(is_set=false)";
    }
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

absl::StatusOr<bool> UpdateLayout(StridedLayout& layout,
                                  const ViewPrimitive& primitive) {
  return std::visit(
      [&layout](const auto& primitive) {
        return UpdateLayout(layout, primitive);
      },
      primitive);
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ViewPrimitive& primitive) {
  return std::visit(
      [&input](const auto& primitive) {
        if constexpr (std::is_same_v<decltype(primitive),
                                     const ConjPrimitive&>) {
          if (primitive.is_set) {
            return BuildConjPhysicalShlo(input);
          }
          return absl::StatusOr<mlir::MlirOp>(input);
        } else {
          return ViewPrimitiveShlo(input, primitive);
        }
      },
      primitive);
}

absl::StatusOr<bool> UpdateLayout(StridedLayout& layout,
                                  ViewSequenceSpan view_sequence) {
  bool updated = false;
  for (const auto& primitive : view_sequence) {
    TT_ASSIGN_OR_RETURN(bool updated_this_time,
                        UpdateLayout(layout, primitive));
    updated |= updated_this_time;
  }
  return updated;
}

absl::Status Simplify(ViewSequence& sequence,
                      absl::Span<const int64_t> contiguous_base_shape) {
  ABSL_VLOG(3) << "[Simplify] Pre-simplified view sequence: "
               << ToString(sequence);
  bool any_changes = false;
  do {
    TT_ASSIGN_OR_RETURN(const bool any_no_ops,
                        RemoveNoOps(sequence, contiguous_base_shape));
    TT_ASSIGN_OR_RETURN(const bool any_merges, MergeAllSequential(sequence));
    any_changes = any_no_ops || any_merges;
  } while (any_changes);
  ABSL_VLOG(3) << "[Simplify] Simplified view sequence: " << ToString(sequence);
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> ViewSequenceShlo(mlir::MlirOp input,
                                              ViewSequenceSpan view_sequence) {
  mlir::MlirOp current = input;
  for (const auto& primitive : view_sequence) {
    TT_ASSIGN_OR_RETURN(current, ViewPrimitiveShlo(current, primitive));
  }
  return current;
}

namespace {

struct ViewCacheKeyVisitor {
  // Special handling for reshape primitive since we only support affine
  // reshapes currently.
  // TODO: Enhance this to support squeeze/unsqueeze.
  absl::StatusOr<std::string> operator()(
      const ReshapePrimitive& primitive) const {
    TT_ASSIGN_OR_RETURN(
        const ReshapeReassociation reassociation,
        GetReshapeReassociation(primitive.base_sizes, primitive.new_sizes));
    return absl::StrCat("reshape_", ReassociationToString(reassociation));
  }

  // Fallback for all other view primitives.
  // Currently unsupported:
  //  - BroadcastPrimitive relies on static shape information for output.
  //  - UnfoldPrimitive relies on static shape information for output.
  //  - SlicePrimitive relies on static shape information for output.
  template <typename T>
  absl::StatusOr<std::string> operator()(const T& primitive) const {
    if constexpr (std::is_same_v<T, TransposePrimitive> ||
                  std::is_same_v<T, PadPrimitive> ||
                  std::is_same_v<T, ConjPrimitive> ||
                  std::is_same_v<T, RealToRealBitcast> ||
                  std::is_same_v<T, ComplexToRealBitcast> ||
                  std::is_same_v<T, ViewAsComplex>) {
      std::ostringstream os;
      os << primitive;
      return os.str();
    }
    return TT_ERROR(error::kUnimplemented) << "View primitive does not support "
                                              "dynamic cache keys for type: "
                                           << typeid(T).name();
  }
};

absl::StatusOr<std::string> ViewToCacheKey(OpParamCacheKeys& param_keys,
                                           const ViewPrimitive& primitive) {
  ViewCacheKeyVisitor visitor{};
  return std::visit(visitor, primitive);
}

absl::StatusOr<OpParamCacheKeys> SymbolicViewCacheKey(
    ViewSequenceSpan view_sequence) {
  // FIXME: How should we iteratively add to a builder?
  // Add method to return empty builder or return a built OpParamCacheKeys
  // on each iteration?
  ABSL_VLOG(3) << "[SymbolicViewCacheKey] View sequence: "
               << ToString(view_sequence);
  OpParamCacheKeys param_keys;
  OpParamCacheKeys::Builder builder = param_keys.SetParam("A", "B");
  for (auto [index, primitive] : llvm::enumerate(view_sequence)) {
    TT_ASSIGN_OR_RETURN(std::string view_cache_key,
                        ViewToCacheKey(param_keys, primitive));
    ABSL_VLOG(3) << "[SymbolicViewCacheKey] Symbolic reshape key: "
                 << view_cache_key;
    builder = param_keys.SetParam(absl::StrCat("view_", index), view_cache_key);
  }
  return *builder;
}

}  // namespace

absl::StatusOr<OpParamCacheKeys> ViewSequenceCacheKey(
    ViewSequenceSpan view_sequence, const c10::TensorImpl& tensor) {
  // First try to create a symbolic cache key that represents view ops as
  // mappings from input to output tensor shapes, i.e. Reshape[A,B]->[A*B]
  // or transpose[A,B]{0,1}, instead of embedding static input and output shapes
  // into the cache key.
  absl::StatusOr<OpParamCacheKeys> symbolic_key =
      SymbolicViewCacheKey(view_sequence);
  if (symbolic_key.ok()) {
    ABSL_VLOG(3) << "[ViewSequenceCacheKey] Built symbolic cache key";
    return symbolic_key;
  }

  // Fall back to using static tensor shapes for the cache key.
  ABSL_VLOG(3) << "[ViewSequenceCacheKey] Failed to create symbolic cache key: "
               << symbolic_key.status().message();
  return *OpParamCacheKeys::SetParam("strides", tensor.strides())
              .SetParam("storage_offset", tensor.storage_offset());
}

}  // namespace torch_tpu
