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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "ATen/TensorUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "c10/util/typeid.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

// Both _reshape_alias() and view() require kernel registrations according to
// https://docs.pytorch.org/docs/main/accelerator/operators.html
//
// view() is expected to:
//   1. Check that the requested size is view-compatible with the input shape.
//   2. If it is, then create an aliasing view of the c10::Storage with the
//      requested shape; otherwise, raise an exception
// This is required because torch assumes that devices may have specific
// requirements for view compatibility.
//
// torch_tpu does not have any specific view compatibility requirements, so
// we simply use PyTorch's view stride computation logic and forward the result
// to AtenAsStrided().
//
// _reshape_alias() is expected to be identical to view(), except that it
// doesn't need to check for view compatibility; it is not a user-facing API,
// and PyTorch guarantees it will only be called when view compatibility has
// already been verified.
//
// Some other device implementations (for example, Vulkan device in
// aten/src/ATen/native/vulkan/ops/Shape.cpp) simply reuse the implementation
// of view() for _reshape_alias(), which redundantly applies compatibility
// checks. For efficiency, we do not do this, and skip the compatibility check
// in _reshape_alias().

namespace torch_tpu {

namespace {

// torch passes the target size for view (and reshape) as SymInts.
// SymInts are used in torch's tracing logic; they record any writes against
// them, and then c10::guard_int is used to resolve them to a concrete, readable
// value. We need to read them here to dispatch MLIR ops with correct shapes.
// Additionally, torch allows one dimension to be specified as -1, which means
// "infer", by keeping the total number of elements the same between input
// and output.
absl::StatusOr<Dimensions> SymIntsToDimensions(const at::Tensor& self,
                                               c10::SymIntArrayRef size_sym) {
  // Reserve space for the output.
  Dimensions new_size_vec;
  new_size_vec.reserve(size_sym.size());

  // Resolve symints and identify the -1 dimension.
  int64_t negative_one_dim = -1;
  int64_t new_size_product = 1;
  for (const auto i : c10::irange(size_sym.size())) {
    const c10::SymInt& s_int = size_sym[i];
    const int64_t dim_size = s_int.guard_int(__FILE__, __LINE__);
    TT_RET_CHECK(dim_size >= -1, error::kInvalidArgument)
        << "dimensions must be non-negative, with at most one -1. Got "
           "dimension with size: "
        << dim_size;
    if (dim_size == -1) {
      TT_RET_CHECK(negative_one_dim == -1, error::kInvalidArgument)
          << "can only infer one dimension (size -1), got reshape dimensions: "
          << ToString(size_sym);
      negative_one_dim = i;
      new_size_vec.push_back(-1);  // Placeholder
    } else {
      TT_ASSIGN_OR_RETURN(
          new_size_product, SafeMultiply(new_size_product, dim_size),
          _.SetOverride() << "cannot infer dimension because the number of "
                             "elements overflows as int64");
      new_size_vec.push_back(dim_size);
    }
  }
  const int64_t self_numel = self.sym_numel().guard_int(__FILE__, __LINE__);
  // Infer the -1 dimension if there is one.
  if (negative_one_dim != -1) {
    TT_RET_CHECK(new_size_product != 0, error::kInvalidArgument)
        << "cannot infer dimension for input shape" << ToString(self.sizes())
        << " and output shape " << ToString(new_size_vec)
        << " because the new size has 0 total elements";
    TT_RET_CHECK(self_numel % new_size_product == 0, error::kInvalidArgument)
        << "cannot infer dimension for input shape " << ToString(self.sizes())
        << " and output shape " << ToString(new_size_vec)
        << " because the total number of input elements (" << self_numel
        << ") is not a multiple of the product of the specified "
        << "target elements (" << new_size_product << ")";
    const int64_t inferred_dim_size = self_numel / new_size_product;
    new_size_vec[negative_one_dim] = inferred_dim_size;
    TT_ASSIGN_OR_RETURN(
        new_size_product, SafeMultiply(new_size_product, inferred_dim_size),
        _.SetOverride()
            << "cannot infer dimension for shape " << ToString(new_size_vec)
            << " because the number of elements overflows as int64");
  }
  TT_RET_CHECK(self_numel == new_size_product, error::kInvalidArgument)
      << "cannot reshape size " << ToString(self.sizes()) << " to shape "
      << ToString(size_sym) << " because the number of elements does not match "
      << "(" << self_numel << " != " << new_size_product << ")";
  return new_size_vec;
}
}  // namespace

at::Tensor AtenReshapeAlias(const at::Tensor& self,
                            c10::SymIntArrayRef size_sym,
                            c10::SymIntArrayRef stride_sym) {
  TT_KERNEL(OpName::kReshapeAlias, _, (self), {
    return AtenAsStrided(self, size_sym, stride_sym, c10::nullopt);
  });
}

at::Tensor AtenView(const at::Tensor& self, c10::SymIntArrayRef size_sym) {
  TT_KERNEL(OpName::kView, _, (self), {
    // view is allowed to have one dimension as "-1", which needs to be resolved
    // to a positive shape.
    TT_ASSIGN_OR_THROW(Dimensions new_size,
                       SymIntsToDimensions(self, size_sym));

    // Use PyTorch's default logic for view stride computation.
    std::optional<std::vector<int64_t>> new_stride =  // INT_VEC_OK=c10 API
        at::detail::computeStride(self.sizes(), self.strides(), new_size);
    TT_CHECK_THROW(new_stride.has_value(), error::kInvalidArgument)
        << "output shape not view-compatible with input shape. Consider using "
           "reshape() instead of view() to return a new contiguous tensor";

    // Redispatch to AtenAsStrided() with the resolved shape and strides.
    c10::SymIntArrayRef new_size_inferred_sym =
        c10::fromIntArrayRefKnownNonNegative(new_size);
    c10::SymIntArrayRef new_stride_sym =
        c10::fromIntArrayRefKnownNonNegative(*new_stride);
    at::Tensor tensor = AtenAsStrided(self, new_size_inferred_sym,
                                      new_stride_sym, c10::nullopt);
    if (self.is_conj() && !tensor.is_conj()) {
      tensor = tensor.conj();
    }
    return tensor;
  });
}

at::Tensor AtenViewAsReal(const at::Tensor& self) {
  TT_KERNEL(OpName::kViewAsReal, _, (self), {
    TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
        << "expected complex dtypes (torch.complex64 "
           "and torch.complex128), got "
        << ToString(self.scalar_type());

    Dimensions output_dims = CopyIntVector(self.sizes());
    output_dims.push_back(2);
    Strides output_strides;
    output_strides.reserve(self.dim() + 1);
    for (auto stride : self.strides()) {
      output_strides.push_back(stride * 2);
    }
    output_strides.push_back(1);
    const int64_t output_storage_offset = self.storage_offset() * 2;
    at::ScalarType output_dtype = c10::toRealValueType(self.scalar_type());
    auto output_type_meta = caffe2::TypeMeta::fromScalarType(output_dtype);

    c10::Storage storage_copy = self.storage();
    at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
        std::move(storage_copy),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), output_type_meta));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        output_dims, output_strides, output_storage_offset);
    return tensor;
  });
}

at::Tensor AtenViewAsComplex(const at::Tensor& self) {
  TT_KERNEL(OpName::kViewAsComplex, _, (self), {
    TT_CHECK_THROW(self.scalar_type() == at::ScalarType::Float ||
                       self.scalar_type() == at::ScalarType::Double,
                   error::kInvalidArgument)
        << "this op currently only supports float32 and float64 dtype as "
           "input, got "
        << torch_tpu::ToString(self.scalar_type());

    TT_CHECK_THROW(self.dim() > 0, error::kInvalidArgument)
        << "complex tensors require at least 2 elements, and cannot be created "
           "from single scalar floats, got "
        << self.dim() << " element";

    TT_CHECK_THROW(self.size(-1) == 2, error::kInvalidArgument)
        << "the last dimension of the input tensor should be 2, got "
        << self.size(-1);

    TT_CHECK_THROW(self.stride(-1) == 1, error::kInvalidArgument)
        << "input tensor must have a stride of 1 for its last dimension, got "
        << self.stride(-1);

    for (int i = 0; i < self.dim() - 1; ++i) {
      TT_CHECK_THROW(self.stride(i) % 2 == 0, error::kInvalidArgument)
          << "the stride of dimension " << i << " must be an even number, got "
          << self.stride(i);
    }

    Dimensions output_dims = CopyIntVector(self.sizes());
    output_dims.pop_back();
    Strides output_strides;
    output_strides.reserve(self.dim() - 1);
    for (auto i = 0; i < self.dim() - 1; ++i) {
      output_strides.push_back(self.stride(i) / 2);
    }
    const int64_t output_storage_offset = self.storage_offset() / 2;
    at::ScalarType output_dtype = c10::toComplexType(self.scalar_type());
    auto output_type_meta = caffe2::TypeMeta::fromScalarType(output_dtype);

    c10::Storage storage_copy = self.storage();
    at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
        std::move(storage_copy),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), output_type_meta));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        output_dims, output_strides, output_storage_offset);
    return tensor;
  });
}

}  // namespace torch_tpu
