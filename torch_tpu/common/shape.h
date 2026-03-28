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

#ifndef TORCH_TPU_COMMON_SHAPE_H_
#define TORCH_TPU_COMMON_SHAPE_H_

#include <cstdint>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "xla/shape.h"

namespace torch_tpu {

// A dimension that should be interpreted as dynamic.
// Tensors always have a concrete shape, but we might want to build operations
// they are involved in with a dynamic dimension so they can be reused.
struct BoundedDynamicDimension {
  int64_t dimension;
  int64_t lower_bound;
  int64_t upper_bound;
};

// A tensor shape, consisting of dimensions and element type.
// This is roughly equivalent to xla::Shape, but is just a struct and not a
// full protobuf definition.
struct Shape {
  Dimensions dimensions;
  mlir::ElementType dtype;
  // We choose 1 as the inlined size because most Shapes are not dynamic, and
  // most dynamic Shapes are dynamic in 1 dimension only, so this allows for
  // no heap allocation in the common case.
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dimensions;
};
static_assert(sizeof(Shape) == 96);

inline bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.dtype == rhs.dtype && lhs.dimensions == rhs.dimensions;
}

// Converts an XLA shape to a torch_tpu shape.
absl::StatusOr<Shape> MakeShape(const xla::Shape& xla_shape);

std::string ToString(const Shape& s);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_SHAPE_H_
