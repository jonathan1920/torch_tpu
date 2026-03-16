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

#include "torch_tpu/common/cache_key.h"

#include <cstdint>
#include <ios>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace internal {

absl::StatusOr<std::string> FormatParamCacheKey(const at::Scalar value) {
  std::string key;
  if (value.isFloatingPoint()) {
    key = LosslessToString(value.toDouble());
  } else if (value.isIntegral(/*include_bool=*/false)) {
    key = absl::StrCat(value.toLong());
  } else if (value.isBoolean()) {
    key = value.toBool() ? "1" : "0";
  } else if (value.isComplex()) {
    key = LosslessToString(value.toComplexDouble());
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "Unable to create key for scalar type.";
  }
  return absl::StrCat(key, ":", c10::toString(value.type()));
}

std::string FormatParamCacheKey(const c10::SymInt& value) {
  return ToString(value);
}

std::string FormatParamCacheKey(c10::SymIntArrayRef value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}

std::string FormatParamCacheKey(const c10d::ReduceOp value) {
  const c10d::ReduceOp::RedOpType reduce_op_type = value;
  return ToString(reduce_op_type);
}

std::string FormatParamCacheKey(const std::string_view value) {
  return std::string(value);
}

std::string FormatParamCacheKey(const absl::Span<const int64_t> value) {
  return absl::StrCat("[", absl::StrJoin(value, ","), "]");
}

std::string FormatParamCacheKey(const at::Layout value) {
  return ToString(value);
}

std::string FormatParamCacheKey(const at::MemoryFormat value) {
  return ToString(value);
}

std::string FormatParamCacheKey(const at::Device value) {
  // The device string may contain `:`, which may cause ambiguity when
  // parsing the cache key, so we quote it.
  return FormatParamCacheKey(value.str());
}

std::string_view ParseNextArgName(std::string_view& args_str) {
  // Find the first ',' or the end of the string.
  auto name_end = args_str.find(',');
  if (name_end == std::string_view::npos) {
    name_end = args_str.size();
  }

  // Extract the first argument name, removing whitespace from both sides.
  std::string_view name = args_str.substr(0, name_end);
  name = absl::StripAsciiWhitespace(name);

  // Remove the name and the following comma (if any) from args_str.
  args_str.remove_prefix(name_end == args_str.size() ? name_end : name_end + 1);
  return name;
}

}  // namespace internal

absl::StatusOr<OpParamCacheKeys> OpParamCacheKeys::Builder::operator*() {
  if (first_error_.ok()) {
    return std::move(param_keys_);
  }
  // Clear the builder state before returning the error.
  auto result = first_error_;
  first_error_ = absl::OkStatus();
  param_keys_.name_to_value_.clear();
  return std::move(result);
}

std::string CompilationCacheKey::CompactFormat() const {
  return absl::StrFormat("%016x_%016x", shapeless_key.key, dimensions_key.key);
}

std::ostream& operator<<(std::ostream& os, CompilationCacheKey key) {
  os << "CompilationCacheKey{shapeless_key=" << std::hex
     << key.shapeless_key.key << ", dimensions_key=" << key.dimensions_key.key
     << std::dec << "}";
  return os;
}

ShapeDynamismMetadata::ShapeDynamismMetadata(
    const std::vector<Shape>& input_shapes) {
  for (const Shape& shape : input_shapes) {
    const int64_t tensor_start_dim = input_dimension_bounds_.size();
    for (int64_t dim : shape.dimensions) {
      input_dimension_bounds_.push_back({dim, dim});
    }
    for (const auto& dynamic_dim : shape.dynamic_dimensions) {
      const int64_t dynamic_dim_index =
          tensor_start_dim + dynamic_dim.dimension;
      input_dimension_bounds_[dynamic_dim_index] = {dynamic_dim.lower_bound,
                                                    dynamic_dim.upper_bound};
    }
  }
}

namespace {
bool IsShapeCompatibleWithBounds(const Shape& shape,
                                 absl::Span<const DimensionBounds> bounds) {
  TT_CHECK_THROW(shape.dimensions.size() == bounds.size(), error::kInternal)
      << "Shape and bounds spans must have the same size.";
  for (int i = 0; i < shape.dimensions.size(); ++i) {
    const int64_t dim = shape.dimensions[i];
    if (dim < bounds[i].lower || dim > bounds[i].upper) {
      return false;
    }
  }
  return true;
}

std::string GetIncompatibilityErrorMsg(
    int64_t index, const Shape& shape,
    absl::Span<const DimensionBounds> bounds) {
  return absl::StrCat(
      "Input shape is incompatible with dynamism metadata at index ", index,
      ". ", "input shapes: ", ToString(shape.dimensions), " dynamism bounds: ",
      absl::StrJoin(
          bounds, ",", [](std::string* out, const DimensionBounds& bounds) {
            absl::StrAppend(out, "[", bounds.lower, ",", bounds.upper, "]");
          }));
}

Shape GetPaddingShape(const Shape& shape,
                      absl::Span<const DimensionBounds> bounds) {
  TT_CHECK_THROW(shape.dimensions.size() == bounds.size(), error::kInternal)
      << "Shape and bounds spans must have the same size.";
  Shape padding_shape{.dimensions = shape.dimensions, .dtype = shape.dtype};
  for (int i = 0; i < shape.dimensions.size(); ++i) {
    TT_CHECK_THROW(padding_shape.dimensions[i] >= bounds[i].lower &&
                       padding_shape.dimensions[i] <= bounds[i].upper,
                   error::kInternal)
        << GetIncompatibilityErrorMsg(i, {shape}, bounds);
    if (bounds[i].lower != bounds[i].upper) {
      padding_shape.dynamic_dimensions.push_back(
          {.dimension = i,
           .lower_bound = bounds[i].lower,
           .upper_bound = bounds[i].upper});
    }
  }
  return padding_shape;
}
}  // namespace

bool ShapeDynamismMetadata::IsCompatible(
    const std::vector<Shape>& input_shapes) const {
  int64_t flattened_size = 0;
  for (const Shape& shape : input_shapes) {
    flattened_size += shape.dimensions.size();
  }
  if (flattened_size != input_dimension_bounds_.size()) {
    return false;
  }
  int64_t index = 0;
  absl::Span<const DimensionBounds> bounds = input_dimension_bounds_;
  for (const Shape& shape : input_shapes) {
    int64_t span_size = shape.dimensions.size();
    if (!IsShapeCompatibleWithBounds(shape, bounds.subspan(index, span_size))) {
      return false;
    }
    index += span_size;
  }
  return true;
}

std::vector<Shape> ShapeDynamismMetadata::GetPaddingShapes(
    const std::vector<Shape>& input_shapes) const {
  TT_CHECK_THROW(IsCompatible(input_shapes), error::kInternal)
      << "Input shapes are incompatible with dynamism metadata.";
  std::vector<Shape> padding_shapes;
  padding_shapes.reserve(input_shapes.size());
  int index = 0;
  absl::Span<const DimensionBounds> bounds = input_dimension_bounds_;
  for (const Shape& shape : input_shapes) {
    int64_t span_size = shape.dimensions.size();
    padding_shapes.push_back(
        GetPaddingShape(shape, bounds.subspan(index, span_size)));
    index += span_size;
  }
  return padding_shapes;
}

}  // namespace torch_tpu
