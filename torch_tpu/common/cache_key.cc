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
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "c10/core/Layout.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"

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
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string FormatParamCacheKey(c10::SymIntArrayRef value) {
  if (value.empty()) {
    return "";
  }
  return absl::StrCat(
      "[",
      absl::StrJoin(value, ",",
                    [](std::string* out, const c10::SymInt& elem) {
                      absl::StrAppend(out, FormatParamCacheKey(elem));
                    }),
      "]");
}

std::string FormatParamCacheKey(const c10d::ReduceOp value) {
  const c10d::ReduceOp::RedOpType reduce_op_type = value;
  switch (reduce_op_type) {
    // go/keep-sorted start
    case c10d::ReduceOp::AVG:
      return "avg";
    case c10d::ReduceOp::BAND:
      return "band";
    case c10d::ReduceOp::BOR:
      return "bor";
    case c10d::ReduceOp::BXOR:
      return "bxor";
    case c10d::ReduceOp::MAX:
      return "max";
    case c10d::ReduceOp::MIN:
      return "min";
    case c10d::ReduceOp::PREMUL_SUM:
      return "premul_sum";
    case c10d::ReduceOp::PRODUCT:
      return "product";
    case c10d::ReduceOp::SUM:
      return "sum";
    case c10d::ReduceOp::UNUSED:
      return "unused";
      // go/keep-sorted end
  };
  return absl::StrFormat("enum%d", reduce_op_type);
}

std::string FormatParamCacheKey(const std::string_view value) {
  return std::string(value);
}

std::string FormatParamCacheKey(const absl::Span<const int64_t> value) {
  return absl::StrCat("[", absl::StrJoin(value, ","), "]");
}

std::string FormatParamCacheKey(const at::Layout value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string FormatParamCacheKey(const at::MemoryFormat value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string FormatParamCacheKey(const at::Device value) {
  // The device string may contain `:`, which may cause ambiguity when
  // parsing the cache key, so we quote it.
  return FormatParamCacheKey(value.str());
}

std::string FormatParamCacheKey(const at::ITensorListRef& value) {
  if (value.empty()) {
    return "";
  }
  return absl::StrCat(
      "[",
      absl::StrJoin(value, ",",
                    [](std::string* out, const at::Tensor& tensor) {
                      absl::StrAppend(out, FormatParamCacheKey(tensor.sizes()));
                    }),
      "]");
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
    return OpParamCacheKeys(std::move(name_to_value_));
  }
  // Clear the builder state before returning the error.
  auto result = first_error_;
  first_error_ = absl::OkStatus();
  name_to_value_.clear();
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

std::string ToString(CompilationCacheKey key) {
  std::ostringstream ss;
  ss << key;
  return ss.str();
}

}  // namespace torch_tpu
