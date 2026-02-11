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

// This file is needed to make sure the header file can be compiled on its own.
#include "torch_tpu/common/macro_utils.h"

#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

namespace torch_tpu {
namespace internal {

std::vector<std::string_view> ArgsAsStrings(const std::string_view csv_string) {
  std::vector<std::string_view> split_args =
      absl::StrSplit(csv_string, absl::ByChar(','), absl::SkipWhitespace());
  for (auto& arg : split_args) {
    arg = absl::StripAsciiWhitespace(arg);
  }
  return split_args;
}

}  // namespace internal
}  // namespace torch_tpu
