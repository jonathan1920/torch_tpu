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

#include "torch_tpu/ops/macros/logging.h"

#include <cctype>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/strip.h"

namespace torch_tpu {
namespace internal {

// Given a string of comma-separated types, returns the index of the end of the
// first type.
[[nodiscard]] static int FindTypeEnd(std::string_view args_str) {
  int open_brackets = 0;
  int open_parentheses = 0;
  int i = 0;
  for (; i < args_str.size(); ++i) {
    const char c = args_str[i];
    if (c == '<') {
      ++open_brackets;
    } else if (c == '>') {
      --open_brackets;
    } else if (c == '(') {
      ++open_parentheses;
    } else if (c == ')') {
      --open_parentheses;
    } else if (c == ',' && open_brackets == 0 && open_parentheses == 0) {
      // The type ends on a comma that is not inside any brackets or
      // parentheses. Or it ends on the end of the string.
      break;
    }
  }
  return i;
}

// Normalizes a type name parsed from __PRETTY_FUNCTION__.
// __PRETTY_FUNCTION__ doesn't normalize the type names, so we may get
// different strings depending on how the function definition is written.
[[nodiscard]] static std::string NormalizePrettyFunctionTypeName(
    std::string_view type_name) {
  // Ignore the "const " prefix if any, as we only care about the underlying
  // type.
  type_name = absl::StripPrefix(type_name, "const ");
  // Ignore the " &" suffix if any, as we only care about the underlying type.
  type_name = absl::StripSuffix(type_name, " &");

  // Replace type aliases with their underlying types.
  static const absl::NoDestructor<std::regex> kOptionalRegex(
      R"(\bc10::optional\b)");
  static const absl::NoDestructor<std::regex> kScalarTypeRegex(
      R"(\bc10::ScalarType\b)");
  static const absl::NoDestructor<std::regex> kOptionalArrayRefRegex(
      R"(\bc10::OptionalArrayRef<int64_t>)");
  std::string type_name_str = std::regex_replace(
      std::string(type_name), *kOptionalRegex, "std::optional");
  type_name_str =
      std::regex_replace(type_name_str, *kScalarTypeRegex, "at::ScalarType");
  type_name_str = std::regex_replace(type_name_str, *kOptionalArrayRefRegex,
                                     "at::OptionalIntArrayRef");
  return type_name_str;
}

std::vector<std::string> ParseArgTypesOrEmpty(const std::string_view func_sig) {
  std::vector<std::string> arg_types;
  // Parse the argument types from the function signature, e.g.
  // "void foo(int a, const char* b, double c)"
  // -> {"int", "const char*", "double"}.

  // Find the last '(' and ')' character. We skip earlier ones because they may
  // be part of the function name, e.g.
  // "void torch_tpu::(anonymous namespace)::NullaryFunc()".
  const int start_idx = func_sig.find_last_of('(');
  if (start_idx == std::string_view::npos) {
    return arg_types;
  }
  const int end_idx = func_sig.find_last_of(')');
  if (end_idx == std::string_view::npos) {
    return arg_types;
  }
  std::string_view args_str =
      func_sig.substr(start_idx + 1, end_idx - start_idx - 1);
  while (!args_str.empty()) {
    const int type_end_idx = FindTypeEnd(args_str);
    arg_types.push_back(
        NormalizePrettyFunctionTypeName(args_str.substr(0, type_end_idx)));
    args_str = args_str.substr(type_end_idx < args_str.size() ? type_end_idx + 1
                                                              : type_end_idx);
    // Skip leading whitespace.
    while (!args_str.empty() && std::isspace(args_str.front())) {
      args_str = args_str.substr(1);
    }
  }
  return arg_types;
}

}  // namespace internal
}  // namespace torch_tpu
