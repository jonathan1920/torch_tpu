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
#include <string_view>
#include <vector>

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

std::vector<std::string_view> ParseArgTypesOrEmpty(
    const std::string_view func_sig) {
  std::vector<std::string_view> arg_types;
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
    arg_types.push_back(args_str.substr(0, type_end_idx));
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
