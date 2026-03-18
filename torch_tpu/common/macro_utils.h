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

#ifndef TORCH_TPU_COMMON_MACRO_UTILS_H_
#define TORCH_TPU_COMMON_MACRO_UTILS_H_

#include <vector>

#include "absl/strings/str_split.h"  // IWYU pragma: keep for macro
#include "absl/strings/string_view.h"  // IWYU pragma: keep for macro

// Preprocessor utilities for making macro implementations more readable.

// TT_REQUIRE_SEMICOLON_ requires a semicolon after its expansion. Useful for
// enforcing that a macro must be used with a trailing semicolon.
//
// Example:
//   #define TT_FOO(x) \
//     void foo##x() { ... } \
//     TT_REQUIRE_SEMICOLON_
//
//   This ensures that TT_FOO() is used with a trailing semicolon, e.g.
//     TT_FOO(bar);
//   but not:
//     TT_FOO(bar)
//
// NOTE: This macro is not needed often. If we define a macro that expands to
// C++ statements or type definitions, we just need to omit the trailing
// semicolon from the macro definition. For example, in:
//
//   #define TT_BAR(x) \
//     class Bar##x { ... }
//   #define TT_BAZ(x) \
//     int x = ::torch_tpu::Baz(...)
//
// since the trailing semicolon is omitted in the macro definitions, users of
// TT_BAR() and TT_BAZ() will be forced to add a trailing semicolon. Therefore
// we don't need TT_REQUIRE_SEMICOLON_ in these macro definitions.
#define TT_REQUIRE_SEMICOLON_ \
  static_assert(true, "A semicolon is needed after this macro.")

// Converts a variadic list of identifiers into a vector of string_views.
//
// The arguments must be a list of single identifiers only. If the list is
// empty, the result will be an empty vector.
//
// Example:
//   std::vector<std::string_view> arg_names = TT_ARG_AS_STRINGS_(x, y, z);
//   // arg_names == {"x", "y", "z"}
//   std::vector<std::string_view> arg_names = TT_ARG_AS_STRINGS_();
//   // arg_names == {}
#define TT_ARGS_AS_STRINGS_(...) \
  ::torch_tpu::internal::ArgsAsStrings(#__VA_ARGS__)

namespace torch_tpu {
namespace internal {

#ifdef NDEBUG
inline constexpr bool kDebugMode = false;
#else
inline constexpr bool kDebugMode = true;
#endif  // NDEBUG

// Checks for whitespace at compile time.
[[nodiscard]] inline constexpr bool IsWhitespace(char c) noexcept {
  switch (c) {
    case ' ':
    case '\t':
    case '\n':
    case '\v':
    case '\f':
    case '\r':
      return true;
    default:
      return false;
  }
}

// Checks for identifier characters at compile time.
[[nodiscard]] inline constexpr bool IsIdentifierChar(char c) noexcept {
  return ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') ||
         ('A' <= c && c <= 'Z') || c == '_';
}

// State of the identifier parser.
enum class ParseIdentifierState {
  kBeforeIdentifier,
  kIdentifier,
  kAfterIdentifier,
};

// Returns true if the string is a comma-separated list of identifier names.
// This must be evaluated at compile time as we don't want to incur any
// runtime overhead.
[[nodiscard]] inline constexpr bool ArgsAreIdentifiers(
    const std::string_view csv_string) noexcept {
  auto state = ParseIdentifierState::kBeforeIdentifier;
  for (const char ch : csv_string) {
    if (state == ParseIdentifierState::kBeforeIdentifier) {
      if (IsWhitespace(ch)) {
        continue;
      }
      if (IsIdentifierChar(ch)) {
        state = ParseIdentifierState::kIdentifier;
        continue;
      }
      // Looking for an identifier, but got something else.
      return false;
    } else if (state == ParseIdentifierState::kIdentifier) {
      if (IsIdentifierChar(ch)) {
        continue;
      }
      if (IsWhitespace(ch)) {
        state = ParseIdentifierState::kAfterIdentifier;
        continue;
      }
      if (ch == ',') {
        state = ParseIdentifierState::kBeforeIdentifier;
        continue;
      }
      // Saw an unexpected character in an identifier.
      return false;
    } else {  // state == ParseIdentifierState::kAfterIdentifier
      if (IsWhitespace(ch)) {
        continue;
      }
      if (ch == ',') {
        state = ParseIdentifierState::kBeforeIdentifier;
        continue;
      }
      // Looking for whitespace or ',' after the identifier, but got something
      // else.
      return false;
    }
  }
  return true;
}

// Splits a comma-separated string into a vector of items, each trimmed of
// whitespace.
std::vector<std::string_view> ArgsAsStrings(std::string_view csv_string);

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_MACRO_UTILS_H_
