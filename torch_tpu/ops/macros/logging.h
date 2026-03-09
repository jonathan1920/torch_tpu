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

#ifndef TORCH_TPU_OPS_MACROS_LOGGING_H_
#define TORCH_TPU_OPS_MACROS_LOGGING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// At verbosity level 1, logs the arguments to a C++ kernel, along with their
// variable names. Example:
//   at::Tensor AtenFoo(const at::Tensor& lhs, const at::Tensor& rhs,
//                      const t::Scalar& value, absl::Span<const int64_t> dims,
//                      bool predicate) {
//     TT_LOG_KERNEL_START_("foo", lhs, rhs, value, dims, predicate);
//     ...
//   }
//  When enabled with absl logging verbosity level 1, the following is logged:
//   [C++ KERNEL foo]
//   lhs: <tensor debug>
//   rhs: <tensor debug>
//   value: <scalar debug>
//   dims: [1, 2, 3]
//   predicate: true
//
// Requirements:
//   - All arguments except the first are variable names.
//   - All argument types are "loggable". See the comments for FormatKernelArg()
//     for details.
//
// Do not use this macro directly. It's an implementation detail of TT_KERNEL().
//
// Implementation note: __VA_OPT__(,) expands to nothing if the variadic
// argument list is empty, and expands to a comma otherwise.
#define TT_LOG_KERNEL_START_(op_name, ...) \
  TT_LOG_KERNEL_START_IMPL_(op_name __VA_OPT__(, ) __VA_ARGS__)

// Implements TT_LOG_KERNEL_START_. The redirection is needed to allow
// TT_LOG_KERNEL_START_() to be used in a macro (e.g. TT_KERNEL).
//
// Implementation note: __VA_OPT__(,) expands to nothing if the variadic
// argument list is empty, and expands to a comma otherwise.
#define TT_LOG_KERNEL_START_IMPL_(op_name, ...)                                \
  do {                                                                         \
    static_assert(::torch_tpu::internal::ArgsAreIdentifiers(#__VA_ARGS__),     \
                  "All arguments for TT_LOG_KERNEL_START_() except the first " \
                  "must be identifier names.");                                \
    if (ABSL_VLOG_IS_ON(1)) {                                                  \
      std::ostringstream ss;                                                   \
      ::torch_tpu::internal::LogKernelName(ss, op_name);                       \
      const std::vector<std::string_view> arg_names =                          \
          TT_ARGS_AS_STRINGS_(__VA_ARGS__);                                    \
      ::torch_tpu::internal::LogKernelArgs(                                    \
          ss, arg_names __VA_OPT__(, ) __VA_ARGS__);                           \
      ABSL_LOG(INFO) << ss.str();                                              \
    }                                                                          \
  } while (false)

namespace internal {

inline void LogKernelName(std::ostream& ss, std::string_view op_name) {
  ss << "[C++ KERNEL " << op_name << "]\n";
}

inline void LogKernelName(std::ostream& ss, OpName op_name) {
  LogKernelName(ss, ToString(op_name));
}

// SupportsStreaming<T>::value is true if and only if T supports streaming
// to std::ostream.
template <typename T, typename = void>
struct SupportsStreaming : std::false_type {};
template <typename T>
struct SupportsStreaming<
    T, std::enable_if_t<!std::is_same_v<
           decltype(std::declval<std::ostream&>() << std::declval<T>()), void>>>
    : std::true_type {};

// Returns a human-readable string representation of the argument for kernel
// logging.
//
// By default, uses operator<< if it's defined for T, and falls back to
// FormatParamCacheKey if it's defined for T. Therefore, to specify how an
// argument type T should be logged, we have 3 choices:
//   1. Define operator<<() for T,
//   2. Define FormatParamCacheKey() for T, or
//   3. Define FormatKernelArg() for T.
// Since #2 is necessary for cache key computation anyway, we should always
// define it. However, we may still want to define #1 or #3 to improve the
// readability of the string, as the goal of FormatParamCacheKey() is to produce
// unique and compact keys for the compilation cache to be correct and efficient
// while the goal of kernel logging is to produce human-readable logs.
template <typename T>
[[nodiscard]] std::string FormatKernelArg(const T& arg) {
  if constexpr (SupportsStreaming<T>::value) {
    std::ostringstream ss;
    ss << arg;
    return ss.str();
  } else {
    absl::StatusOr<std::string> str_or = FormatParamCacheKey(arg);
    if (!str_or.ok()) {
      return absl::StrCat(
          "(failed to format kernel arg via FormatParamCacheKey())");
    }
    return *std::move(str_or);
  }
}

// Overloads for various types of kernel arguments.
[[nodiscard]] inline std::string_view FormatKernelArg(const bool arg) {
  return arg ? "true" : "false";
}
[[nodiscard]] inline std::string FormatKernelArg(const at::Scalar& arg) {
  return ToString(arg);
}
[[nodiscard]] inline std::string FormatKernelArg(const at::Tensor& arg) {
  return ToString(arg);
}
[[nodiscard]] inline std::string FormatKernelArg(const at::ScalarType& arg) {
  return std::string(ToString(arg));
}
[[nodiscard]] inline std::string FormatKernelArg(const at::Generator& arg) {
  return "at::Generator";
}
[[nodiscard]] inline std::string FormatKernelArg(
    const c10::OptionalArrayRef<int64_t>& arg) {
  return arg.has_value() ? absl::StrCat("[", absl::StrJoin(*arg, ", "), "]")
                         : "nullopt";
}

// To guarantee that the correct overload of FormatKernelArg() is found,
// overloads that invoke other FormatKernelArg() overloads should be declared
// first and their definitions should be put after all the other overloads.
template <typename T>
[[nodiscard]] std::string FormatKernelArg(const c10::List<T>& arg);
template <typename T, std::size_t N>
[[nodiscard]] std::string FormatKernelArg(const std::array<T, N>& arg);
template <typename T>
[[nodiscard]] std::string FormatKernelArg(const std::optional<T>& arg);
template <typename T>
[[nodiscard]] std::string FormatKernelArg(const std::vector<T>& arg);
template <typename T, std::size_t N>
std::string FormatKernelArg(const std::array<T, N>& arg) {
  return absl::StrCat("[",
                      absl::StrJoin(arg, ", ",
                                    [](std::string* out, const T& elem) {
                                      absl::StrAppend(out,
                                                      FormatKernelArg(elem));
                                    }),
                      "]");
}
template <typename T>
std::string FormatKernelArg(const std::optional<T>& arg) {
  return arg.has_value() ? std::string(FormatKernelArg(*arg))
                         : std::string("nullopt");
}
template <typename T>
std::string FormatKernelArg(const std::vector<T>& arg) {
  if constexpr (std::is_same_v<T, at::Tensor>) {
    return absl::StrCat(
        "[\n  ",
        absl::StrJoin(arg, "\n  ",
                      [](std::string* out, const at::Tensor& elem) {
                        absl::StrAppend(out, FormatKernelArg(elem));
                      }),
        "\n]");
  } else {
    return absl::StrCat("[",
                        absl::StrJoin(arg, ", ",
                                      [](std::string* out, const T& elem) {
                                        absl::StrAppend(out,
                                                        FormatKernelArg(elem));
                                      }),
                        "]");
  }
}

template <typename T>
std::string FormatKernelArg(const c10::List<T>& arg) {
  return FormatKernelArg(arg.vec());  // VEC_OK
}

// Variadic function LogKernelArgs() logs the arguments one by one.

// Base case: 0 arguments to log.
inline void LogKernelArgs(std::ostream& ss,
                          const absl::Span<const std::string_view> arg_names) {
  ABSL_CHECK(arg_names.empty())  // CRASH_OK
      << "INTERNAL: LogKernelArgs called with too many arg_names.";
}

// Recursive case: log the next argument and then call the function recursively
// with the remaining arguments.
template <typename T, typename... Args>
inline void LogKernelArgs(std::ostream& ss,
                          const absl::Span<const std::string_view> arg_names,
                          const T& arg, const Args&... rest_args) {
  ABSL_CHECK(!arg_names.empty())  // CRASH_OK
      << "INTERNAL: LogKernelArgs called with too few arg_names.";
  ss << arg_names.front() << ": " << FormatKernelArg(arg) << "\n";
  LogKernelArgs(ss, arg_names.subspan(1), rest_args...);
}

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_LOGGING_H_
