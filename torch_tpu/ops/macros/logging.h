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

#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/common/to_string.h"
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
//   - All argument types are printable via ToString().
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
  ss << arg_names.front() << ": " << ToString(arg) << "\n";
  LogKernelArgs(ss, arg_names.subspan(1), rest_args...);
}

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_LOGGING_H_
