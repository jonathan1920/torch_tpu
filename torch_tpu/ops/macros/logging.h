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

#include <optional>
#include <ostream>
#include <set>
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
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// In dbg mode, checks that the arguments passed to TT_KERNEL() match the
// arguments for the kernel function that encloses the TT_KERNEL() call.
// Also, at verbosity level 1, logs the arguments to a C++ kernel, along with
// their variable names. Example:
//
//    TT_CHECK_AND_LOG_KERNEL_ARGS_("foo", lhs, rhs, value, dims, predicate);
//
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
#define TT_CHECK_AND_LOG_KERNEL_ARGS_(op_name, ...) \
  TT_CHECK_AND_LOG_KERNEL_ARGS_IMPL_(op_name __VA_OPT__(, ) __VA_ARGS__)

// Implements TT_CHECK_AND_LOG_KERNEL_ARGS_. The redirection is needed to allow
// TT_CHECK_AND_LOG_KERNEL_ARGS_() to be used in a macro (e.g. TT_KERNEL).
//
// Implementation note: __VA_OPT__(,) expands to nothing if the variadic
// argument list is empty, and expands to a comma otherwise.
#define TT_CHECK_AND_LOG_KERNEL_ARGS_IMPL_(op_name, ...)                       \
  do {                                                                         \
    static_assert(::torch_tpu::internal::ArgsAreIdentifiers(#__VA_ARGS__),     \
                  "The op parameter list passed to TT_KERNEL() must contain "  \
                  "only identifier names.");                                   \
    if (ABSL_VLOG_IS_ON(1)) {                                                  \
      std::ostringstream _ss;                                                  \
      ::torch_tpu::internal::LogKernelName(_ss, op_name);                      \
      const std::vector<std::string_view> _arg_names =                         \
          TT_ARGS_AS_STRINGS_(__VA_ARGS__);                                    \
      ::torch_tpu::internal::LogKernelArgs(                                    \
          _ss, _arg_names __VA_OPT__(, ) __VA_ARGS__);                         \
      ABSL_LOG(INFO) << _ss.str();                                             \
    }                                                                          \
    if constexpr (::torch_tpu::internal::kDebugMode) {                         \
      /* Only check the argument types once per kernel function. */            \
      constexpr std::string_view _func_sig = __PRETTY_FUNCTION__;              \
      static const bool _checked = [&] {                                       \
        ::torch_tpu::internal::CheckTtKernelArgList(                           \
            ::torch_tpu::internal::KernelArgCheckerContext{                    \
                __FILE__, __LINE__, _func_sig,                                 \
                ::torch_tpu::internal::ParseArgTypesOrEmpty(_func_sig),        \
                TT_ARGS_AS_STRINGS_(__VA_ARGS__)} __VA_OPT__(, ) __VA_ARGS__); \
        return true;                                                           \
      }();                                                                     \
      static_cast<void>(_checked); /* Avoid "unused variable" warning. */      \
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

// Parses and normalizes the argument types from a C++ function signature
// generated by __PRETTY_FUNCTION__. Returns an empty vector if parsing fails.
[[nodiscard]] std::vector<std::string> ParseArgTypesOrEmpty(
    std::string_view func_sig);

// Information needed for checking kernel argument types.
struct KernelArgCheckerContext {
  std::string_view file;  // Path to the file where TT_KERNEL() is called.
  int line = -1;          // Line number of the TT_KERNEL() call.
  // Signature of the kernel function, as generated by __PRETTY_FUNCTION__.
  std::string_view func_sig;
  // Argument types of the kernel function, as parsed and normalized from
  // func_sig.
  std::vector<std::string> arg_types;
  // Argument names passed to TT_KERNEL().
  std::vector<std::string_view> tt_kernel_arg_names;
};

// Crashes if the type T does not match the given argument's type string.
template <typename T>
void CheckKernelArgType(const KernelArgCheckerContext& context,
                        const int arg_idx) {
  const std::string_view arg_name = context.tt_kernel_arg_names[arg_idx];
  const std::string_view normalized_arg_type_in_func_sig =
      context.arg_types[arg_idx];
  const auto message = [&]() {
    return absl::StrCat("\n", context.file, ":", context.line,
                        ": INTERNAL: expected the '", arg_name,
                        "' argument in TT_KERNEL()'s inner parenthesized list "
                        "to have normalized type ",
                        normalized_arg_type_in_func_sig,
                        " to match the kernel function signature");
  };
  if constexpr (std::is_same_v<T, at::Tensor>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::Tensor")
        << message();
  } else if constexpr (std::is_same_v<T, at::Scalar>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::Scalar")
        << message();
  } else if constexpr (std::is_same_v<T, int>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "int")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, std::string>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "std::string")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, at::OptionalIntArrayRef>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::OptionalIntArrayRef")
        << message();
  } else if constexpr (std::is_same_v<T, at::TensorList>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::TensorList")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<c10::ScalarType>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::ScalarType>")
        << message();
  } else {
    // TODO: Check that T and arg_type match for other types.
  }
}

// Variadic function CheckKernelArgTypes() validates the arguments one by
// one.
//
// Base case: 0 arguments to validate.
inline void CheckKernelArgTypes(const KernelArgCheckerContext& context,
                                const int arg_idx) {}

// Recursive case: validate the next argument and then call the function
// recursively with the remaining arguments.
template <typename T, typename... Args>
inline void CheckKernelArgTypes(const KernelArgCheckerContext& context,
                                const int arg_idx, const T& arg,
                                const Args&... rest_args) {
  CheckKernelArgType<T>(context, arg_idx);
  CheckKernelArgTypes(context, arg_idx + 1, rest_args...);
}

// Checks that the arguments passed to TT_KERNEL() match the arguments for the
// kernel function that encloses the TT_KERNEL() call.
template <typename... Args>
inline void CheckTtKernelArgList(const KernelArgCheckerContext& context,
                                 const Args&... args) {
  // 1. Check the number of arguments.
  const int num_args = sizeof...(args);
  ABSL_CHECK_EQ(context.arg_types.size(), num_args)  // CRASH_OK
      << "\n"
      << context.file << ":" << context.line
      << ": TT_KERNEL() called with mismatching number of arguments.\n"
      << "Function " << context.func_sig << " has " << context.arg_types.size()
      << " arguments:\n  " << absl::StrJoin(context.arg_types, ",\n  ")
      << "\nbut " << num_args << " arguments were passed to TT_KERNEL():\n  "
      << absl::StrJoin(context.tt_kernel_arg_names, ",\n  ");

  // 2. Check that the arguments contain no duplicates.
  std::set<std::string_view> arg_names_seen;
  for (const std::string_view arg_name : context.tt_kernel_arg_names) {
    ABSL_CHECK(  // CRASH_OK
        arg_names_seen.find(arg_name) == arg_names_seen.end())
        << "\n"
        << context.file << ":" << context.line << ": argument '" << arg_name
        << "' was passed to TT_KERNEL() multiple times.";
    arg_names_seen.insert(arg_name);
  }

  // 3. Check that the argument types match the kernel function signature.
  CheckKernelArgTypes(context, 0, args...);
}

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_LOGGING_H_
