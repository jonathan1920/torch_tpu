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

#ifndef TORCH_TPU_OPS_MACROS_KERNEL_H_
#define TORCH_TPU_OPS_MACROS_KERNEL_H_

#include <string_view>

#include "absl/status/status.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/logging.h"

// TT_KERNEL(op_name, param_keys, (arg1, arg2, ...), statements) wraps the
// implementation of an op kernel and takes care of logging, capturing the
// Python context, computing parameter cache keys, and attributing errors
// to the correct root op.
//
// Arguments:
//   op_name: The name of the op. E.g. OpName::kAddOut.
//   param_keys: Name of a new variable that will hold the op parameter cache
//       keys. TT_KERNEL() computes the cache keys by going through all
//       non-Tensor-typed parameters in (arg1, arg2, ...) and adding their
//       cache keys. The name `_` is special and means that computing parameter
//       cache keys is skipped (hence the variable `_` will hold an empty
//       OpParamCacheKeys object). This is useful for avoiding the cost of
//       computing the parameter cache keys when they are not needed or when
//       we want to compute them ourselves.
//   (arg1, arg2, ...): The op kernel function's exact parameter list. Always
//       pass *all* parameters of the kernel function exactly as they are,
//       including both tensors and non-tensors. Do not omit any parameters
//       or pass extra parameters.
//   statements: The actual logic of the op kernel. It can reference
//       `param_keys` to access the op parameter cache keys.
//
// Requirements:
//    - OpParamCacheKeys::Builder knows how to compute the cache keys for
//      all argument types. If you get a compiler error that
//      FormatParamCacheKey() cannot be resolved, add an overload of
//      FormatParamCacheKey() for that type.
//    - All argument types are "loggable". See the comments for
//      FormatKernelArg() in logging.h for details.
//
// See go/torch-tpu-op-easy for more details.
//
// Implementation note: the last argument ... is for the statements. We use
// a variadic macro parameter as the statements may contain unprotected commas.

#define TT_KERNEL(op_name, param_keys, args, ...)                             \
  do {                                                                        \
    TT_LOG_KERNEL_START_(op_name, TT_REMOVE_PARENS_(args));                   \
    ::torch_tpu::ScopedPythonContextCapturer _capturer(op_name);              \
    if constexpr (std::string_view(#param_keys) == "_") {                     \
      /* Use InvalidOpParamCacheKeys as the type of _, to prevent using it in \
       * ...                                                                  \
       */                                                                     \
      ::torch_tpu::internal::InvalidOpParamCacheKeys param_keys;              \
      try {                                                                   \
        __VA_ARGS__;                                                          \
      } catch (const ::torch_tpu::TtError& e) {                               \
        ::torch_tpu::TranslateToC10ErrorAndThrow(e);                          \
      }                                                                       \
    } else {                                                                  \
      TT_ASSIGN_OR_THROW(/* ERROR_COV_INFEASIBLE=in macro definition. */      \
                         ::torch_tpu::OpParamCacheKeys param_keys,            \
                         TT_MAKE_OP_PARAM_CACHE_KEYS args);                   \
      try {                                                                   \
        __VA_ARGS__;                                                          \
      } catch (const ::torch_tpu::TtError& e) {                               \
        ::torch_tpu::TranslateToC10ErrorAndThrow(e);                          \
      }                                                                       \
    }                                                                         \
  } while (false)

// Invokes MakeOpParamCacheKeys() with the given arguments, with the
// stringified argument list as the first argument. E.g.
//   TT_MAKE_OP_PARAM_CACHE_KEYS(a, b, c)
// expands to
//   MakeOpParamCacheKeys("a, b, c", a, b, c)
// while
//   TT_MAKE_OP_PARAM_CACHE_KEYS()
// expands to
//   MakeOpParamCacheKeys("")
//
// Implementation note: the __VA_OPT__(,) expands to nothing if the argument
// list is empty, and expands to a comma otherwise.
#define TT_MAKE_OP_PARAM_CACHE_KEYS(...)                                  \
  ::torch_tpu::internal::MakeOpParamCacheKeys(#__VA_ARGS__ __VA_OPT__(, ) \
                                                  __VA_ARGS__)

namespace torch_tpu {
namespace internal {

// A type that can't be used to do anything useful.
//
// We use this type for the `_` param_keys variable in TT_KERNEL() to prevent
// the user from accidentally using it in the body of the kernel, which is
// a programmer error.
class InvalidOpParamCacheKeys {
 public:
  InvalidOpParamCacheKeys() = default;

  // Allow implicit conversion to OpParamCacheKeys to compile.
  // These functions have no implementation as they should only be used in
  // inactive code paths. E.g.
  //   TT_KERNEL(OpName::kAdd, param_keys, (size), {
  //     ... std::move(param_keys) ...
  //   });
  // expands to
  //   ...
  //   if constexpr (std::string_view("param_keys") == "_") {
  //     InvalidOpParamCacheKeys param_keys;
  //     ... std::move(param_keys) ...  // block A
  //   } else {
  //     TT_ASSIGN_OR_THROW(OpParamCacheKeys param_keys, ...);
  //     ... std::move(param_keys) ...  // block B
  //   }
  //   ...
  // Since the constexpr condition is false, block A is inactive, and thus
  // the code for InvalidOpParamCacheKeys doesn't need to be linked.
  template <typename T>
  absl::Status SetParam(std::string_view, const T&);
  operator OpParamCacheKeys&() &;    // NOLINT
  operator OpParamCacheKeys&&() &&;  // NOLINT
};

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_KERNEL_H_
