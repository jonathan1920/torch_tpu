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

#ifndef TORCH_TPU_COMMON_CONTEXT_STATES_H_
#define TORCH_TPU_COMMON_CONTEXT_STATES_H_

#include <map>
#include <optional>
#include <string>

#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

// This library defines the context states for different types of context
// managers. By convention, the context state type is denoted by appending
// `ContextState` to the name of the context manager.

// The state of the `precision` context manager.
using PrecisionContextState = mlir::stablehlo::Precision;

// Whether to capture tracebacks for each dispatched op.
enum class TracebackMode {
  // Tracebacks are not captured. This is the default for eager mode, where MLIR
  // locations are not typically needed.
  kDisabled,
  // Tracebacks are captured for each dispatched op. This is the default for
  // FX compiled/export modes
  kEnabled,
};

// The state of the `enable_tracebacks` context manager.
using EnableTracebacksContextState = std::optional<TracebackMode>;

// Maps an XLA compiler option name to its string value. We pick this
// representation for easy interop with Python.
using CompilerOptionOverrides = std::map<std::string, std::string>;

// The state of the `custom_compiler_options` context manager.
using CustomCompilerOptionsContextState = CompilerOptionOverrides;

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTEXT_STATES_H_
