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

#ifndef TORCH_TPU_COMMON_COMPILATION_SPEC_H_
#define TORCH_TPU_COMMON_COMPILATION_SPEC_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "torch_tpu/common/compile_options_key.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// Mode for compiling a computation graph.
enum class CompilationMode {
  kFastCompile,  // Reduces compile time, but may result in slower execution.
  kFastRuntime,  // Produces more optimized executables, but with longer
                 // compile.
};

// `xla::CompileOptions` is a complex object with many fields. Even moving it
// is expensive. We wrap it in a unique_ptr to allow for cheap moving.
using UniqueCompileOptions = absl_nonnull std::unique_ptr<xla::CompileOptions>;

// Specifications to compile a computation graph, including
//   1. a fully resolved `xla::CompileOptions` to generate compiled kernels, and
//   2. its fingerprint used as part of compilation cache key.
struct CompilationSpec {
  CompilationSpec(UniqueCompileOptions options, CompileOptionsKey key)
      : xla_compile_options(std::move(options)),
        compile_options_key(std::move(key)) {}

  // Movable but not copyable.
  CompilationSpec(const CompilationSpec&) = delete;
  CompilationSpec& operator=(const CompilationSpec&) = delete;
  CompilationSpec(CompilationSpec&&) = default;
  CompilationSpec& operator=(CompilationSpec&&) = default;

  UniqueCompileOptions xla_compile_options;
  CompileOptionsKey compile_options_key;
};

using CompilationSpecsByMode = std::map<CompilationMode, CompilationSpec>;

// Maps an XLA compiler option name to its string value. We pick this
// representation for easy interop with Python.
using CompilerOptionOverrides = std::map<std::string, std::string>;

// Python thread-local context to manage currently effective XLA compile options
// along with its fingerprint.
//
// All op compilation tasks initiated by the same thread will reuse the cached
// `xla::CompileOptions`.
class CompilationContext {
 public:
  // Movable but not copyable.
  CompilationContext(const CompilationContext&) = delete;
  CompilationContext& operator=(const CompilationContext&) = delete;
  CompilationContext(CompilationContext&&) = default;
  CompilationContext& operator=(CompilationContext&&) = default;

  const CompilationSpecsByMode& compilation_specs() const {
    return compilation_specs_;
  }
  const CompilerOptionOverrides& compiler_option_overrides() const {
    return compiler_option_overrides_;
  }

 private:
  CompilationContext(CompilationSpecsByMode specs,
                     CompilerOptionOverrides overrides)
      : compilation_specs_(std::move(specs)),
        compiler_option_overrides_(std::move(overrides)) {}

  CompilationSpecsByMode compilation_specs_;
  CompilerOptionOverrides compiler_option_overrides_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_SPEC_H_
