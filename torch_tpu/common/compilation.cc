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

#include "torch_tpu/common/compilation.h"

#include <cstdlib>
#include <memory>
#include <stack>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_init.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/translate/register.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/xla.pb.h"

namespace torch_tpu {

// Names of special XLA compiler options.
constexpr std::string_view kOptimizationLevelOption = "xla_optimization_level";
constexpr std::string_view kMemoryFittingLevelOption =
    "xla_memory_fitting_level";

ContextedModule::ContextedModule()
    : context_(std::make_unique<mlir::MLIRContext>()) {
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context_->appendDialectRegistry(registry);
  context_->loadAllAvailableDialects();
}  // NOLINT - module_ will be set in the Make() factory.

absl::StatusOr<ContextedModule> ContextedModule::Make(
    const MlirComputationBuilder& computation_builder) {
  ContextedModule contexted_module;
  TT_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> computation,
                      computation_builder(*contexted_module.context_));
  contexted_module.module_ =
      std::make_unique<mlir::OwningOpRef<mlir::ModuleOp>>(
          std::move(computation));
  return std::move(contexted_module);
}

absl::StatusOr<SharedLoadedExecutable> Compile(
    xla::PjRtClient& client, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                      executable_builder(client, std::move(compile_options)));
  TT_RET_CHECK(executable, error::kInternal)
      << "compilation succeeded but returned a null executable.";
  return executable;
}

absl::StatusOr<LoadedExecutableBuilder>
MlirComputationBuilderToExecutableBuilder(
    const MlirComputationBuilder& computation_builder) {
  TT_ASSIGN_OR_RETURN(ContextedModule contexted_module,
                      ContextedModule::Make(computation_builder));
  return [contexted_module = std::move(contexted_module)](
             xla::PjRtClient& client, UniqueCompileOptions options)
             -> absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> {
    return client.CompileAndLoad(contexted_module.get(), std::move(*options));
  };
}

// Returns the compile option overrides for the current thread as a stack.
// Each element in the stack represents a set of overrides that have been
// pushed but not yet popped. Each time we push a set of overrides, it is
// merged with the existing overrides (later pushes take precedence).
//
// For example, given Python code:
//
//  with compiler.custom_compiler_options({
//      "foo": "1",
//      "bar": "2",
//  }):
//    f().to("cpu")  // triggers compilation
//    with compiler.custom_compiler_options({
//      "bar": "3",
//      "baz": "4",
//    }):
//      g().to("cpu")  // triggers compilation
//    h().to("cpu")  // triggers compilation
//
// 1. initially the stack is empty,
// 2. when we are in the outer custom_compiler_options context but not in
//    the inner one, the stack contains one element
//      { "foo": "1", "bar": "2" },
// 3. when we are in the inner custom_compiler_options context, the stack
//    contains two elements:
//      (top)    { "foo": "1", "bar": "3", "baz": "4" }
//      (bottom) { "foo": "1", "bar": "2" }
//
// Therefore f() and h() will be compiled with { "foo": "1", "bar": "2" },
// and g() will be compiled with { "foo": "1", "bar": "3", "baz": "4" }.
[[nodiscard]] static std::stack<CompilerOptionOverrides>&
GetMutableCompileOptionOverridesStack() {
  // User PyTorch code may set different overrides in different Python
  // threads, so this needs to be thread-local.
  static thread_local absl::NoDestructor<std::stack<CompilerOptionOverrides>>
      overrides;
  return *overrides;
}

// Returns the compile option overrides for the current thread, as set in
// the user Python code. Thread-safe.
[[nodiscard]] static CompilerOptionOverrides
GetCompilerOptionOverridesFromPython() {
  const auto& stack = GetMutableCompileOptionOverridesStack();
  if (stack.empty()) {
    return {};
  }
  return stack.top();
}

// Updates `map` with the contents of `updates`. If a key appears in both `map`
// and `updates`, the value from `updates` is used.
template <typename Map>
static void UpdateMap(Map& map, Map updates) {
  for (auto& [key, value] : updates) {
    map[std::move(key)] = std::move(value);
  }
}

void PushCompilerOptionOverrides(CompilerOptionOverrides overrides) {
  // When we push the overrides, the new overrides are merged with the existing
  // overrides. The innermost map takes precedence.
  auto merged_overrides = GetCompilerOptionOverridesFromPython();
  UpdateMap(merged_overrides, std::move(overrides));
  GetMutableCompileOptionOverridesStack().push(std::move(merged_overrides));
}

void PopCompilerOptionOverrides() {
  auto& stack = GetMutableCompileOptionOverridesStack();
  ABSL_CHECK(!stack.empty())  // CRASH_OK
      << "Attempted to pop empty compiler option overrides stack. This is a "
         "bug in TorchTPU.";
  stack.pop();
}

static absl::StatusOr<xla::ExecutionOptions::EffortLevel> ParseEffortLevel(
    std::string_view level) {
  if (level == "O0") {
    return xla::ExecutionOptions::EFFORT_O0;
  }
  if (level == "O1") {
    return xla::ExecutionOptions::EFFORT_O1;
  }
  if (level == "O2") {
    return xla::ExecutionOptions::EFFORT_O2;
  }
  if (level == "O3") {
    return xla::ExecutionOptions::EFFORT_O3;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "unknown XLA compiler effort level: " << level;
}

// Returns the compile option overrides from the environment variable
// TORCH_TPU_INTERNAL_XLA_OPTIONS, in the format of "key1=value1 key2=value2
// ...".
//
// This function is memoized, so the environment variable is only read once.
static const CompilerOptionOverrides& GetCompilerOptionOverridesFromEnvVar() {
  static const absl::NoDestructor<CompilerOptionOverrides> overrides([] {
    CompilerOptionOverrides overrides;
    const char* const xla_flags = getenv("TORCH_TPU_INTERNAL_XLA_OPTIONS");
    if (xla_flags != nullptr) {
      for (std::string_view flag : absl::StrSplit(xla_flags, ' ')) {
        if (flag.empty()) {
          continue;
        }
        std::pair<std::string, std::string> p =
            absl::StrSplit(flag, absl::MaxSplits('=', 1));
        overrides[p.first] = p.second;
      }
    }
    return overrides;
  }());
  return *overrides;
}

[[nodiscard]] EagerCompilationMode GetEagerCompilationMode() {
  const char* const env_var =
      getenv("TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE");
  if (env_var != nullptr && std::string_view(env_var) == "optimized") {
    return EagerCompilationMode::kOptimized;
  }
  return EagerCompilationMode::kFastCompile;
}

static absl::Status SetDefaultDeviceAssignment(
    xla::ExecutableBuildOptions& options) {
  TT_ASSIGN_OR_RETURN(const int num_devices, GetGlobalDeviceCount());
  // These options are sensible for 1 partition and 1 replica per device,
  // which aligns with typical single-host parallelism in native PyTorch.
  // Every device runs identical HLO and collectives are included explicitly.
  // We will revisit this as we explore multiple hosts, shardy, etc.
  options.set_num_replicas(num_devices);
  options.set_num_partitions(1);
  xla::DeviceAssignment da(num_devices, 1);
  for (int idx = 0; idx < num_devices; ++idx) {
    da(idx, 0) = idx;
  }
  options.set_device_assignment(da);
  return absl::OkStatus();
}

static absl::StatusOr<bool> SetTpuOptions(xla::CompileOptions& options) {
  const xla::PjRtClient* const client = GetPjRtClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized.";
  const bool is_tpu = client->platform_id() == xla::TpuId();
  if (is_tpu) {
    // WARNING: When assigning string values below, make sure to explicitly
    // pass std::string objects as opposed to using string literals, since the
    // C++ compiler used for the OSS version incorrectly casts const char* to
    // bool as opposed to building an std::string object.
    options.env_option_overrides = {
        // Reduces TPU binary size by using calls for deduplicated HLOs.
        {"xla_tpu_enable_deduplicated_calls", std::string("ENABLED")},
        // Enable "safe" XLA scavenge mode (where "safe" is needed by Pallas),
        // which helps with a bit for performance, but also helps kernels
        // which
        // specify their own scoped limit.
        {"xla_tpu_vmem_scavenging_mode", std::string("SAFE")},
        // Enable all known SparseCore offloading flags.
        {"xla_tpu_enable_offloading_gather_to_sparsecore", true},
        {"xla_tpu_enable_offloading_scatter_to_sparsecore", true},
        {"xla_tpu_enable_sparse_core_collective_offload_all_gather", true},
        {"xla_tpu_enable_sparse_core_collective_offload_2d_all_gather", true},
        {"xla_tpu_enable_sparse_core_collective_offload_reduce_scatter", true},
        {"xla_tpu_enable_sparse_core_reduce_scatter_v2", true},
        {"xla_tpu_enable_sparse_core_collective_offload_all_reduce", true},
        {"xla_tpu_enable_concurrent_sparse_core_offloading", true},
    };
  }
  return is_tpu;
}

static CompilerOptionOverrides MakeCompilerOptionOverrides(
    const bool is_tpu, const GraphCompilationMode mode) {
  CompilerOptionOverrides overrides;
  if (is_tpu && mode == GraphCompilationMode::kEager) {
    // Use O1 for the eager mode and the default optimization level (O2) for
    // the torch.compile mode.
    overrides[std::string(kOptimizationLevelOption)] = "O1";
    // We want lowering parameters to be chosen according to the optimization
    // level, rather than being pulled out of the autofdo database.
    // TODO(b/456145756): Remove this once we figure out the interaction
    // between autofdo and optimization level.
    overrides["xla_tpu_autofdo"] = "false";
  }
  UpdateMap(overrides, GetCompilerOptionOverridesFromPython());
  // When merging the overrides, the environment variable takes precedence.
  UpdateMap(overrides, GetCompilerOptionOverridesFromEnvVar());
  return overrides;
}

absl::Status ApplyCompilerOptionOverrides(
    const CompilerOptionOverrides& overrides,
    xla::CompileOptions& compile_options) {
  ABSL_VLOG(1) << "Setting XLA compiler option override: start";
  for (auto& [key, value] : overrides) {
    ABSL_VLOG(1) << "Setting XLA compiler option override: "
                 << absl::CEscape(key) << " = " << absl::CEscape(value);
    TT_RET_CHECK(!key.empty(), error::kInvalidArgument)
        << "XLA compiler option name must not be empty";
    if (key == kOptimizationLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_optimization_level(level);
    } else if (key == kMemoryFittingLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_memory_fitting_level(level);
    } else {
      // Apply a new override by replacing the entry with the same key, if
      // present, or by appending the new (key, value) pair at the end fo the
      // options.
      bool replaced = false;
      for (auto& p : compile_options.env_option_overrides) {
        if (p.first == key) {
          p.second = std::move(value);
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        compile_options.env_option_overrides.push_back(
            std::make_pair(std::move(key), std::move(value)));
      }
    }
  }
  ABSL_VLOG(1) << "Setting XLA compiler option override: end";
  return absl::OkStatus();
}

absl::StatusOr<UniqueCompileOptions> MakeCompilerOptions(
    GraphCompilationMode mode) {
  if (GetEagerCompilationMode() == EagerCompilationMode::kOptimized) {
    mode = GraphCompilationMode::kTorchCompile;
  }

  auto compile_options = std::make_unique<xla::CompileOptions>();
  // Call mutable_debug_options to parse XLA_FLAGS into compile options.
  compile_options->executable_build_options.mutable_debug_options();
  TT_RETURN_IF_ERROR(
      SetDefaultDeviceAssignment(compile_options->executable_build_options));

  TT_ASSIGN_OR_RETURN(const bool is_tpu, SetTpuOptions(*compile_options));

  // Finally, override the default flags if needed.
  TT_RETURN_IF_ERROR(ApplyCompilerOptionOverrides(
      MakeCompilerOptionOverrides(is_tpu, mode), *compile_options));
  return compile_options;
}

}  // namespace torch_tpu
