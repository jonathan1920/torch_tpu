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

#include "torch_tpu/common/compile_options_key.h"

#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compilation_test_helper.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {

CompileOptionsKey GetCompileOptionsKey(std::string_view xla_flags,
                                       const xla::CompileOptions& options);

namespace {

class CompileOptionsKeyTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before `MakeCompilerOptions` is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

absl::StatusOr<UniqueCompileOptions> MakeCompilerOptions(CompilationMode mode) {
  TT_ASSIGN_OR_RETURN(CompilationSpecsByMode specs, MakeCompilationSpecs(mode));
  return std::move(specs.at(mode).xla_compile_options);
}

[[nodiscard]] std::string GetXlaFlags() {
  return GetEnvOnce<kXlaFlagsEnvVar>().value_or("");
}

constexpr FingerprintType kFastCompileDefaultFingerprint =
    11255405118680073921ULL;
constexpr FingerprintType kFastRuntimeDefaultFingerprint =
    10955459558934170598ULL;

TEST_F(CompileOptionsKeyTest, StableFingerprint) {
  {
    TF_ASSERT_OK_AND_ASSIGN(const UniqueCompileOptions options,
                            MakeCompilerOptions(CompilationMode::kFastCompile));

    const CompileOptionsKey key = GetCompileOptionsKey(GetXlaFlags(), *options);
    EXPECT_EQ(key.key(), kFastCompileDefaultFingerprint)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(const UniqueCompileOptions options,
                            MakeCompilerOptions(CompilationMode::kFastRuntime));

    const CompileOptionsKey key = GetCompileOptionsKey(GetXlaFlags(), *options);
    EXPECT_EQ(key.key(), kFastRuntimeDefaultFingerprint)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }
}

TEST_F(CompileOptionsKeyTest,
       DifferentCompilationModesYieldDifferentFingerprints) {
  EXPECT_NE(kFastCompileDefaultFingerprint, kFastRuntimeDefaultFingerprint);
}

TEST_F(CompileOptionsKeyTest, XlaExecutionEffortLevelOverrides) {
  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O3"}});
    TF_ASSERT_OK_AND_ASSIGN(const UniqueCompileOptions options,
                            MakeCompilerOptions(CompilationMode::kFastCompile));

    const CompileOptionsKey key = GetCompileOptionsKey(GetXlaFlags(), *options);
    EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
    EXPECT_EQ(key.key(), 14056296875166197543ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  {
    ScopedCompilerOptionOverrides overrides(
        {{"xla_memory_fitting_level", "O1"}});
    TF_ASSERT_OK_AND_ASSIGN(const UniqueCompileOptions options,
                            MakeCompilerOptions(CompilationMode::kFastCompile));

    const CompileOptionsKey key = GetCompileOptionsKey(GetXlaFlags(), *options);
    EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
    EXPECT_EQ(key.key(), 12100832961370432928ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }
}

TEST_F(CompileOptionsKeyTest, EnvOptionOverrides) {
  ScopedCompilerOptionOverrides overrides(
      {{"xla_tpu_enable_deduplicated_calls", "DISABLED"}});
  TF_ASSERT_OK_AND_ASSIGN(const UniqueCompileOptions options,
                          MakeCompilerOptions(CompilationMode::kFastCompile));

  const CompileOptionsKey key = GetCompileOptionsKey(GetXlaFlags(), *options);
  EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
  EXPECT_EQ(key.key(), 9003441201055756418ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass "
         "unless the code changes how `xla::CompileOptions` is fingerprinted.";
}

}  // namespace
}  // namespace torch_tpu
