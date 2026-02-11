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

#include "torch_tpu/common/compilation_cache.h"

#include <cstdlib>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "torch_tpu/common/compilation.h"
#include "xla/xla.pb.h"
#include "torch_tpu/pjrt/pjrt_init.h"

ABSL_DECLARE_FLAG(int, torch_tpu_internal_num_compilation_threads);

namespace torch_tpu {
namespace {

class MakeCompilerOptionsTest : public testing::Test {
 protected:
  MakeCompilerOptionsTest() {
    // Before each test case, reset the eager compilation mode to the default
    // value.
    unsetenv("TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE");
  }

  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    ABSL_CHECK_OK(InitializePjRt({.device_type = "tpu", .world_size = 1}));
  }
};

TEST_F(MakeCompilerOptionsTest, DefaultToO1ForEagerMode) {
  const auto options_or = MakeCompilerOptions(GraphCompilationMode::kEager);
  ASSERT_EQ(options_or.status(), absl::OkStatus());
  const auto& options = options_or.value();
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_O1);
}

TEST_F(MakeCompilerOptionsTest, UseO2ForOptimizedEagerMode) {
  // Set the TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE environment variable to
  // "optimized".
  setenv("TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE", "optimized",
         /*overwrite=*/1);
  const auto options_or = MakeCompilerOptions(GraphCompilationMode::kEager);
  ASSERT_EQ(options_or.status(), absl::OkStatus());
  const auto& options = options_or.value();
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_UNKNOWN);
}

TEST_F(MakeCompilerOptionsTest, DefaultToUnsetForTorchCompileMode) {
  const auto options_or =
      MakeCompilerOptions(GraphCompilationMode::kTorchCompile);
  ASSERT_EQ(options_or.status(), absl::OkStatus());
  const auto& options = options_or.value();
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_UNKNOWN);
}

class GetNumCompilationThreadsTest : public testing::Test {
 protected:
  // Saves the original value of the environment variable NPROC. gtest already
  // saves/restores flags, so we don't need to do that.
  GetNumCompilationThreadsTest() : original_nproc_(std::getenv("NPROC")) {
    unsetenv("NPROC");
  }

  // Restores the original value of the environment variable.
  ~GetNumCompilationThreadsTest() override {
    if (original_nproc_ != nullptr) {
      setenv("NPROC", original_nproc_, 1);
    } else {
      unsetenv("NPROC");
    }
  }

 private:
  const char* const original_nproc_ = nullptr;
};

TEST_F(GetNumCompilationThreadsTest,
       ReturnsHardwareConcurrencyWhenNprocNotSet) {
  EXPECT_EQ(GetNumCompilationThreads(),
            std::thread::hardware_concurrency() - 1);
}

TEST_F(GetNumCompilationThreadsTest, ReturnsNprocTimesTwoWhenNprocSet) {
  setenv("NPROC", "10", 1);
  EXPECT_EQ(GetNumCompilationThreads(), 36);
}

TEST_F(GetNumCompilationThreadsTest, ReturnsFlagValueWhenSet) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
  EXPECT_EQ(GetNumCompilationThreads(), 42);
}

TEST_F(GetNumCompilationThreadsTest,
       ReturnsHardwareConcurrencyWhenFlagSetToZero) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 0);
  EXPECT_EQ(GetNumCompilationThreads(),
            std::thread::hardware_concurrency() - 1);
}

TEST_F(GetNumCompilationThreadsTest, FlagTakesPrecedenceOverNproc) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
  setenv("NPROC", "10", 1);
  EXPECT_EQ(GetNumCompilationThreads(), 42);
}

}  // namespace
}  // namespace torch_tpu
