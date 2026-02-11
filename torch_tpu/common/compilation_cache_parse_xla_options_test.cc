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


#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "torch_tpu/common/compilation.h"
#include "xla/xla.pb.h"
#include "torch_tpu/pjrt/pjrt_init.h"

namespace torch_tpu {
namespace {

using testing::Contains;
using testing::Pair;

class MakeCompilerOptionsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    ABSL_CHECK_OK(InitializePjRt({.device_type = "tpu", .world_size = 1}));
  }
};

TEST_F(MakeCompilerOptionsTest, ParsesXlaOptions) {
  setenv("TORCH_TPU_INTERNAL_XLA_OPTIONS",
         // Spaces are intentional to test parsing.
         " xla_optimization_level=O3  xla_tpu_enable_deduplicated_calls=AUTO ",
         1);
  absl::Cleanup cleanup = [&]() { unsetenv("TORCH_TPU_INTERNAL_XLA_OPTIONS"); };
  const auto options_or = MakeCompilerOptions(GraphCompilationMode::kEager);
  ASSERT_EQ(options_or.status(), absl::OkStatus());
  const auto& options = options_or.value();
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_O3);
  EXPECT_THAT(
      options->env_option_overrides,
      Contains(Pair("xla_tpu_enable_deduplicated_calls", std::string("AUTO"))));
}

}  // namespace
}  // namespace torch_tpu
