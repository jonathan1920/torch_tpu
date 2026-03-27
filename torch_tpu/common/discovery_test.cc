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
#include "torch_tpu/common/discovery.h"

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/env_vars.h"

namespace torch_tpu {
namespace {

TEST(DiscoveryTest, GetPremappedBufferSizeWithDefaultReturnsValue) {
  unsetenv(kTpuPremappedBufferSizeEnvVar);

  const auto& status_or_size = GetPremappedBufferSizeFromEnvOnce();
  ASSERT_TRUE(status_or_size.ok()) << status_or_size.status();
  EXPECT_EQ(*status_or_size, 0);
  EXPECT_EQ(std::string(getenv(kTpuPremappedBufferSizeEnvVar)), "0");
}

// TODO: determine how to un-memoize environment variable reads for testing.
TEST(DiscoveryTest, DISABLED_GetPremappedBufferSizeWithDefaultReadsEnv) {
  setenv(kTpuPremappedBufferSizeEnvVar, "1024", 1);

  const auto& status_or_size = GetPremappedBufferSizeFromEnvOnce();
  ASSERT_TRUE(status_or_size.ok()) << status_or_size.status();
  EXPECT_EQ(*status_or_size, 1024);
}

}  // namespace
}  // namespace torch_tpu
