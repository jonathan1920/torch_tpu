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

#include "torch_tpu/distributed/slicebuilder/discovery.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/env_vars.h"

namespace torch_tpu {
namespace {

TEST(DiscoveryTest, GetDistributedWorkerConfiguration) {
  setenv(kRankEnvVar, "0", 1);
  setenv(kLocalRankEnvVar, "0", 1);
  setenv(kWorldSizeEnvVar, "2", 1);
  setenv(kMasterAddrEnvVar, "localhost", 1);
  setenv(kMasterPortEnvVar, "12345", 1);
  // Use standard BNS addresses with numeric ports.
  setenv(kTpuSlicebuilderAddressesEnvVar, "host0:54321,host1:54322", 1);
  setenv(kTpuTopologyEnvVar, "1x1x1", 1);

  auto config_or = GetDistributedWorkerConfiguration();
  ASSERT_TRUE(config_or.ok()) << config_or.status();
  EXPECT_EQ(config_or->rank, 0);
  EXPECT_EQ(config_or->sb_port, "54321");
  EXPECT_EQ(config_or->sb_addrs, "host0:54321,host1:54322");
}

}  // namespace
}  // namespace torch_tpu
