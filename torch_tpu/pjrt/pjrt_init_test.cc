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

#include "torch_tpu/pjrt/pjrt_init.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/pjrt/pjrt_shutdown.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

class PjRtInitTest : public ::testing::Test {
 protected:
  void TearDown() override { ShutdownPjRt(); }
};

TEST_F(PjRtInitTest, InitializePjRtRespectsWorldSize) {
  EXPECT_FALSE(IsPjRtInitialized());

  PjRtInitializationOptions options = {
      .device_type = "xla_cpu",
      .world_size = 1,
  };

  ASSERT_OK_AND_ASSIGN(PjRtInitializationResult result,
                       InitializePjRt(options));
  EXPECT_TRUE(IsPjRtInitialized());

  // CPU backend usually has 1 device by default anyway,
  // but we can at least assert the return value matches what we asked for.
  EXPECT_EQ(result.device_count, 1);
  EXPECT_EQ(GetGlobalDeviceCount().value(), 1);
  EXPECT_EQ(GetPjRtDeviceType(), PjRtDeviceType::kCpu);

  // Call it again to test the caching logic
  ASSERT_OK_AND_ASSIGN(PjRtInitializationResult result_2,
                       InitializePjRt(options));

  EXPECT_EQ(result_2.device_count, 1);
  // Ensure the second call returned the exact same thing
  EXPECT_EQ(result_2.global_device_id, result.global_device_id);
}

}  // namespace
}  // namespace torch_tpu
