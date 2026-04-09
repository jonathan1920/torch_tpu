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

#include "torch_tpu/_internal/profiler/tpu_profiler_plugin.h"

#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/IActivityProfiler.h>
#include "torch_tpu/common/utils.h"

namespace torch_tpu {
namespace {

#if TT_IS_INTERNAL_TORCH_TPU

TEST(TpuProfilerPluginTest, BasicSanity) {
  TpuProfiler profiler;

  // 1. Check name
  EXPECT_EQ(profiler.name(), "tpu_profiler");

  // 2. Check available activities
  const auto& activities = profiler.availableActivities();
  EXPECT_NE(activities.find(libkineto::ActivityType::PRIVATEUSE1_RUNTIME),
            activities.end());

  // 3. Configure session
  libkineto::Config config;
  std::set<libkineto::ActivityType> active_types = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  auto session = profiler.configure(active_types, config);
  ASSERT_NE(session, nullptr);

  // 4. Start and Stop
  session->start();
  session->stop();

  // We expect some errors since we are not running on a real TPU
  EXPECT_TRUE(session->errors().empty());
}

#endif  // TT_IS_INTERNAL_TORCH_TPU

}  // namespace
}  // namespace torch_tpu
