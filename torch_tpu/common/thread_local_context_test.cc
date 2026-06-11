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

#include "torch_tpu/common/thread_local_context.h"

#include <memory>
#include <thread>
#include <utility>

#include "c10/util/ThreadLocalDebugInfo.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

constexpr int kTestValue = 42;

class TestDebugInfo : public c10::DebugInfoBase {
 public:
  explicit TestDebugInfo(int value) : value_(value) {}
  int value() const { return value_; }

 private:
  int value_;
};

[[nodiscard]] TestDebugInfo* GetTestDebugInfo() {
  return dynamic_cast<TestDebugInfo*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO));
}

TEST(ThreadLocalContext, DoesNotPropagateAcrossThreadsWhenNotApplied) {
  auto info = std::make_shared<TestDebugInfo>(kTestValue);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, info);

  std::thread t([]() { EXPECT_EQ(GetTestDebugInfo(), nullptr); });
  t.join();
}

TEST(ThreadLocalContext, PropagatesThreadLocalStateWhenApplied) {
  auto info = std::make_shared<TestDebugInfo>(kTestValue);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, info);

  auto context = ThreadLocalContext::Capture();
  std::thread t([ctx = std::move(context)]() {
    ctx.Apply([]() {
      auto* info = GetTestDebugInfo();
      ASSERT_NE(info, nullptr);
      EXPECT_EQ(info->value(), kTestValue);
    });
  });
  t.join();
}

}  // namespace
}  // namespace torch_tpu
