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

#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/pjrt/pjrt_init.h"
#include "xla/xla.pb.h"

ABSL_DECLARE_FLAG(int, torch_tpu_internal_num_compilation_threads);

namespace torch_tpu {

int GetNumCompilationThreads(int num_procs);

namespace {

CompilationCacheKey DummyKey() {
  return CompilationCacheKey(ShapelessKey{0}, DimensionsKey({}));
}

class GetNumCompilationThreadsTest : public testing::Test {};

TEST_F(GetNumCompilationThreadsTest,
       ReturnsHardwareConcurrencyWhenNprocNotSet) {
  EXPECT_EQ(GetNumCompilationThreads(0),
            std::thread::hardware_concurrency() - 1);
}

TEST_F(GetNumCompilationThreadsTest, ReturnsNprocTimesTwoWhenNprocSet) {
  EXPECT_EQ(GetNumCompilationThreads(10), 36);
}

TEST_F(GetNumCompilationThreadsTest, ReturnsFlagValueWhenSet) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
  EXPECT_EQ(GetNumCompilationThreads(), 42);
}

TEST_F(GetNumCompilationThreadsTest,
       ReturnsHardwareConcurrencyWhenFlagSetToZero) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 0);
  EXPECT_EQ(GetNumCompilationThreads(0),
            std::thread::hardware_concurrency() - 1);
}

TEST_F(GetNumCompilationThreadsTest, FlagTakesPrecedenceOverNproc) {
  absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
  EXPECT_EQ(GetNumCompilationThreads(10), 42);
}

TEST(PerfStatsPrinterTest, EmptyPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  EXPECT_EQ(absl::StrCat(stats),
            "num_cache_reqs=10\nnum_cache_hits=5 {50.0%}\n");
}

TEST(PerfStatsPrinterTest, WithPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  stats.per_entry_stats.push_back({
      {.compilation_duration = absl::Milliseconds(100),
       .last_read = absl::FromUnixMillis(1000),
       .read_count = 10},
      DummyKey(),
  });
  stats.per_entry_stats.push_back({
      {.compilation_duration = absl::Milliseconds(50),
       .last_read = absl::FromUnixMillis(2000),
       .read_count = 5},
      DummyKey(),
  });
  // NOLINTBEGIN
  static constexpr std::string_view kExpected = R"(num_cache_reqs=10
num_cache_hits=5 {50.0%}
CompilationCacheKey{shapeless_key=0, dimensions_key=5825f5f3bd962979}{
	compilation_duration=100ms,
	last_read=1969-12-31T16:00:01-08:00,
	read_count=10,
}
CompilationCacheKey{shapeless_key=0, dimensions_key=5825f5f3bd962979}{
	compilation_duration=50ms,
	last_read=1969-12-31T16:00:02-08:00,
	read_count=5,
}
num_compilation_events=2
sum_compilation_time=150ms
)";
  // NOLINTEND
  EXPECT_EQ(absl::StrCat(stats), kExpected);
}

TEST(CacheEntryStatsPrinterTest, Works) {
  CacheEntryStats stats;
  stats.compilation_duration = absl::Milliseconds(100);
  stats.last_read = absl::FromUnixMillis(1000);
  stats.read_count = 10;
  EXPECT_EQ(absl::StrCat(stats),
            "{\n\tcompilation_duration=100ms,\n\tlast_read=1969-12-31T16:00:01-"
            "08:00,\n\tread_count=10,\n}");
}

TEST(CompilationCacheTest, DumpOnMissMode) {
  CompilationCache& cache = CompilationCache::GetInstance();
  bool initial_mode = cache.GetDumpOnCacheMissMode();
  cache.SetDumpOnCacheMissMode(!initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), !initial_mode);
  cache.SetDumpOnCacheMissMode(initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), initial_mode);
}

TEST(CompilationCacheTest, GetOrCompileLogsOnMiss) {
  CompilationCache& cache = CompilationCache::GetInstance();
  bool initial_mode = cache.GetDumpOnCacheMissMode();
  cache.SetDumpOnCacheMissMode(true);

  // Trigger a miss with a unique key.
  CompilationCacheKey key(ShapelessKey{12345}, DimensionsKey({}));
  std::vector<Shape> input_shapes;

  MlirComputationBuilder builder = [](mlir::MLIRContext& context) {
    return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  };

  // We need to initialize PjRt to make MakeCompilerOptions work.
  // It's safe to call this multiple times; it will return early if already
  // initialized. Use xla_cpu for unit testing as it doesn't require real
  // hardware.
  absl::Status pjrt_status =
      InitializePjRt({.device_type = "xla_cpu", .world_size = 1}).status();

  auto options_or = MakeCompilerOptions(CompilationMode::kFastCompile);
  ASSERT_TRUE(options_or.ok())
      << "MakeCompilerOptions failed: " << options_or.status()
      << " (PjRt status: " << pjrt_status << ")";

  absl::ScopedMockLog log;
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, testing::_,
          testing::HasSubstr("Dumping StableHLO module due to cache miss")));

  log.StartCapturingLogs();
  auto result = cache.GetOrCompile(key, input_shapes, std::move(builder),
                                   std::move(*options_or));

  // Clean up.
  cache.SetDumpOnCacheMissMode(initial_mode);
}

}  // namespace
}  // namespace torch_tpu
