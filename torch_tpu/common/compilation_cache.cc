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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <ios>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/thread_pool.h"
#include "torch_tpu/common/tier2_compilation_cache.h"
#include "torch_tpu/common/tier3_compilation_cache.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla.pb.h"
#include "tsl/platform/numbers.h"

ABSL_FLAG(int32_t, torch_tpu_internal_num_compilation_threads, 0,
          "The number of threads to use for compilation. If 0, use the default "
          "number of threads based on the number of logical CPUs.");

namespace torch_tpu {

absl::Mutex CompilationCache::g_mutex_(absl::kConstInit);

namespace {

// TODO(mvoz): move to utils
std::string PercAsStr(uint64_t num, uint64_t den) {
  if (den == 0) {
    return "NA";
  }
  auto k = num * 1000 / den;
  return absl::StrCat(k / 10, ".", k % 10, "%");
}

// Returns true if the future is ready.
bool IsFutureReady(const SharedLoadedExecutableFuture& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Returns the value of the NPROC environment variable (a positive integer), or
// 0 if it is not set or has an invalid value. This function is memoized, so
// the environment variable is only read once.
[[nodiscard]] int GetNumProcs() {
  static const int num_procs = [] {
    const auto& env_var = GetEnvOnce<kNprocEnvVar>();
    if (env_var.has_value()) {
      int num_proc;
      if (absl::SimpleAtoi(*env_var, &num_proc) && num_proc > 0) {
        return num_proc;
      }
    }
    return 0;
  }();
  return num_procs;
}

int GetNumCpus() { return sysconf(_SC_NPROCESSORS_ONLN); }

}  // namespace

// Returns the number of threads to use for compilation.
//
// `num_procs` is the value of the `NPROC` environment variable, or 0 if it is
// not set or has an invalid value.
int GetNumCompilationThreads(const int num_procs) {
  // Is the flag set?
  if (const int num_threads =
          absl::GetFlag(FLAGS_torch_tpu_internal_num_compilation_threads);
      num_threads != 0) {
    ABSL_CHECK_GE(num_threads, 0)  // CRASH_OK
        << "Invalid number of flag "
           "--torch_tpu_internal_num_compilation_threads: "
        << num_threads << ". Must be >= 0.";
    ABSL_VLOG(1) << "Using " << num_threads
                 << " compilation threads based on "
                    "--torch_tpu_internal_num_compilation_threads.";
    return num_threads;
  }

  int num_cpus;
  int num_hyperthreads_per_cpu;
  std::string_view method;
  // The flag is not set. Is NPROC set?
  if (num_procs > 0) {
    num_cpus = num_procs;
    num_hyperthreads_per_cpu = 4;
    method = "NPROC";
  } else {
    // NPROC is not set; use the hardware concurrency, which already accounts
    // for num_hyper-threading.
    num_cpus = GetNumCpus();
    num_hyperthreads_per_cpu =
        std::thread::hardware_concurrency() / GetNumCpus();
    method = "HW concurrency";
  }

  // Take out one entire CPU core so as to not interfere with the PyTorch main
  // thread. If the machine has only 1 CPU core, then use only 1 thread for the
  // compilation cache.
  int concurrency =
      (num_cpus > 1) ? std::max(1, num_cpus - 1) * num_hyperthreads_per_cpu : 1;

  ABSL_VLOG(1) << "Using " << concurrency << " compilation threads based on "
               << method;
  return concurrency;
}

int GetNumCompilationThreads() {
  return GetNumCompilationThreads(GetNumProcs());
}

void CompilationCache::Initialize(
    const CompilationCacheInitializationOptions& options) {
  ABSL_VLOG(1) << "InitializeCompilationCache: cache_only="
               << options.cache_only;
  auto& cache = CompilationCache::GetInstance();
  cache.SetCacheOnlyMode(options.cache_only);
}

CompilationCache::CompilationCache()
    : compilation_pool_(std::make_unique<ThreadPool>(
          // The actual thread name will be "tf_tt_compile" in logs.
          "tt_compile", GetNumCompilationThreads())),
      backup_compilation_pool_(
          UsesLocalBackupTaskForTier3Read()
              ? std::make_unique<ThreadPool>(
                    // The actual thread name will be "tf_tt_compile2" in
                    // logs. Unfortunately, we cannot use a longer, more
                    // descriptive name, as that will cause the thread name to
                    // be truncated in logs.
                    "tt_compile2",
                    // Use fewer threads for backup compilations.
                    std::max(1, compilation_pool_->NumThreads() / 2),
                    // Don't optimize for low latency - we want to leave room
                    // for the main compilation pool.
                    /*low_latency_hint=*/false)
              : nullptr) {}

CompilationCache::~CompilationCache() {
  ABSL_VLOG(1) << "CompilationCache shutting down.";
  compilation_pool_.reset();
  backup_compilation_pool_.reset();
  EvictAll();
  TT_MUTEX_LOCK(lock, cache_mutex_);
  ABSL_LOG(INFO) << "CompilationCache final stats: "
                 << "Requests=" << perf_stats_.num_cache_reqs
                 << ", Hits=" << perf_stats_.num_cache_hits << " ("
                 << PercAsStr(perf_stats_.num_cache_hits,
                              perf_stats_.num_cache_reqs)
                 << ")";
}

CompilationCache& CompilationCache::GetInstance() {
  TT_MUTEX_LOCK(lock, g_mutex_);
  static CompilationCache* const cache = new CompilationCache();
  return *cache;
}

void CompilationCache::Shutdown() { GetInstance().~CompilationCache(); }

// Sets the executable promise for the given key. If the promise is already
// satisfied, this function will do nothing.
static void TrySetExecutablePromise(
    CompilationCacheKey key, LoadedExecutablePromise& promise,
    const absl::StatusOr<SharedLoadedExecutable>& executable) {
  try {
    promise.set_value(std::move(executable));
    ABSL_VLOG(1) << "Set executable for key: " << key;
  } catch (const std::future_error& e) {
    // Check if it's the specific "already satisfied" error.
    if (e.code() == std::future_errc::promise_already_satisfied) {
      // This is expected, as another thread may have already compiled the
      // executable.
      ABSL_VLOG(1) << "Another thread already set the executable for key: "
                   << key;
    } else {
      ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
          << "Unexpected future_error: " << e.what();
    }
  }
}

void CompilationCache::EvictAll() {
  // Get all existing keys from the cache - these are the entries we need to
  // evict. We cannot promise to evict new entries created during the eviction
  // process, as that work may never end.
  absl::flat_hash_set<CompilationCacheKey, CompilationCacheKey::Hash>
      keys_to_evict;
  {
    TT_MUTEX_LOCK(lock, cache_mutex_);
    for (const auto& [key, cache_entry] : executable_cache_) {
      keys_to_evict.insert(key);
    }
  }

  // Keep evicting compiled keys until there are no more keys to evict,
  // waiting 1 second between iterations to give in-flight compilations a
  // chance to complete and update the cache.
  //
  // We chose a busy-wait loop as the alternative is much more complex.
  for (; !keys_to_evict.empty(); absl::SleepFor(absl::Seconds(1))) {
    TT_MUTEX_LOCK(lock, cache_mutex_);
    const auto keys = keys_to_evict;
    // Loop over a copy of keys_to_evict, as we will modify it during
    // the loop and invalidate its iterator.
    for (const auto key : keys) {
      const auto it = executable_cache_.find(key);
      if (it == executable_cache_.end()) {
        ABSL_VLOG(1) << "Key already evicted by another thread: " << key;
        continue;
      }

      const auto& cache_entry = it->second;
      ABSL_VLOG_EVERY_N(1, 100)
          << "Evicting compilation future for key: " << key;
      if (IsFutureReady(cache_entry.executable_future())) {
        // The compilation is completed, so we can evict it immediately.
        executable_cache_.erase(it);
        keys_to_evict.erase(key);
        ABSL_VLOG(1) << "Evicted compilation future for key: " << key;
      } else {
        ABSL_VLOG_EVERY_N(1, 100)
            << "Waiting for compilation to complete for key: " << key;
      }
    }
    // Release the lock here to give in-flight compilations a chance to
    // complete and update the cache.
  }

  ABSL_LOG(INFO) << "Compilation cache evicted.";
}

void CompilationCache::SetAllowCacheMode(bool allow) {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  allow_cache_mode_ = allow;
  ABSL_LOG(INFO) << "CompilationCache allow-cache mode set to: " << allow;
}

void CompilationCache::SetCacheOnlyMode(bool cache_only) {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  cache_only_mode_ = cache_only;
  ABSL_LOG(INFO) << "CompilationCache cache-only mode set to: " << cache_only;
}

int64_t CompilationCache::GetCacheRequests() const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  return perf_stats_.num_cache_reqs;
}

int64_t CompilationCache::GetCacheHits() const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  return perf_stats_.num_cache_hits;
}

int64_t CompilationCache::GetCacheMisses() const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  return perf_stats_.num_cache_misses();
}

PerfStats CompilationCache::GetCacheStats() const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  PerfStats stats = perf_stats_;
  stats.per_entry_stats.reserve(executable_cache_.size());
  for (const auto& [key, cache_entry] : executable_cache_) {
    stats.per_entry_stats.push_back({
        {cache_entry.stats()},
        key,
    });
  }
  return stats;
}

CompilationCache::CacheLookupInternal CompilationCache::LookupCacheEntry(
    CompilationCacheKey key) const {
  if (!allow_cache_mode_) {
    ABSL_LOG(WARNING) << "CompilationCache is disabled. key: " << key;
    return std::monostate();
  }
  perf_stats_.num_cache_reqs++;
  if (const auto it = executable_cache_.find(key);
      it != executable_cache_.end()) {
    perf_stats_.num_cache_hits++;
    it->second.stats().read_count++;
    it->second.stats().last_read = absl::Now();
    ABSL_VLOG(2) << "Compilation cache HIT for key: " << key;
    return &it->second;
  }

  if (const auto it = shape_dynamism_multimap_.find(key.shapeless_key);
      it != shape_dynamism_multimap_.end()) {
    if (ABSL_VLOG_IS_ON(2)) {
      const int num_metadata = it->second.size();
      for (auto i = 0; i < num_metadata; ++i) {
        ABSL_VLOG(2) << "[TtPerf] Compilation cache SHAPE-DYNAMIC HIT for key "
                     << key << " " << i + 1 << " of " << num_metadata;
      }
    }
    return absl::MakeConstSpan(it->second);
  }

  ABSL_VLOG(2) << "[TtPerf] Compilation cache MISS #"
               << perf_stats_.num_cache_misses() << " for key: " << key;
  return std::monostate();
}

std::vector<std::vector<ShapeDynamismMetadata>>
CompilationCache::GetShapeDynamism(absl::Span<const CompilationCacheKey> keys) {
  std::vector<std::vector<ShapeDynamismMetadata>> result;
  result.reserve(keys.size());
  TT_MUTEX_LOCK(lock, cache_mutex_);
  for (const auto key : keys) {
    CacheLookupInternal key_result = LookupCacheEntry(key);
    if (const auto* shape_dynamism_metadata =
            std::get_if<absl::Span<const ShapeDynamismMetadata>>(&key_result)) {
      result.push_back(std::vector<ShapeDynamismMetadata>(
          shape_dynamism_metadata->begin(), shape_dynamism_metadata->end()));
    } else {
      result.push_back({});
    }
  }
  return result;
}

bool CompilationCache::IsExecutableReady(CompilationCacheKey key) const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  if (const auto it = executable_cache_.find(key);
      it != executable_cache_.end()) {
    return IsFutureReady(it->second.executable_future());
  }
  return false;
}

absl::StatusOr<SharedLoadedExecutableFuture> CompilationCache::GetOrCompile(
    const CompilationCacheKey key,
    const ShapeDynamismMetadata& shape_dynamism_metadata,
    MlirComputationBuilder computation_builder,
    UniqueCompileOptions compile_options) {
  std::shared_ptr<LoadedExecutablePromise> compile_promise;
  SharedLoadedExecutableFuture future_compile_result;

  // Critical section for cache lookups and insertion.
  {
    TT_MUTEX_LOCK(lock, cache_mutex_);

    // Check the cache first.
    CacheLookupInternal cache_lookup = LookupCacheEntry(key);
    if (const auto* cache_entry =
            std::get_if<const CacheEntry*>(&cache_lookup)) {
      return (*cache_entry)->executable_future();
    }

    // If we are in cache-only mode, we are not allowed to compile.
    TT_RET_CHECK(!cache_only_mode_, error::kFailedPrecondition)
        << "The user has asserted that no more compilation should happen; "
           "yet compilation is needed for key "
        << key;

    // The cache is a miss, so we need to compile. Insert the future into the
    // cache while still holding the lock, so we don't race with another
    // thread that may have just seen the same key.
    CacheEntry& entry = executable_cache_[key];
    compile_promise = entry.executable_promise();
    future_compile_result = entry.executable_future();
    // Insert the shape dynamism metadata into the multimap. Do this at the
    // same time as we create the compilation future; even if it fails, we
    // want to record that we attempted to compile it.
    const auto shapeless_key = key.shapeless_key;
    auto it = shape_dynamism_multimap_.find(shapeless_key);
    if (it == shape_dynamism_multimap_.end()) {
      it = shape_dynamism_multimap_.insert({shapeless_key, {}}).first;
    }
    it->second.push_back(shape_dynamism_metadata);  // intentional copy
    ABSL_VLOG(1) << "Inserted compilation future for key: " << key;
  }  // End of critical section.

  // The MLIR module *must* be built inside of the caller thread.
  absl::StatusOr<ContextedModule> contexted_module =
      ContextedModule::Make(computation_builder);
  ABSL_VLOG(1) << "[TtPerf] Built MLIR module for key: " << key;
  if (!contexted_module.ok()) {
    ABSL_VLOG(1) << "[TtPerf] Failed to build MLIR module for key: " << key
                 << " with status: " << contexted_module.status();
    TrySetExecutablePromise(key, *compile_promise, contexted_module.status());
    return future_compile_result;
  }
  if (ABSL_VLOG_IS_ON(3)) {
    LogLines(absl::StrCat("[TtPerf] Compiling Module for key: ", key, "\n",
                          DebugString(contexted_module->get(),
                                      DebugStringOptions::kEnableDebugInfo)));
  }
  LoadedExecutableBuilder builder =
      [contexted_module = *std::move(contexted_module)](
          xla::PjRtClient& client, UniqueCompileOptions options) mutable
      -> absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> {
    return client.CompileAndLoad(
        std::move(contexted_module).ToMaybeOwningMlirModule(),
        std::move(*options));
  };

  compilation_pool_->Schedule(
      [this, key, builder = std::move(builder),
       compile_options = std::move(compile_options)]() mutable {
        this->GetFromTier2OrCompile(key, std::move(builder),
                                    std::move(compile_options));
      });
  ABSL_VLOG(1) << "[TtPerf] Scheduled compilation for key: " << key;
  return future_compile_result;
}

void CompilationCache::SetExecutable(
    CompilationCacheKey key, absl::StatusOr<SharedLoadedExecutable> executable,
    CacheEntryStats stats) {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  // If the user requested to evict the cache while we were compiling this
  // executable, the key may be missing. In this case, adding the key back to
  // the cache doesn't help as EvictAll() already set the executable future
  // to a failed state, so we just log and return.
  const auto it = executable_cache_.find(key);
  if (it == executable_cache_.end()) {
    ABSL_LOG(WARNING) << "Key already evicted when setting executable for key "
                      << key;
    return;
  }

  CacheEntry& cache_entry = it->second;
  if (IsFutureReady(cache_entry.executable_future())) {
    // Another thread has already set the executable future.
    return;
  }

  cache_entry.stats() = std::move(stats);
  if (cache_entry.stats().compilation_duration > absl::ZeroDuration()) {
    ABSL_VLOG(1) << "Compile duration for key " << key << ": "
                 << absl::ToInt64Milliseconds(
                        cache_entry.stats().compilation_duration)
                 << " ms";
  }

  TrySetExecutablePromise(key, *cache_entry.executable_promise(),
                          std::move(executable));
}

void CompilationCache::GetFromTier2OrCompile(
    CompilationCacheKey key, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options) {
  const bool uses_tier2 = UsesTier2CompilationCache();
  ABSL_VLOG(1) << "Compiling executable for key: " << key
               << (uses_tier2 ? absl::StrCat(" with tier-2 cache at ",
                                             GetTier2CompilationCachePath())
                              : "");

  if (!uses_tier2) {
    Compile(std::move(key), std::move(executable_builder),
            std::move(compile_options));
    return;
  }

  // Using tier-2 cache.
  const absl::Time request_start = absl::Now();

  // Check 1: if the compilation result is already in the tier-2 cache, use it.
  Tier2CacheEntryStats tier2_stats;
  auto executable_or = GetFromTier2Cache(key, request_start, tier2_stats);
  if (executable_or.ok()) {
    SetExecutable(key, std::move(executable_or),
                  {.tier2 = std::move(tier2_stats)});
    return;
  }
  ABSL_VLOG(2) << "Tier-2 cache MISS for key: " << key;

  // Otherwise, try to compile the graph.
  // Critical section for updating the tier-2 cache entry.
  {
    // Lock the tier-2 cache entry for the key to avoid multiple processes
    // doing the same compilation. This is a blocking call.
    Tier2CacheEntryLock lock(key);
    {
      // Check again if the compilation result is already in the tier-2 cache,
      // in case another thread or process has just finished the compilation and
      // released the lock.
      Tier2CacheEntryStats tier2_stats;
      auto executable_or = GetFromTier2Cache(key, request_start, tier2_stats);
      if (executable_or.ok()) {
        SetExecutable(key, std::move(executable_or),
                      {.tier2 = std::move(tier2_stats)});
        return;
      }
    }

    GetFromTier3OrCompile(std::move(key), std::move(executable_builder),
                          std::move(compile_options), lock, request_start);
  }  // End of critical section.
}

void CompilationCache::GetFromTier3OrCompile(
    CompilationCacheKey key, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options, Tier2CacheEntryLock& lock,
    const absl::Time request_start) {
  const bool uses_tier3 = UsesTier3CompilationCache();
  const auto pre_compile_duration = absl::Now() - request_start;
  absl::StatusOr<SharedLoadedExecutable> executable_or;
  CacheTier tier = CacheTier::kUnknown;

  const bool backup_compilation = UsesLocalBackupTaskForTier3Read();
  if (backup_compilation) {
    backup_compilation_pool_->Schedule(
        [this, key, builder = std::move(executable_builder),
         options = std::move(compile_options)]() mutable {
          // As an optimization, if the tier-3 cache read has already
          // populated the executable, skip the backup compilation.
          if (this->IsExecutableReady(key)) {
            ABSL_VLOG(1) << "Skipping backup compilation for key: " << key
                         << " because it is already ready.";
            return;
          }
          this->Compile(key, std::move(builder), std::move(options));
        });
  }

  // If tier-3 cache is enabled, try to get the executable from it first.
  if (uses_tier3) {
    executable_or = GetFromTier3Cache(key);
    if (executable_or.ok()) {
      ABSL_VLOG(1) << "Tier-3 cache HIT for key " << key;
      tier = CacheTier::kTier3;
      SetExecutable(key, executable_or, /*stats=*/{});
    } else {
      ABSL_VLOG(1) << "Tier-3 cache MISS for key " << key
                   << " with status: " << executable_or.status();
    }
  }

  if (!executable_or.ok()) {
    // Either tier-3 cache is disabled, or the executable was not found in it.
    // Compile the graph and save the result to the tier-1 cache.
    tier = CacheTier::kTier1;
    if (!backup_compilation) {
      // Clang-tidy reports a false positive here that executable_builder and
      // compile_options are used after being moved in the backup compilation
      // above. It is wrong as that move only happens when backup_compilation is
      // true.
      Compile(key, std::move(executable_builder),  // NOLINT
              std::move(compile_options));         // NOLINT
    }
    // When the above call finishes, the tier-1 cache will contain the compiled
    // executable and its initial stats.
  }

  // Read the tier-1 cache.
  SharedLoadedExecutableFuture f;
  {
    TT_MUTEX_LOCK(lock, cache_mutex_);
    CacheEntry& cache_entry = executable_cache_[key];
    f = cache_entry.executable_future();
  }
  // Important: the .get() must be called outside the lock region to avoid a
  // deadlock. For example, if this function (GetFromTier3OrCompile) scheduled
  // a backup compilation above, and the future is not ready yet, f.get() will
  // block until the backup compilation finishes and sets the future. However,
  // the backup compilation is running in a separate thread and won't be able
  // to set the future as the lock is still held by this thread.
  executable_or = f.get();

  if (!executable_or.ok()) {
    // The compilation failed.
    return;
  }

  // Write to tier-2 cache.
  const absl::Time write_start = absl::Now();
  const std::string tier2_entry_path = GetTier2CacheEntryPath(key);
  ABSL_VLOG(1) << "Writing to tier-2 cache for key " << key << " at path "
               << tier2_entry_path;
  const absl::Status status =
      AtomicWriteToCacheFile(tier2_entry_path, *executable_or);
  // Release the lock on the tier-2 cache entry now to unblock
  // processes that are waiting for it sooner.
  lock.Release();

  const absl::Duration write_duration = absl::Now() - write_start;
  if (!status.ok()) {
    // Writing to the tier-2 cache is best-effort - we don't need to fail
    // the user if it fails (the program will just run slower).
    ABSL_LOG(ERROR) << status;
  } else {
    ABSL_VLOG(2) << "Tier-2 cache WRITE for key: " << key
                 << "\n  Pre-compile wait: " << pre_compile_duration
                 << "\n  Write duration: " << write_duration;
    TT_MUTEX_LOCK(lock, cache_mutex_);
    CacheEntry& cache_entry = executable_cache_[key];
    auto& tier2_stats = cache_entry.stats().tier2;
    if (!tier2_stats.has_value()) {
      tier2_stats.emplace();
    }
    tier2_stats->pre_compile_duration = pre_compile_duration;
    tier2_stats->write_duration = write_duration;
  }

  if (tier == CacheTier::kTier1 && uses_tier3) {
    // Write to tier-3 cache.
    const std::string tier3_entry_path = GetTier3CacheEntryPath(key);
    ABSL_VLOG(1) << "Writing to tier-3 cache for key " << key << " at path "
                 << tier3_entry_path;
    const absl::Status status =
        AtomicWriteToCacheFile(tier3_entry_path, *executable_or);
    if (!status.ok()) {
      // Writing to the tier-3 cache is best-effort - we don't need to fail
      // the user if it fails (the program will just run slower).
      ABSL_LOG(ERROR) << status;
    }
  }
}

void CompilationCache::Compile(CompilationCacheKey key,
                               LoadedExecutableBuilder executable_builder,
                               UniqueCompileOptions compile_options) {
  xla::PjRtClient* const client = GetPjRtClient();
  if (client == nullptr) {
    absl::Status error = TT_ERROR(error::kFailedPrecondition)
                         << "PjRtClient must be initialized";
    ABSL_LOG(ERROR) << error.message();

    // Fulfill the promise immediately with the error status.
    SetExecutable(key, std::move(error), /*stats=*/{});
    return;
  }

  const absl::Time compile_start = absl::Now();
  absl::StatusOr<SharedLoadedExecutable> executable = torch_tpu::Compile(
      *client, std::move(executable_builder), std::move(compile_options));

  SetExecutable(key, std::move(executable),
                {
                    .compilation_duration = absl::Now() - compile_start,
                });
}

std::string CompilationCache::HbmUsageSummary() const {
  TT_MUTEX_LOCK(lock, cache_mutex_);
  int64_t total_size_bytes = 0;
  int64_t num_executables = 0;
  for (const auto& [key, cache_entry] : executable_cache_) {
    if (!IsFutureReady(cache_entry.executable_future())) {
      continue;
    }
    const auto& executable = cache_entry.executable_future().get();
    if (!executable.ok() || *executable == nullptr) {
      continue;
    }
    const auto memory_stats = (*executable)->GetCompiledMemoryStats();
    if (!memory_stats.ok()) {
      ABSL_LOG(WARNING) << "Failed to get memory stats for key: " << std::hex
                        << key << " with status: " << memory_stats.status();
      continue;
    }
    total_size_bytes += memory_stats->generated_code_size_in_bytes;
    num_executables++;
  }
  return absl::StrCat("CompilationCache HBM usage: ",
                      tsl::strings::HumanReadableNumBytes(total_size_bytes),
                      " for ", num_executables, " executables.");
}

}  // namespace torch_tpu
