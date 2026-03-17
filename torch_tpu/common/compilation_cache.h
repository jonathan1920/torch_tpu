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

#ifndef TORCH_TPU_COMMON_COMPILATION_CACHE_H_
#define TORCH_TPU_COMMON_COMPILATION_CACHE_H_

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/thread_pool.h"
#include "torch_tpu/common/tier2_compilation_cache.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

struct CompilationCacheInitializationOptions {
  bool cache_only = false;
};

// Statistics for a single entry in the tier-1 compilation cache.
struct CacheEntryStats {
  // If the entry was successfully compiled by this process, this is the time it
  // took to compile. Otherwise (e.g. if the compilation failed, or if the entry
  // was read from the tier-2 cache), it is zero.
  absl::Duration compilation_duration;
  // The last time this entry was read. The initial compilation counts as one
  // read as the executable will be used by whoever triggered the compilation.
  absl::Time last_read = absl::Now();
  // How many times this entry has been read. The initial compilation counts as
  // one read as the executable will be used by whoever triggered the
  // compilation.
  int64_t read_count = 1;
  // Set only when the entry is populated from the tier-2 cache as opposed to
  // being compiled from scratch.
  std::optional<Tier2CacheEntryStats> tier2;
};

class CompilationCache;

// A cache entry, which is a future executable with its metadata.
// Invariant: executable_future is associated with executable_promise.
class CacheEntry {
 public:
  CacheEntry()
      : executable_promise_(std::make_shared<LoadedExecutablePromise>()),
        executable_future_(executable_promise_->get_future().share()) {}

  // This class is movable but not copyable.
  CacheEntry(CacheEntry&&) = default;
  CacheEntry& operator=(CacheEntry&&) = default;
  CacheEntry(const CacheEntry&) = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;

  [[nodiscard]] const SharedLoadedExecutableFuture& executable_future() const {
    return executable_future_;
  }

  [[nodiscard]] const std::shared_ptr<LoadedExecutablePromise>&
  executable_promise() const {
    return executable_promise_;
  }

  [[nodiscard]] CacheEntryStats& stats() const { return stats_; }

 private:
  // Order the fields so that the promise is constructed before the future
  // and destroyed after it.
  std::shared_ptr<LoadedExecutablePromise> executable_promise_;
  SharedLoadedExecutableFuture executable_future_;
  mutable CacheEntryStats stats_;
};

class BoundedDynamicCacheEntry {
 public:
  explicit BoundedDynamicCacheEntry(
      ShapeDynamismMetadata shape_dynamism_metadata)
      : middle_cache_entry_(CacheEntry()),
        shape_dynamism_metadata_(std::move(shape_dynamism_metadata)) {}

  BoundedDynamicCacheEntry(BoundedDynamicCacheEntry&&) = default;
  BoundedDynamicCacheEntry& operator=(BoundedDynamicCacheEntry&&) = default;
  BoundedDynamicCacheEntry(const BoundedDynamicCacheEntry&) = delete;
  BoundedDynamicCacheEntry& operator=(const BoundedDynamicCacheEntry&) = delete;

  [[nodiscard]] const ShapeDynamismMetadata& shape_dynamism_metadata() const {
    return shape_dynamism_metadata_;
  }

  // Returns the cache entry corresponding to the middle part of the bounded
  // dynamic cache entry: BoundedDynamicCacheEntries can be thought of as
  // triplets [preamble, middle, postamble] where the preamble and postamble are
  // lightweight computations.
  [[nodiscard]] const CacheEntry& middle_cache_entry() const {
    return middle_cache_entry_;
  }

  [[nodiscard]] const SharedLoadedExecutableFuture& middle_executable_future()
      const {
    return middle_cache_entry_.executable_future();
  }

  [[nodiscard]] const absl_nonnull std::shared_ptr<LoadedExecutablePromise>&
  middle_executable_promise() const {
    return middle_cache_entry_.executable_promise();
  }

 private:
  CacheEntry middle_cache_entry_;
  ShapeDynamismMetadata shape_dynamism_metadata_;
};

// Aggregated statistics for the compilation cache.
struct PerfStats {
  // Statistics for a single entry in the compilation cache.
  struct EntryStats : public CacheEntryStats {
    CompilationCacheKey key;
  };

  // These fields refer to the tier-1 cache. E.g. a hit here means the entry was
  // found in the tier-1 cache. A miss means the entry was not found in the
  // tier-1 cache, and we had to look it up in the tier-2 cache, where it may
  // or may not be found.
  int64_t num_cache_misses() const { return num_cache_reqs - num_cache_hits; }
  int64_t num_cache_reqs = 0;
  int64_t num_cache_hits = 0;

  std::vector<EntryStats> per_entry_stats;
};

// An in-memory cache for compiled XLA programs. The ownership of cached
// executables is shared between the cache and the user of the executable.
// This shared ownership is needed to avoid deleting an executable still in use
// when the cache decides to evict it.
class CompilationCache {
 public:
  // CompilationCache is neither copyable nor movable.
  CompilationCache(const CompilationCache&) = delete;
  CompilationCache& operator=(const CompilationCache&) = delete;
  CompilationCache(CompilationCache&&) = delete;
  CompilationCache& operator=(CompilationCache&&) = delete;

  // Initializes the global compilation cache singleton.
  // If called more than once, a new singleton will **not** be created; the
  // existing singleton will be updated with the new options.
  // To clear the cache and create a new singleton, call Shutdown() first.
  static void Initialize(const CompilationCacheInitializationOptions& options);

  // Returns the singleton cache instance.
  [[nodiscard]] static CompilationCache& GetInstance();

  // Shuts down the cache and its thread pool, evicting all entries.
  static void Shutdown();

  // Evicts all existing executables from the cache. The function waits for all
  // in-flight compilations to complete.
  void EvictAll() ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the cache to allow caching or not. On initialization, the cache is
  // in allow-cache mode by default. We only need to call this function if we
  // want to disable caching (e.g. for debugging or perf analysis).
  void SetAllowCacheMode(bool allow = true) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the cache to only lookup and not compile if cache_only is true.
  // In this mode, the cache will not compile any programs, and will return
  // errors for all cache misses.
  void SetCacheOnlyMode(bool cache_only = true)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Cache statistics.
  [[nodiscard]] int64_t GetCacheRequests() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] int64_t GetCacheHits() const ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] int64_t GetCacheMisses() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] PerfStats GetCacheStats() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns true if the executable associated with a cache key is compiled and
  // ready for execution.
  bool IsExecutableReady(CompilationCacheKey key) const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Fetches a CompiledKernel with the right executable futures from the cache,
  // and enqueues compilations if necessary.
  absl::StatusOr<CompiledKernel> GetOrCompile(
      CompilationCacheKey key, const std::vector<Shape>& input_shapes,
      MlirComputationBuilder computation_builder,
      UniqueCompileOptions compile_options) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Schedules compilation of this builder and sets it in the executable
  // promise.
  //
  // Parameters:
  //  - executable_promise: The promise to set the executable in.
  //  - computation_builder: A function that builds the MLIR computation.
  //  - compile_options: The compile options to use for the compilation.
  //  - key: The key where this entry is stored in the cache, used for tiered
  //    cache lookups. If not provided, we will not perform any tiered cache
  //    lookups. TODO(unda): remove this parameter once we store the dynamic
  //    kernel in the flat executable cache.
  void EnqueueCompilation(
      absl_nonnull std::shared_ptr<LoadedExecutablePromise> executable_promise,
      MlirComputationBuilder computation_builder,
      UniqueCompileOptions compile_options,
      std::optional<CompilationCacheKey> key = std::nullopt)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Debugging function to return the total resident size of the loaded
  // executables in HBM.
  std::string HbmUsageSummary() const ABSL_LOCKS_EXCLUDED(cache_mutex_);

 private:
  // Private constructor and destructor to enforce singleton pattern.
  CompilationCache();
  ~CompilationCache();

  // The internal variant type for LookupCacheEntry, which returns references
  // to the cache instead, given a CompilationCacheKey. This is only to be used
  // internally while the mutex is held.
  // If the key matches something in executable_cache_, a CacheEntry will be
  // returned - this is the best case scenario. If not, and the ShapelessKey
  // component of the key matches something in bounded_dynamic_cache_, we will
  // check if one of the existing BoundedDynamicCacheEntry matches the input
  // shapes, and return one if so. Otherwise, monostate will be
  // returned. Note that if there are both static and dynamic cache hits, the
  // static cache hit will be returned. If a cache entry had to be created,
  // needs_compilation will be true, otherwise it will be false.
  struct CacheLookupInternal {
    std::variant<const CacheEntry* absl_nonnull,
                 const BoundedDynamicCacheEntry* absl_nonnull, std::monostate>
        cache_entry;
    bool needs_compilation = false;
  };

  // Private helper to retrieve a cache entry or create a new one as needed.
  // This updates the access count and last accessed time for the cache entry
  // if it is found.
  //
  // Parameters:
  //  - key: The key to lookup in the cache.
  //  - input_shapes: The input shapes to the computation, in the order given by
  // a Traversal. This is used for checking compatibility with existing bounded
  // dynamic cache entries, or to create metadata for a new one.
  //
  // Returns:
  //   A pair containing the cache lookup result and a boolean indicating
  //   whether the entry had to be created (true) or was found (false).
  absl::StatusOr<CacheLookupInternal> GetOrCreateCacheEntry(
      CompilationCacheKey key, const std::vector<Shape>& input_shapes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Sets the executable for the given key if it is not already set.
  // Also sets the stats for the cache entry.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_`.
  //  - `cache_mutex_` is not held.
  void SetExecutable(CompilationCacheKey key,
                     absl::StatusOr<SharedLoadedExecutable> executable,
                     CacheEntryStats stats) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Compiles the graph and stores the executable in the tier-1 cache.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_`.
  //  - `cache_mutex_` is not held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  //
  // This function is used by `GetFromTier2OrCompile()` to compile the graph
  // after a failed tier-2 cache lookup.
  void Compile(CompilationCacheKey key,
               LoadedExecutableBuilder executable_builder,
               UniqueCompileOptions compile_options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Ensures that the padding compilation for this entry is enqueued
  // exactly once, even if called concurrently by multiple threads. It is safe
  // to call this method without holding any external locks.
  // TODO(unda): replace this with EnqueueCompilation, once we have a way to
  // generate a cache key for the padding compilation.
  //
  // Parameters:
  //  - shape_dynamism_metadata: The shape dynamism metadata to use for the
  //    padding compilation.
  //  - input_shapes: The input shapes to the computation.
  //  - compile_options: The compile options to use for the compilation.
  SharedLoadedExecutableFuture EnqueuePaddingCompilation(
      const ShapeDynamismMetadata& shape_dynamism_metadata,
      const std::vector<Shape>& input_shapes,
      UniqueCompileOptions compile_options) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Tries to get the compilation result from the tier-2 cache; if not
  // found, compiles the graph and stores the executable in the tier-1/2/3
  // caches.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_` but the executable is not set.
  //  - `cache_mutex_` is not held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  void GetFromTier2OrCompile(CompilationCacheKey key,
                             LoadedExecutableBuilder executable_builder,
                             UniqueCompileOptions compile_options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Tries to get the compilation result from the tier-3 cache; if not found,
  // compiles the graph and stores the executable in the tier-1/2/3 caches.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_` but the executable is not set.
  //  - `key` is not in the tier-2 cache.
  //  - `cache_mutex_` is not held.
  //  - `lock` is the lock for the tier-2 cache entry for `key` and held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  void GetFromTier3OrCompile(CompilationCacheKey key,
                             LoadedExecutableBuilder executable_builder,
                             UniqueCompileOptions compile_options,
                             Tier2CacheEntryLock& lock,
                             absl::Time request_start)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Global mutex for singleton creation.
  static absl::Mutex g_mutex_;

  // Main mutex for protecting cache statistics and cache-only mode.
  mutable absl::Mutex cache_mutex_;

  // Thread pool for running compilations concurrently.
  std::unique_ptr<ThreadPool> compilation_pool_;

  // Lower-priority thread pool for running backup compilations.
  // Set only when tier-3 cache is enabled and local backup task is enabled.
  std::unique_ptr<ThreadPool> backup_compilation_pool_;

  // The cache of successfully compiled and in flight executables.
  absl::flat_hash_map<CompilationCacheKey, CacheEntry,
                      CompilationCacheKey::Hash>
      executable_cache_ ABSL_GUARDED_BY(cache_mutex_);

  // The cache of successfully compiled and in flight executables with bounded
  // dynamic shapes.
  // TODO(unda): replace the cache_entry value with a reference to
  // executable_cache_.
  absl::flat_hash_map<ShapelessKey, std::vector<BoundedDynamicCacheEntry>,
                      ShapelessKey::Hash>
      bounded_dynamic_cache_ ABSL_GUARDED_BY(cache_mutex_);

  // Cache statistics.
  mutable PerfStats perf_stats_ ABSL_GUARDED_BY(cache_mutex_);
  // If true, the cache will be enabled. If false, the cache will be disabled
  // and every computation graph will be compiled from scratch (this is useful
  // for debugging and perf analysis).
  bool allow_cache_mode_ ABSL_GUARDED_BY(cache_mutex_) = true;
  // If true, the cache will only lookup and not compile. Any cache miss will
  // result in an error.
  bool cache_only_mode_ ABSL_GUARDED_BY(cache_mutex_) = false;
};

// Returns the number of threads to use for compilation based on the flag value,
// the NPROC environment variable, or the hardware concurrency. Guaranteed to be
// > 0.
[[nodiscard]] int GetNumCompilationThreads();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_CACHE_H_
