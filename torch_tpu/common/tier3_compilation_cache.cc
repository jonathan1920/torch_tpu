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

#include "torch_tpu/common/tier3_compilation_cache.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/tier2_compilation_cache.h"

namespace torch_tpu {

namespace fs = std::filesystem;

constexpr std::string_view kTier3CacheFileExtension = ".bin";

// Returns the root directory of the tier-3 compilation cache, as set by the
// TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT environment variable.  If the
// environment variable is not set, it returns an empty string.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] static const std::string& GetTier3CacheRootDir() {
  static const absl::NoDestructor<std::string> root_dir([]() {
    const auto root_dir =
        GetEnvOnce<kTorchTpuInternalTier3CompilationCacheRootEnvVar>().value_or(
            "");
    ABSL_LOG(INFO) << "Tier-3 compilation cache root directory: " << root_dir;
    return root_dir;
  }());
  return *root_dir;
}

bool UsesTier3CompilationCache() {
  // To use tier-3 cache, we must use tier-2 cache as well.
  return UsesTier2CompilationCache() && !GetTier3CacheRootDir().empty();
}

// Returns the path to the tier-3 compilation cache directory for the current
// process.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] static const std::string& GetTier3CompilationCachePath() {
  static const absl::NoDestructor<std::string> cache_path([]() {
    ABSL_CHECK(UsesTier3CompilationCache())  // CRASH_OK
        << "Tier-3 compilation cache is not enabled.";
    std::string path = absl::StrCat(
        GetTier3CacheRootDir(), "/",
        absl::Hex(GetTorchTpuBinaryFingerprint(), absl::kZeroPad16));
    ABSL_LOG(INFO) << "Tier-3 compilation cache path for this process: "
                   << path;
    return path;
  }());
  return *cache_path;
}

// Ensures that the tier-3 cache directory exists. This function only does
// the check once.
static void EnsureTier3CacheDirExistsOnceOrDie() {
  static const bool dir_exists = []() {
    const std::string cache_path = GetTier3CompilationCachePath();
    const absl::Status status = EnsureDirExistsRecursively(cache_path);
    ABSL_CHECK(status.ok())  // CRASH_OK
        << "Failed to create tier-3 cache directory: " << cache_path
        << " with error: " << status.message();
    return true;
  }();
  static_cast<void>(dir_exists);  // Avoid unused variable warning.
}

std::string GetTier3CacheEntryPath(CompilationCacheKey key) {
  EnsureTier3CacheDirExistsOnceOrDie();
  return absl::StrCat(GetTier3CompilationCachePath(), "/", key.CompactFormat(),
                      kTier3CacheFileExtension);
}

absl::StatusOr<SharedLoadedExecutable> GetFromTier3Cache(
    CompilationCacheKey key) {
  // Read the cache file into a char buffer.
  const std::string cache_entry_path = GetTier3CacheEntryPath(key);
  ABSL_VLOG(1) << "Reading tier-3 cache for key " << key << " at path "
               << cache_entry_path;
  TT_RET_CHECK(fs::exists(cache_entry_path), error::kNotFound)
      << "cannot find tier-3 cache file " << cache_entry_path;
  TT_RET_CHECK(fs::is_regular_file(cache_entry_path), error::kInvalidArgument)
      << cache_entry_path << " is not a regular file";
  const auto file_size = fs::file_size(cache_entry_path);
  std::ifstream file(cache_entry_path, std::ios::binary);
  TT_RET_CHECK(file.is_open(), error::kNotFound)
      << "cannot open tier-3 cache file " << cache_entry_path;
  std::string data;
  data.resize(file_size);
  file.read(data.data(), file_size);
  TT_RET_CHECK(file.gcount() == file_size, error::kNotFound)
      << "cannot read tier-3 cache file " << cache_entry_path;
  ABSL_VLOG(1) << "Read " << data.size() << " bytes from tier-3 cache for key "
               << key << " at path " << cache_entry_path;

  // Load the executable from the char buffer.
  return LoadSerializedExecutable(CacheTier::kTier3, key, data);
}

}  // namespace torch_tpu
