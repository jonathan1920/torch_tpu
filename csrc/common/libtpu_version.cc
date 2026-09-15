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

#include "csrc/common/libtpu_version.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {
namespace {

using VersionComponents = std::array<int64_t, 4>;

constexpr VersionComponents kDefaultVersionComponents = {
    0, 0, 0, std::numeric_limits<int64_t>::max()};

struct LibtpuVersionState {
  absl::Mutex mutex;
  std::optional<std::string> version ABSL_GUARDED_BY(mutex);
  VersionComponents parsed_version ABSL_GUARDED_BY(mutex) =
      kDefaultVersionComponents;
};

LibtpuVersionState& GetState() {
  static absl::NoDestructor<LibtpuVersionState> state;
  return *state;
}

absl::StatusOr<VersionComponents> ParseVersion(std::string_view version) {
  VersionComponents components = kDefaultVersionComponents;
  size_t idx = 0;
  for (std::string_view part : absl::StrSplit(version, '.')) {
    TT_RET_CHECK(idx < components.size(), error::kInvalidArgument)
        << "expected version string to be of the format "
           "<major>.<minor>.<patch>[.dev<date>], got '"
        << version << "'";
    size_t start = 0;
    while (start < part.size() && !absl::ascii_isdigit(part[start])) {
      ++start;
    }
    if (start >= part.size()) {
      continue;
    }
    size_t end = start;
    while (end < part.size() && absl::ascii_isdigit(part[end])) {
      ++end;
    }
    int64_t value = 0;
    if (absl::SimpleAtoi(part.substr(start, end - start), &value)) {
      components[idx++] = value;
    }
  }
  return components;
}

enum class ComparisonResult {
  kLess,
  kEqual,
  kGreater,
};

// Compares two versions component-by-component on-the-fly without copying or
// vector resizing.
// Returns kLess if lhs < rhs, kEqual if lhs == rhs, kGreater if lhs > rhs.
ComparisonResult CompareVersions(const VersionComponents& lhs,
                                 const VersionComponents& rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return lhs[i] < rhs[i] ? ComparisonResult::kLess
                             : ComparisonResult::kGreater;
    }
  }
  return ComparisonResult::kEqual;
}

}  // namespace

void SetLibtpuVersion(std::string_view version) {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  if (state.version.has_value()) {
    TT_CHECK_THROW(*state.version == version, error::kInvalidArgument)
        << "expected libtpu version to match previously set version '"
        << *state.version << "', got '" << version << "'";
    return;
  }
  TT_ASSIGN_OR_THROW(const VersionComponents parsed_version,
                     ParseVersion(version));
  state.version = std::string(version);
  state.parsed_version = parsed_version;
  if (version.empty()) {
    ABSL_LOG(INFO) << "libtpu not installed.";
  } else {
    ABSL_LOG(INFO) << "Setting libtpu version to: " << *state.version;
  }
}

std::optional<std::string> GetLibtpuVersion() {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  return state.version;
}

void ResetLibtpuVersionForTesting() {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  state.version.reset();
  state.parsed_version = kDefaultVersionComponents;
}

bool IsLibtpuVersionAtLeast(std::string_view min_version) {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  if (!state.version.has_value() || state.version->empty()) {
    return true;
  }
  const absl::StatusOr<VersionComponents> req_components =
      ParseVersion(min_version);
  if (!req_components.ok()) {
    ABSL_LOG(ERROR) << req_components.status().message();
    return false;
  }
  return CompareVersions(state.parsed_version, *req_components) !=
         ComparisonResult::kLess;
}

}  // namespace torch_tpu
