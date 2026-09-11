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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu {
namespace {

struct LibtpuVersionState {
  absl::Mutex mutex;
  std::optional<std::string> version ABSL_GUARDED_BY(mutex);
  std::vector<int> parsed_version ABSL_GUARDED_BY(mutex);
};

LibtpuVersionState& GetState() {
  static auto* state = new LibtpuVersionState();
  return *state;
}

std::vector<int> ParseVersion(std::string_view version) {
  std::vector<int> components;
  for (std::string_view part : absl::StrSplit(version, '.')) {
    size_t start = 0;
    while (start < part.size() && !absl::ascii_isdigit(part[start])) {
      start++;
    }
    if (start >= part.size()) {
      continue;
    }
    size_t end = start;
    while (end < part.size() && absl::ascii_isdigit(part[end])) {
      end++;
    }
    int value = 0;
    if (absl::SimpleAtoi(part.substr(start, end - start), &value)) {
      components.push_back(value);
    }
  }
  return components;
}

void PadVersion(std::vector<int>& components, size_t target_len) {
  // Pads `components` up to `target_len`.
  // - The first three components (major, minor, patch) are zero-padded so that
  //   shorter versions like "0.1" are treated as "0.1.0".
  // - Subsequent components (4th component onwards) are padded with `INT_MAX`.
  //   This ensures that a stable release (e.g. "0.0.47" -> [0, 0, 47,
  //   INT_MAX]) compares greater than a pre-release or nightly build of the
  //   same release (e.g. "0.0.47.dev20260824" -> [0, 0, 47, 20260824]).
  while (components.size() < target_len) {
    if (components.size() < 3) {
      components.push_back(0);
    } else {
      components.push_back(std::numeric_limits<int>::max());
    }
  }
}

}  // namespace

void SetLibtpuVersion(std::string_view version) {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  if (state.version.has_value()) {
    ABSL_CHECK_EQ(  // CRASH_OK=Libtpu version is initialized once at startup;
                    // resetting to a different version implies an internal
                    // TorchTPU bug.
        *state.version, version)
        << "LibtpuVersion has already been set to '" << *state.version
        << "' and cannot be reset to a different version '" << version << "'.";
    return;
  }
  state.version = std::string(version);
  state.parsed_version = ParseVersion(version);
  if (state.version->empty()) {
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
  state.parsed_version.clear();
}

bool IsLibtpuVersionAtLeast(std::string_view min_version) {
  LibtpuVersionState& state = GetState();
  std::vector<int> current_components;
  {
    absl::MutexLock lock(state.mutex);
    if (!state.version.has_value() || state.version->empty()) {
      return true;
    }
    current_components = state.parsed_version;
  }
  std::vector<int> req_components = ParseVersion(min_version);
  const size_t target_len =
      std::max(current_components.size(), req_components.size());
  PadVersion(current_components, target_len);
  PadVersion(req_components, target_len);
  return current_components >= req_components;
}

}  // namespace torch_tpu
