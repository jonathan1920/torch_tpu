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

#ifndef TORCH_TPU_COMMON_CONTAIN_H_
#define TORCH_TPU_COMMON_CONTAIN_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace torch_tpu {

// Dynamic container for tracking memory usage.
//
// This is a global singleton class that is initialized lazily on first use. It
// supports only one implicit container. Entering via ScopedEnter multiple
// times simply moves into said container for the duration of the scope, it
// does not create or destroy containers per call. Attempting to enter the
// container from a thread which is already inside of it has no effect.
//
// Example:
//  auto threads = torch_tpu::ThreadPool();
//  threads.Schedule([]() {
//    auto guard = ScopedEnter();
//    DoWork();
//    auto child_threads = torch_tpu::ThreadPool();
//    child_threads.Schedule([]() {
//      DoWork();
//    });
//  });
//
// GetPeakHostMemoryBytes() will reflect all calls to DoWork();

class ScopedContainer {
 public:
  ScopedContainer(const ScopedContainer&) = delete;
  ScopedContainer& operator=(const ScopedContainer&) = delete;
  ScopedContainer(ScopedContainer&&) = delete;
  ScopedContainer& operator=(ScopedContainer&&) = delete;
  ~ScopedContainer();

 private:
  friend std::unique_ptr<ScopedContainer> ScopedEnter();
  ScopedContainer();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Move the current thread of execution into the container, if enabled. Any
// threads spawned within the container will remain there, even when the current
// thread leaves. Returns a guard that restores the previous container on
// destruction. Lazily initializes the container on first call if enabled.
// Returns nullptr if disabled or initialization fails.
std::unique_ptr<ScopedContainer> ScopedEnter();

// Returns peak memory usage in bytes inside the container. Work done by any
// thread while inside the container, or any thread spawned inside the container
// will be accounted for.
absl::StatusOr<int64_t> GetPeakHostMemoryBytes();

// Destroys the container and resets the singleton state.
absl::Status Cleanup();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTAIN_H_
