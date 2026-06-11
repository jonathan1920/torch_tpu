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

#ifndef TORCH_TPU_COMMON_THREAD_LOCAL_CONTEXT_H_
#define TORCH_TPU_COMMON_THREAD_LOCAL_CONTEXT_H_

#include <utility>

#include "ATen/ThreadLocalState.h"

namespace torch_tpu {

// A RAII wrapper to facilitate capturing and propagating thread-local states
// across thread boundaries.
class ThreadLocalContext {
 public:
  // Movable but not copyable.
  ThreadLocalContext(ThreadLocalContext&&) = default;
  ThreadLocalContext& operator=(ThreadLocalContext&&) = default;
  ThreadLocalContext(const ThreadLocalContext&) = delete;
  ThreadLocalContext& operator=(const ThreadLocalContext&) = delete;

  // Captures thread-local state of the current (calling) thread.
  [[nodiscard]] static ThreadLocalContext Capture() {
    return ThreadLocalContext(at::ThreadLocalState());
  }

  // Restores the previously captured thread-local state and applies it to
  // execute the callback `f`.
  template <typename F>
  decltype(auto) Apply(F&& f) const {
    at::ThreadLocalStateGuard guard(state_);
    return std::forward<F>(f)();
  }

 private:
  explicit ThreadLocalContext(at::ThreadLocalState state)
      : state_(std::move(state)) {}

  at::ThreadLocalState state_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_THREAD_LOCAL_CONTEXT_H_
