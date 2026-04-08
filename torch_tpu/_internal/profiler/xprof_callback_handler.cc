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

#include "torch_tpu/_internal/profiler/xprof_callback_handler.h"

#include <memory>
#include <string>
#include <utility>

#include "ATen/record_function.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace torch_tpu {

namespace {

// Observer context for RecordFunction callbacks.
// Each RecordFunction push creates a TraceMe instance via onFunctionEnter,
// and OnFunctionExit destroys it, effectively measuring the duration of the
// RecordFunction.
class XProfObserverContext : public at::ObserverContext {
 public:
  explicit XProfObserverContext(tsl::profiler::TraceMe tm)
      : traceme_(std::move(tm)) {}

 private:
  tsl::profiler::TraceMe traceme_;
};

// The callback function that is called when a RecordFunction is pushed.
// It creates a TraceMe instance and returns a XProfObserverContext.
std::unique_ptr<at::ObserverContext> OnFunctionEnter(
    const at::RecordFunction& fn) {
  return std::make_unique<XProfObserverContext>(tsl::profiler::TraceMe([&] {
    return tsl::profiler::TraceMeEncode(
        fn.name(), {{"correlation_id", std::to_string(fn.handle())}});
  }));
}

// The callback function that is called when a RecordFunction is popped.
void OnFunctionExit(const at::RecordFunction& /*fn*/,
                    at::ObserverContext* /*context_ptr*/) {
  // TraceMe destructor handles the end event.
}

}  // namespace

XProfCallbackHandler::XProfCallbackHandler()
    : handle_(at::addGlobalCallback(
          at::RecordFunctionCallback(OnFunctionEnter, OnFunctionExit)
              .needsIds(true))) {}

}  // namespace torch_tpu
