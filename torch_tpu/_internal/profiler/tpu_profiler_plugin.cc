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

#include "torch_tpu/_internal/profiler/tpu_profiler_plugin.h"

#include <cstdlib>
#include <fstream>
#include <ios>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/output_base.h>
#include "torch_tpu/_internal/profiler/xprof_callback_handler.h"
#include "torch_tpu/common/utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if TT_IS_INTERNAL_TORCH_TPU
#include "torch/csrc/profiler/standalone/privateuse1_profiler.h"
#endif

namespace torch_tpu {

#if TT_IS_INTERNAL_TORCH_TPU

TpuKinetoProfilerSession::TpuKinetoProfilerSession(
    const libkineto::Config& config,
    std::set<libkineto::ActivityType> activity_types)
    : config_(config), activity_types_(std::move(activity_types)) {}

void TpuKinetoProfilerSession::start() {
  absl::MutexLock lock(mutex_);
  if (session_ != nullptr) {
    ABSL_LOG(WARNING) << "TpuKinetoProfilerSession already started";
    return;
  }
  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  opts.set_device_type(tensorflow::ProfileOptions::TPU);
  // device_tracer_level: 0=disabled, 1=enabled.
  opts.set_device_tracer_level(1);
  // host_tracer_level: 0=disabled, 1=user-instrumented tracemes,
  // 2=1+XLA tracemes, 3=2+low-level XLA tracemes.
  opts.set_host_tracer_level(2);
  // python_tracer_level: 0=disabled, 1=enabled.
  opts.set_python_tracer_level(0);

  // TODO(b/500368753): Introduce custom config parsing.

  session_ = tsl::ProfilerSession::Create(opts);
  if (session_ == nullptr) {
    ABSL_LOG(ERROR) << "Failed to create Tpu ProfilerSession!";
    errors_.push_back("Failed to create Tpu ProfilerSession");
    status_ = libkineto::TraceStatus::ERROR;
  } else {
    ABSL_LOG(INFO) << "Successfully created Tpu ProfilerSession!";
    status_ = libkineto::TraceStatus::RECORDING;
    callback_handler_ = std::make_unique<XProfCallbackHandler>();
  }
}

void TpuKinetoProfilerSession::stop() {
  absl::MutexLock lock(mutex_);
  if (session_ != nullptr) {
    callback_handler_.reset();
    status_ = libkineto::TraceStatus::PROCESSING;
    absl::Status status = session_->CollectData(&xspace_);
    ABSL_LOG(INFO) << "TpuKinetoProfilerSession::stop() CollectData status: "
                   << status;
    ABSL_LOG(INFO)
        << "TpuKinetoProfilerSession::stop() Collected XSpace planes: "
        << xspace_.planes_size();
    if (!status.ok()) {
      errors_.push_back("Failed to collect TPU profiling data: " +
                        status.ToString());
      status_ = libkineto::TraceStatus::ERROR;
      session_.reset();
      return;
    }
    status_ = libkineto::TraceStatus::READY;

    std::string run_dir =
        std::string(tsl::io::Dirname(config_.activitiesLogFile()));

    std::string output_path = "/tmp/xplane.pb";
    const char* tpu_xplane_path =
        std::getenv("TPU_XPLANE_PATH");  // GETENV_OK=test override
    if (tpu_xplane_path != nullptr) {
      output_path = tpu_xplane_path;
    } else if (!run_dir.empty()) {
      output_path = tsl::io::JoinPath(run_dir, "xplane.pb");
    }

    std::ofstream f(output_path, std::ios::binary);
    ABSL_VLOG(1) << "Attempting to write XPlane to " << output_path;
    if (!f) {
      errors_.push_back("Failed to open XSpace output file: " + output_path);
      ABSL_LOG(ERROR) << "Failed to open XSpace output file: " << output_path;
    } else {
      xspace_.SerializeToOstream(&f);
      f.close();
      ABSL_LOG(INFO) << "Successfully wrote XPlane to " << output_path;
    }
  }
  session_.reset();
}

void TpuKinetoProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  // TODO(b/499240330): implement this.
}

REGISTER_PRIVATEUSE1_PROFILER(
    TpuProfiler);  // NOLINT(readability-named-parameter)

#endif  // TT_IS_INTERNAL_TORCH_TPU

}  // namespace torch_tpu
