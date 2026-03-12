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

#include "torch_tpu/_internal/profiler/profiler.h"

#include <memory>
#include <string>

#include "absl/log/absl_log.h"
#include "torch_tpu/common/error_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace torch_tpu {

namespace py = pybind11;

// We use raw pointers for global variables because they are trivially
// destructible, and complex static destructors are disallowed by ClangTidy.
static tsl::ProfilerSession* global_session = nullptr;
static tsl::profiler::ProfilerServer* global_server = nullptr;

void StartProfilerServer(int port) {
  if (!global_server) {
    // 1. Create the server object
    global_server = new tsl::profiler::ProfilerServer();

    // 2. Start it on the specific port
    global_server->StartProfilerServer(port);
  }
}

void StopProfilerServer() {
  // Destroying the object stops the server and releases the port.
  if (global_server) {
    delete global_server;
    global_server = nullptr;
  }
}

void StartTrace(const std::string& logdir, py::object options_obj) {
  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  if (!options_obj.is_none()) {
    py::dict options = options_obj.cast<py::dict>();
    if (options.contains("host_tracer_level")) {
      opts.set_host_tracer_level(options["host_tracer_level"].cast<int>());
    }
    if (options.contains("device_tracer_level")) {
      opts.set_device_tracer_level(options["device_tracer_level"].cast<int>());
    }
    if (options.contains("python_tracer_level")) {
      opts.set_python_tracer_level(options["python_tracer_level"].cast<int>());
    }
  }
  if (!logdir.empty()) {
    opts.set_repository_path(logdir);
  }
  ABSL_LOG(INFO) << "Starting trace, logdir: " << logdir;
  // If a session is already running, delete it before starting a new one.
  delete global_session;
  // Create returns a unique_ptr, we release it to get the raw pointer.
  global_session = tsl::ProfilerSession::Create(opts).release();
  if (global_session) {
    TT_THROW_IF_ERROR(global_session->Status())
        << "failed to start profiler session";
  }
}

void StopTrace(const std::string& filename) {
  if (global_session) {
    tensorflow::profiler::XSpace xspace;
    TT_THROW_IF_ERROR(global_session->CollectData(&xspace))
        << "failed to collect trace data";
    ABSL_LOG(INFO) << "Collected " << xspace.planes_size() << " planes.";
    tsl::Env* env = tsl::Env::Default();

    // Ensure the parent directory exists
    std::string dirname = std::string(tsl::io::Dirname(filename));
    if (!dirname.empty()) {
      TT_THROW_IF_ERROR(env->RecursivelyCreateDir(dirname))
          << "failed to create directory: " << dirname;
    }

    std::unique_ptr<tsl::WritableFile> outfile;
    TT_THROW_IF_ERROR(env->NewWritableFile(filename, &outfile))
        << "failed to create file: " << filename;

    // Serialize the collected XSpace data to a string.
    std::string serialized_proto;
    TT_CHECK_THROW(xspace.SerializeToString(&serialized_proto),
                   error::kInternal)
        << "failed to serialize profile data";
    ABSL_LOG(INFO) << "Writing " << serialized_proto.size() << " bytes to "
                   << filename;

    // Write the serialized XSpace to the file.
    TT_THROW_IF_ERROR(outfile->Append(serialized_proto))
        << "failed to write data to " << filename;
    TT_THROW_IF_ERROR(outfile->Close()) << "failed to close file: " << filename;
    delete global_session;
    global_session = nullptr;
  }
}

PYBIND11_MODULE(_profiler_backend, m) {
  m.doc() = "PjRt backend for PyTorch profiler.";

  m.def("start_trace", &torch_tpu::StartTrace, py::arg("logdir"),
        py::arg("options") = py::none(), "Starts profiler trace.");
  m.def("stop_trace", &torch_tpu::StopTrace);
  m.def("start_profiler_server", &torch_tpu::StartProfilerServer,
        py::arg("port"), "Starts the profiler gRPC server on the given port.");
  m.def("stop_profiler_server", &torch_tpu::StopProfilerServer,
        "Stops the profiler gRPC server.");
}

}  // namespace torch_tpu
