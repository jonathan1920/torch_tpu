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

#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>

#include "pybind11/pybind11.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
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

  // If a session is already running, delete it before starting a new one.
  delete global_session;
  // Create returns a unique_ptr, we release it to get the raw pointer.
  global_session = tsl::ProfilerSession::Create(opts).release();
}

void StopTrace(const std::string& filename) {
  if (global_session) {
    tensorflow::profiler::XSpace xspace;

    // (void) casts away the nodiscard warning
    (void)global_session->CollectData(&xspace);

    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (outfile.good()) {
      xspace.SerializeToOstream(&outfile);
      outfile.close();
    }
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
