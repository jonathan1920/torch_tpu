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

#ifndef TORCH_TPU_INTERNAL_PROFILER_PROFILER_H_
#define TORCH_TPU_INTERNAL_PROFILER_PROFILER_H_

#include <string>

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace torch_tpu {

void StartProfilerServer(int port);

void StopProfilerServer();

void StartTrace(const std::string& logdir, py::object options);

void StopTrace(const std::string& filename);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_PROFILER_PROFILER_H_
