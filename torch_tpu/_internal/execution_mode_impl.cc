/*
 * Copyright 2025 Google LLC
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
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tpu_aten_kernels.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

enum class FallbackMode { kNoFallback, kAllowFallback };

}  // namespace

PYBIND11_MODULE(execution_mode_impl, m) {
  py::enum_<DeferMode>(m, "DeferMode")
      .value("DEFAULT", DeferMode::kDefault)
      .value("ALL", DeferMode::kAll)
      .value("NEVER", DeferMode::kNever)
      .export_values();

  m.def("set_defer_mode", SetDeferMode, py::arg("defer_mode"));
  m.def("get_defer_mode", GetDeferMode);

  m.def("enable_cpu_fallback", EnableCpuFallback, py::arg("enabled"));
  m.def("is_cpu_fallback_enabled", IsCpuFallbackEnabled);
}
}  // namespace torch_tpu
