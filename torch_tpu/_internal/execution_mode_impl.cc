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
#include "torch_tpu/eager/eager_mode.h"
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
  py::enum_<EagerMode>(m, "EagerMode")
      .value("DEFAULT", EagerMode::kDefault)
      .value("OPTIMIZED", EagerMode::kOptimized)
      .value("DEFER_ALL", EagerMode::kDeferAll)
      .value("DEFER_NEVER", EagerMode::kDeferNever)
      .export_values();

  m.def("get_eager_mode", GetEagerMode);
  m.def("set_eager_mode", SetEagerMode, py::arg("eager_mode"));

  m.def("enable_cpu_fallback", EnableCpuFallback, py::arg("enabled"));
  m.def("is_cpu_fallback_enabled", IsCpuFallbackEnabled);
}
}  // namespace torch_tpu
