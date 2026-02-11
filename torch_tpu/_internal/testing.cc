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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "torch_tpu/eager/op_dispatcher.h"

ABSL_DECLARE_FLAG(std::optional<bool>, torch_tpu_internal_mlir_tracebacks);

namespace torch_tpu {
namespace py = pybind11;

// Internal testing utilities.
PYBIND11_MODULE(testing, m) {
  // Forces DynamicDispatchOp() to fail with the given message for ops whose
  // base name matches `op_base_name`. If `op_base_name` is empty, no op is
  // forced to fail.
  //
  // This is NOT accumulative. If you call this multiple times, only the last
  // call will take effect.
  m.def("set_op_dispatch_failure", internal::SetOpDispatchFailure,  //
        py::arg("op_base_name"), py::arg("failure_message"));

  // Exposes the C++ flag accessors.
  m.def("set_mlir_tracebacks_flag", [](std::optional<bool> val) {
    absl::SetFlag(&FLAGS_torch_tpu_internal_mlir_tracebacks, val);
  });
  m.def("get_mlir_tracebacks_flag", []() -> std::optional<bool> {
    return absl::GetFlag(FLAGS_torch_tpu_internal_mlir_tracebacks);
  });
}

}  // namespace torch_tpu
