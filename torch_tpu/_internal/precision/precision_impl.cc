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

#include "torch_tpu/ops/precision_context.h"
#include "pybind11/pybind11.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

PYBIND11_MODULE(precision_impl, m) {
  pybind11::enum_<mlir::stablehlo::Precision>(m, "Precision")
      .value("default", mlir::stablehlo::Precision::DEFAULT)
      .value(
          "high",
          mlir::stablehlo::Precision::HIGH)  // EXPLICIT_PRECISION_OK=root usage
      .value("highest",
             mlir::stablehlo::Precision::HIGHEST)  // EXPLICIT_PRECISION_OK=root
                                                   // usage
      .export_values();

  m.def("_get_precision", &GetPrecision, "Internal get for context manager");
  m.def("_push_precision", &PushPrecision, "Internal push for context manager");
  m.def("_pop_precision", &PopPrecision, "Internal pop for context manager");
}

}  // namespace torch_tpu
