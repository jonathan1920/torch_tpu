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

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/layout_utils.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

void PyEnterLayoutContext(const LayoutAnnotation& layout_hint) {
  std::vector<xla::Tile> xla_tiles;
  xla_tiles.reserve(layout_hint.tiles.size());
  for (const auto& tile_dims : layout_hint.tiles) {
    xla_tiles.push_back(xla::Tile(tile_dims));
  }
  xla::Layout layout(layout_hint.minor_to_major, xla_tiles,
                     layout_hint.element_size_in_bits);
  PushContextState<LayoutContextState>(
      std::make_shared<xla::Layout>(std::move(layout)));
}

void PyExitLayoutContext() { PopContextState<LayoutContextState>(); }

}  // namespace

PYBIND11_MODULE(annotations_py, m) {
  py::class_<LayoutAnnotation>(m, "TpuLayout")
      .def(
          py::init<std::vector<int64_t>,               // INT_VEC_OK
                   std::vector<std::vector<int64_t>>,  // INT_VEC_OK
                   int64_t>(),
          py::arg("minor_to_major"),
          py::arg("tiles") = std::vector<std::vector<int64_t>>(),  // INT_VEC_OK
          py::arg("element_size_in_bits") = 0)
      .def_readwrite("minor_to_major", &LayoutAnnotation::minor_to_major)
      .def_readwrite("tiles", &LayoutAnnotation::tiles)
      .def_readwrite("element_size_in_bits",
                     &LayoutAnnotation::element_size_in_bits)
      .def("__eq__", &LayoutAnnotation::operator==)
      .def("__len__", [](const LayoutAnnotation&) { return 3; })
      .def("__getitem__",
           [](const LayoutAnnotation& self, int64_t index) -> py::object {
             if (index == 0) return py::cast(self.minor_to_major);
             if (index == 1) return py::cast(self.tiles);
             if (index == 2) return py::cast(self.element_size_in_bits);
             throw py::index_error(  // pybind exception ok
                 "index out of range (must be 0, 1, or 2)");  // pybind
                                                              // exception ok
           })
      .def("__repr__", [](const LayoutAnnotation& self) {
        std::stringstream ss;
        ss << "TpuLayout(minor_to_major=[";
        for (size_t i = 0; i < self.minor_to_major.size(); ++i) {
          ss << self.minor_to_major[i];
          if (i + 1 < self.minor_to_major.size()) ss << ", ";
        }
        ss << "], tiles=[";
        for (size_t i = 0; i < self.tiles.size(); ++i) {
          ss << "[";
          for (size_t j = 0; j < self.tiles[i].size(); ++j) {
            ss << self.tiles[i][j];
            if (j + 1 < self.tiles[i].size()) ss << ", ";
          }
          ss << "]";
          if (i + 1 < self.tiles.size()) ss << ", ";
        }
        ss << "], element_size_in_bits=" << self.element_size_in_bits << ")";
        return ss.str();
      });

  m.def("enter_layout_context", &PyEnterLayoutContext, py::arg("layout"));
  m.def("exit_layout_context", &PyExitLayoutContext);
}

}  // namespace torch_tpu
