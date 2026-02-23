load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "pybind_extension")
load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_cc_library")

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("//shims/py_rules:pytype.bzl", "pytype_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "sync_cc",
    srcs = [
        "sync.cc",
    ],
    hdrs = [
        "sync.h",
    ],
    deps = [
        "//shims/torch:aten_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/eager:traversal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla/hlo/translate:register",
    ],
)

torch_tpu_cc_library(
    name = "sync_py_cc",
    srcs = ["sync_py.cc"],
    deps = [
        ":sync_cc",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

pytype_library(
    name = "sync",
    srcs = [
        "__init__.py",
        "sync.py",
    ],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        ":_tpu_torch_sync",
        "//shims/torch:pytorch",
    ],
)

pybind_extension(
    name = "_tpu_torch_sync",
    srcs = ["sync_py.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    # The pybind import test doesn't work with "third_party/py/torch/torch/extension.h"
    # for some reason; complains about missing symbol pybind1::detail::type_caster::at::Tensor[...]
    wrap_py_init = True,
    deps = [
        ":sync_cc",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
