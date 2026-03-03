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

load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "pybind_extension")
load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")
load("//shims/py_rules:pytype.bzl", "pytype_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "dynamism_cc",
    srcs = [
        "dynamism.cc",
    ],
    hdrs = [
        "dynamism.h",
    ],
    deps = [
        "//shims/torch:aten_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_py_cc",
    srcs = ["dynamism_py.cc"],
    deps = [
        ":dynamism_cc",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_types",
        "//torch_tpu/ops:op_builder_utils",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_ops",
    srcs = ["dynamism_ops.cc"],
    hdrs = ["dynamism_ops.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fixed_size_span",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/pjrt:pjrt_utils",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@xla//xla:xla_data_proto_cc",
    ],
)

pybind_extension(
    name = "_tpu_torch_dynamism",
    srcs = ["dynamism_py.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    # The pybind import test doesn't work with "third_party/py/torch/torch/extension.h"
    # for some reason; complains about missing symbol pybind1::detail::type_caster::at::Tensor[...]
    wrap_py_init = True,
    deps = [
        ":dynamism_cc",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_types",
        "//torch_tpu/ops:op_builder_utils",
    ],
)

pytype_library(
    name = "dynamism",
    srcs = [
        "__init__.py",
    ],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [":_tpu_torch_dynamism"],
)

torch_tpu_package_end()
