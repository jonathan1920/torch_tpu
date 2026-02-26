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

load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "a_min_max_aten_kernels",
    srcs = ["a_min_max_aten_kernels.cc"],
    hdrs = ["a_min_max_aten_kernels.h"],
    deps = [
        ":a_min_max",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_types",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/reductions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/status",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "a_min_max",
    srcs = ["a_min_max.cc"],
    hdrs = ["a_min_max.h"],
    deps = [
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops/reductions",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
