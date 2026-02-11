# Copyright 2026 Google LLC
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

define_cpp_filegroup(name = "all_cpp_files_recursive")

torch_tpu_cc_library(
    name = "linalg_lu_kernels",
    srcs = ["linalg_lu_kernels.cc"],
    hdrs = ["linalg_lu_kernels.h"],
    deps = [
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/linalg/solve_triangular:linalg_solve_triangular_kernels",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)
