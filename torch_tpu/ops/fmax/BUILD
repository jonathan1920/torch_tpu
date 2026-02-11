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

torch_tpu_cc_library(
    name = "fmax_aten_kernels",
    srcs = ["fmax_aten_kernels.cc"],
    hdrs = ["fmax_aten_kernels.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:binary_aten_kernels",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
