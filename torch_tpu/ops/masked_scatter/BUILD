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
    name = "masked_scatter_aten_kernels",
    srcs = ["masked_scatter_aten_kernels.cc"],
    hdrs = ["masked_scatter_aten_kernels.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/ops:nullary_aten_kernels",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:xla_utils",
        "//torch_tpu/ops/copy_from:copy_from_aten_kernels",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/reductions",
        "//torch_tpu/ops/reductions:reduction_utils",
        "//torch_tpu/ops/scatter:scatter_aten_kernels",
        "//torch_tpu/ops/sort:sort_aten_kernels",
        "//torch_tpu/ops/view:view_aten_kernels",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_builder",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
