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
    name = "resize_aten_kernels",
    srcs = ["resize_aten_kernels.cc"],
    hdrs = ["resize_aten_kernels.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/ops:nullary_aten_kernels",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/as_strided:as_strided_aten_kernels",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/stride:stride_helper",
        "@com_google_absl//absl/log:absl_log",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:xla_data_proto_cc",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
