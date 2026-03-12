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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = [
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:macro_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_names",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    tags = [
        "oss_ready_cpu",
    ],
    deps = [
        ":logging",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:macro_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [
        ":logging",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    tags = [
        "oss_ready_cpu",
    ],
    deps = [
        ":kernel",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_names",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_package_end()
