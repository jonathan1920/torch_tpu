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

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/build_defs_testing:bzl_library.bzl", "bzl_library")
load(":defs.bzl", "define_scaled_mm_kernel")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# Hermetic build-time kernel generation targets for scaled mm.
#
# Extension Point:
# Additional block-scaling kernel libraries can be instantiated here using
# `define_scaled_mm_kernel` by specifying the target recipe name (registered in
# `scaled_mm_generate.py`), data types, and tile geometries.

define_scaled_mm_kernel(
    name = "blockwise_scaled_mm_e4m3",
    in_dtype = "float8_e4m3fn",
    kernel_name = "kBlockwiseScaledMmE4m3",
    recipe = "mxfp8",
)

define_scaled_mm_kernel(
    name = "blockwise_scaled_mm_e5m2",
    in_dtype = "float8_e5m2",
    kernel_name = "kBlockwiseScaledMmE5m2",
    recipe = "mxfp8",
)

bzl_library(
    name = "defs_bzl",
    srcs = ["defs.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
    deps = ["//build_files:build_defs_bzl"],
)

torch_tpu_package_end()
