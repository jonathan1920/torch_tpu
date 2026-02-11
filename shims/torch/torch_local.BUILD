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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")

package(default_visibility = ["//visibility:public"])

# Expose all header files in torch/include
# We exclude conflicting third-party deps and explicitly depend on the Bazel versions
# to avoid ODR violations and missing headers.
cc_library(
    name = "torch_headers",
    hdrs = glob(
        # 1. Standard Install Layout
        ["include/**"],
        exclude = [
            "include/google/**",   # Use repo's protobuf, not PyTorch's bundled one
        ],
    ) + glob(
        # 2. Local Source/Develop Layout
        ["torch/include/**"],
        exclude = [
            "torch/include/google/**", # Exclude Protobuf (Conflict)
            # DO NOT exclude pybind11. We want the bundled version.
        ],
    ) + glob(
        # 3. Direct Source for Pybind11 (Keep as backup)
        ["third_party/pybind11/include/**"],
    ) + glob(
        # 4. MISSING PIECE: Header-only utilities (like util/complex.h)
        ["torch/headeronly/**"],
    ),
    includes = [
        # Standard paths
        "include",
        "include/torch/csrc/api/include",

        # Source/Develop paths
        "torch/include",
        "torch/include/torch/csrc/api/include",

        # Local pybind11 path
        "third_party/pybind11/include",

        # Header-only utilities
        "torch/headeronly",
    ],
)

# Use cc_import for prebuilt libraries to properly handle RPATH and DT_NEEDED
cc_import(
    name = "libc10",
    shared_library = "torch/lib/libc10.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "torch/lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "torch/lib/libtorch_python.so",
)

# Catch-all target for convenience
cc_library(
    name = "torch_libs",
    deps = [
        ":libc10",
        ":libtorch_cpu",
        ":libtorch_python",
    ],
)