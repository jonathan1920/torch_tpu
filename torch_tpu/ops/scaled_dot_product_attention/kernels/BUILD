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

load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_cc_library", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "scaled_dot_product_attention_kernels",
    srcs = ["scaled_dot_product_attention_kernels.py"],
    deps = [
        "//torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
        "@pypi//jax",
    ],
)

pytype_strict_binary(
    name = "scaled_dot_product_attention_generate",
    srcs = ["scaled_dot_product_attention_generate.py"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    deps = [
        ":scaled_dot_product_attention_kernels",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
    ],
)

torch_tpu_py_test(
    name = "scaled_dot_product_attention_kernels_test",
    srcs = ["scaled_dot_product_attention_kernels_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":scaled_dot_product_attention_kernels",
        "//shims/absl_py/testing:absltest",
        "@pypi//jax",
    ],
)

genrule(
    name = "scaled_dot_product_attention_forward_mlir",
    outs = [
        "scaled_dot_product_attention_forward_mlir_embed.h",
        "scaled_dot_product_attention_forward_mlir_embed.cc",
    ],
    cmd = """
        set -e
        read -ra out_files <<< "$(OUTS)"
        h_path=$${out_files[0]}
        cc_path=$${out_files[1]}
        $(location :scaled_dot_product_attention_generate) --forward \
            --header $$h_path \
            --implementation $$cc_path
    """,
    tools = [":scaled_dot_product_attention_generate"],
)

torch_tpu_cc_library(
    name = "scaled_dot_product_attention_forward_mlir_embed",
    srcs = [
        "scaled_dot_product_attention_forward_mlir_embed.cc",
    ],
    hdrs = [
        "scaled_dot_product_attention_forward_mlir_embed.h",
    ],
)

genrule(
    name = "scaled_dot_product_attention_backward_mlir",
    outs = [
        "scaled_dot_product_attention_backward_mlir_embed.h",
        "scaled_dot_product_attention_backward_mlir_embed.cc",
    ],
    cmd = """
        set -e
        read -ra out_files <<< "$(OUTS)"
        h_path=$${out_files[0]}
        cc_path=$${out_files[1]}
        $(location :scaled_dot_product_attention_generate) --backward \
            --header $$h_path \
            --implementation $$cc_path
    """,
    tools = [":scaled_dot_product_attention_generate"],
)

torch_tpu_cc_library(
    name = "scaled_dot_product_attention_backward_mlir_embed",
    srcs = [
        "scaled_dot_product_attention_backward_mlir_embed.cc",
    ],
    hdrs = [
        "scaled_dot_product_attention_backward_mlir_embed.h",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
