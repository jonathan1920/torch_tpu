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

"""Tests for TorchTPU documentation."""

load("@rules_python//python:py_library.bzl", "py_library")
load("//build_files:build_defs.bzl", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_index",
    srcs = ["test_index.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/docs/notebooks:index",
    ],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_get_started_tpu",
    srcs = ["test_get_started_tpu.py"],
    deps = ["//shims/torch:pytorch"],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_porting_vit_mnist",
    srcs = ["test_porting_vit_mnist.py"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//numpy",
        "@pypi//scikit-learn:pkg",
    ],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_numerical_parity",
    srcs = ["test_numerical_parity.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
    ],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_compilation_cache",
    srcs = ["test_compilation_cache.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "test_error_glossary",
    srcs = ["test_error_glossary.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_package_end()
