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

"""Documentation for TorchTPU."""

load("@rules_python//python:py_library.bzl", "py_library")
load("//build_files:build_defs.bzl", "define_cpp_filegroup")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//torch_tpu/docs:__subpackages__"],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "index",
    srcs = ["index.py"],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "get_started_tpu",
    srcs = ["get_started_tpu.py"],
    deps = ["//shims/torch:pytorch"],
)

py_library(
    # PY_LIBRARY_OK=used in OSS only.
    name = "tpu_utils",
    srcs = ["tpu_utils.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/api:_device_module",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
