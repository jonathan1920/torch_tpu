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

load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "pybind_extension")
load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

pytype_strict_library(
    name = "profiler",
    srcs = [
        "__init__.py",
        "_impl.py",
        "profiler_api.py",
    ],
    deps = [
        ":_profiler_backend",
    ],
)

pybind_extension(
    name = "_profiler_backend",
    srcs = [
        "profiler.cc",
        "profiler.h",
    ],
    # Torch_tpu uses rules_pywrap in XLA for OSS builds:
    # https://github.com/openxla/xla/tree/main/third_party/py/rules_pywrap.
    # These args are required to link extensions against common libs.
    common_lib_packages = [
        "//torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/backends/profiler/cpu:host_tracer",
        "@xla//xla/backends/profiler/cpu:python_tracer",
        "@xla//xla/backends/profiler/tpu:tpu_tracer",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
    ],
)

torch_tpu_package_end()
