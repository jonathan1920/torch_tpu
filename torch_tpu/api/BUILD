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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "pybind_extension")
load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_cc_library")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")
load("//shims/py_rules:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

pytype_strict_library(
    name = "api",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":_device_module",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/sync",
    ],
)

pytype_strict_library(
    name = "_device_module",
    srcs = ["_device_module.py"],
    deps = [
        ":_device_ops_backend",
        ":streams",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

pytype_strict_library(
    name = "streams",
    srcs = ["streams.py"],
    deps = [
    ],
)

py_strict_test(
    name = "device_module_test",
    srcs = ["device_module_test.py"],
    tags = [
        "oss_ready_cpu",
    ],
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

py_strict_test(
    name = "tpu_module_test",
    srcs = ["tpu_module_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

py_strict_test(
    name = "xla_cuda_module_test",
    srcs = ["xla_cuda_module_test.py"],
    tags = ["requires-gpu-sm60-only"],
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

py_strict_test(
    name = "xla_cpu_module_test",
    srcs = ["xla_cpu_module_test.py"],
    tags = [
        "oss_ready_cpu",
    ],
    deps = [
        ":_device_module",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that tpu_rt.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "device_rt",
    srcs = ["device_rt.cc"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/pjrt:pjrt_init",
        "//torch_tpu/pjrt:pjrt_shutdown",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# Ensures that :device_rt builds. Without this, TAP
# won't build it as there's no test depending on it.
build_test(
    name = "device_rt_build_test",
    tags = [
        "oss_ready_cpu",
    ],
    targets = [":device_rt"],
)

pybind_extension(
    name = "_device_ops_backend",
    srcs = ["device_rt.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:tpu_hooks",
        "//torch_tpu/pjrt:pjrt_init",
        "//torch_tpu/pjrt:pjrt_shutdown",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
