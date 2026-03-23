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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/mpm:genmpm.bzl", "genmpm")
load("//shims/pkg:pkg.bzl", "pkg_binary_with_runfiles")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "model_utils",
    testonly = True,
    srcs = ["model_utils.py"],
    deps = [
        "//shims/flax_nnx",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

pytype_strict_library(
    name = "performance_utils",
    testonly = True,
    srcs = ["performance_utils.py"],
    deps = [
        ":model_utils",
        "//examples/benchmarks/e2e:benchmark_utils",
        "//examples/benchmarks/e2e:mlcompass_utils",
        "//examples/benchmarks/e2e:performance_utils",
        "//shims/absl_py/logging",
        "//shims/flax_nnx",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "layer_performance_benchmarks",
    timeout = "long",
    srcs = ["layer_performance_benchmarks.py"],
    main = "layer_performance_benchmarks.py",
    notap = "Tested on guitar workflow.",  # NOTAP_OK=Tested on guitar workflow.
    strict = True,
    tags = [
        "cpu:4",

        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":performance_utils",
        "//examples/benchmarks/e2e:benchmark_utils",
        "//examples/benchmarks/e2e:layer_configs",
        "//examples/benchmarks/e2e:performance_utils",
        "//examples/benchmarks/e2e:test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

pkg_binary_with_runfiles(
    name = "layer_performance_benchmarks_with_runfiles",
    testonly = 1,
    binary = ":layer_performance_benchmarks",
)

# Using a temporal MPM instead of a durable MPM as these are only used for testing. It also allows
# testing changes from a local workspace.
genmpm(
    name = "layer_performance_benchmarks_mpm",
    testonly = 1,
    temporal = True,
    deps = [":layer_performance_benchmarks_with_runfiles"],
)

torch_tpu_package_end()
