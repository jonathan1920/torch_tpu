load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_py_test")

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
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "small_graph_test",
    srcs = ["small_graph_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "ensure_no_dynamic_test",
    srcs = ["ensure_no_dynamic_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "compile_with_distributed_test",
    srcs = ["compile_with_distributed_test.py"],
    env = {"ACCELERATOR_TYPE": "v5litepod-8"},
    shard_count = 1,
    tags = [
        "exclusive",
        "oss_ready_tpu",
        "requires-tpu-v5lite:8",
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/distributed:tpu_env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",  # buildcleaner: keep
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "compile_api_test",
    srcs = ["compile_api_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/api",
    ],
)

pytype_strict_library(
    name = "stablehlo_compile_testing",
    srcs = ["stablehlo_compile_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/api",
        "//torch_tpu/api:_device_ops_backend",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "stablehlo_compile_test",
    size = "enormous",
    srcs = ["stablehlo_compile_test.py"],
    args = [
        "--enable_tensorboard_logging=False",
    ],
    linking_mode = "static",
    main = "stablehlo_compile_test.py",
    tags = [
        "cpu:1",  # Ensure reproducible performance on Forge.
        "manual",  # Exclude from wildcard test patterns. Still runs on TAP.
        "nofastbuild",  # Run this in opt build mode only.
        "notap",  # # NOTAP_OK=Disable on TAP until we can get faster performance or reduce number of runs. Currently runs ~1h.
        "oss_ready_tpu",
        "requires-tpu-v6e",
    ],
    deps = [
        ":stablehlo_compile_testing",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "output_modify_test",
    srcs = ["output_modify_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "compile_with_inference_mode_test",
    srcs = ["compile_with_inference_mode_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
