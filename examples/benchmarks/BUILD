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

load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "ml_layers_test_cuda",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=cuda",
        "--framework=pytorch",
    ],
    main = "ml_layers_test.py",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "manual",  # Run this test only manually, since it needs --config=cuda
        "notap",  # NOTAP_OK=this is a manual test
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "ml_layers_test_tpu",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=tpu",
        "--framework=pytorch",
    ],
    main = "ml_layers_test.py",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # Matches the CUDA version of the test for a fair comparison.
        "nofastbuild",  # Run in postsubmit only to avoid long Forge queues.
        "oss_ready_tpu",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "ml_layers_test_xla_cuda",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=xla_cuda",
        "--framework=pytorch",
    ],
    main = "ml_layers_test.py",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "manual",  # Run this test only manually, since it needs --config=cuda
        "notap",  # NOTAP_OK=this is a manual test
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
