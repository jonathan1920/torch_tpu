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

load("@rules_python//python:py_library.bzl", "py_library")
load("//build_files:build_defs.bzl", "define_cpp_filegroup", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
)

pytype_strict_library(
    name = "module_registry",
    srcs = ["module_registry.py"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "@pypi//timm",
        "@pypi//torchvision",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "module_registry_test",
    timeout = "long",
    srcs = ["module_registry_test.py"],
    shard_count = 6,
    tags = [
        "requires-net:external",
    ],
    deps = [
        ":module_registry",
        "//shims/absl_py/testing:absltest",
    ],
)

py_library(
    # PY_LIBRARY_OK=type checking too slow for this
    name = "op_testing",
    testonly = True,
    srcs = ["op_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

py_library(
    # PY_LIBRARY_OK=type checking too slow for this
    name = "dynamism_test_utils",
    testonly = True,
    srcs = ["dynamism_test_utils.py"],
    deps = [
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils",
    ],
)

# We have multiple targets for ops_test.py:
# 1. ops_test_tpu: TPU vs CPU.
# 2. ops_test_xla_cuda: XLA:CUDA vs CPU.
# 3. ops_test_xla_cpu: XLA:CPU vs CPU.
# 4. ops_test_tpu_vs_gpu: TPU vs GPU.
# 5. ops_test_gen_gpu_golden: generates the golden file for the GPU backend
# 6. ops_test_grad: gradient on TPU vs CPU.
# 7. ops_test_perf: TPU only, for measuring compilation/execution time.
# 8. ops_test_dynamic: TPU only, bounds a dimension and compares TPU vs CPU.
# 9. ops_test_compiled: TPU only, torch.compile TpuBackend vs CPU aot_eager backend.

#
# TorchTPU can run on "tpu" or "xla_cuda" or "xla_cpu" devices using the --torch_tpu_device flag.
#

# (1) and (2) are identical except for the compilation mode. We chose to
# manually maintain the duplicate tests instead of using a for-loop to
# generate the targets because:
#
# - The for-loop approach is incompatible with build_cleaner.
# - The for-loop approach breaks the Cider-torch_tpu_py_test integration (in particular,
#   it breaks running a single test case directly in the Cider UI).

torch_tpu_py_test(
    name = "ops_test_tpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    shard_count = 50,
    tags = [
        # "oss_ready_tpu",  # Will re-enable this slow test until we have a proper way to speed it up.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_grad",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--compute_grad=True",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    shard_count = 50,
    tags = [
        # "oss_ready_tpu",  # Will re-enable this slow test until we have a proper way to speed it up.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_compiled",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--use_compiled=True",
        "--max_samples_per_op_dtype=1",  # No need for many samples to catch compiled mode bugs.
    ],
    main = "ops_test.py",
    shard_count = 10,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_xla_cuda",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=xla_cuda",
        "--max_samples_per_op_dtype=3",  # Reduce test samples as XLA:CUDA is not a priority.
    ],
    main = "ops_test.py",
    shard_count = 20,
    tags = ["requires-gpu-sm80-only"],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "ops_test_xla_cpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=xla_cpu",
        "--max_samples_per_op_dtype=3",  # Reduce test samples as XLA:CPU is not a priority.
    ],
    main = "ops_test.py",
    shard_count = 25,
    tags = [
        # This test takes almost 30 minutes to run in the OSS CI, which
        # uses just one machine for each build for now. It's not essential
        # for the OSS release, so we skip it for OSS.
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_tpu_vs_gpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=torch_tpu_vs_gpu",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    shard_count = 50,
    tags = [
        # "oss_ready_tpu",  # Will re-enable this slow test until we have a proper way to speed it up.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_perf",
    timeout = "long",  # 15 minutes. The test finishes in ~5 minutes, which is the default limit.
    srcs = ["ops_test.py"],
    args = [
        # go/keep-sorted start
        "--alsologtostderr",
        "--max_samples_per_op_dtype=10",
        "--print_op_inputs=2",
        "--test_mode=perf",
        # Needed to fix flag 'vmodule' parsing error when using C++ flag in Python with Bazel.
        "--undefok=vmodule",
        "--vmodule=compilation_cache=3",  # Log SHLO strings.
        # go/keep-sorted end
    ],
    main = "ops_test.py",
    shard_count = 50,
    tags = [
        "cpu:8",  # Specify the number of CPUs for reproducible results.
        "manual",  # Exclude from wildcard test patterns as it's slow.
        "nofastbuild",  # Run this in opt build mode only.
        "requires-net:external",  # For writing remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_dynamic",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--check_dynamism_using_seed=-1",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=1",  # No need to test many samples to catch dynamism bugs.
    ],
    main = "ops_test.py",
    shard_count = 16,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
    ],
    main = "ops_test.py",
    # Each shard generates one golden file that contains one slice of the
    # golden data. Therefore the smaller the count, the bigger each file
    # will be.
    #
    # We should minimize the shard count without making a golden file larger
    # than 100 MB (a limit imposed by GitHub). The problem with a large shard
    # count is that it makes updating the golden files more tedious as it's
    # still done manually.
    #
    # If we adjust this, we must also adjust the data attribute in the
    # ops_test_vs_gpu target.
    shard_count = 3,
    tags = [
        "manual",  # Skip slow test locally.
        "requires-gpu-sm80",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_unit_test",
    srcs = ["ops_unit_test.py"],
    main = "ops_unit_test.py",
    shard_count = 20,
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# Tests the tier-3 compilation cache.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache",
    srcs = ["ops_unit_test.py"],
    env = {
        # To use tier-3 cache, we must use tier-2 cache as well.
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "my_cache",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
    },
    main = "ops_unit_test.py",
    shard_count = 20,
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "compiler_option_test",
    srcs = ["compiler_option_test.py"],
    main = "compiler_option_test.py",
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "index_put_op_test",
    srcs = ["index_put_op_unit_test.py"],
    main = "index_put_op_unit_test.py",
    shard_count = 20,
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "nn_test",
    srcs = ["nn_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "error_testing",
    testonly = True,
    srcs = ["error_testing.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_tpu",
    srcs = ["errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "errors_test.py",
    shard_count = 4,
    tags = [
        "notap",  # NOTAP_OK=TODO(wan): Re-enable after the March 2026 TorchTPU release.
        "oss_ready_tpu",
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_cov",
    srcs = ["errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "errors_test.py",
    shard_count = 4,
    tags = [
        "notap",  # NOTAP_OK=TODO(wan): Re-enable after the March 2026 TorchTPU release.
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_cpu",
    srcs = ["errors_test.py"],
    args = ["--test_mode=cpu"],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    main = "errors_test.py",
    shard_count = 4,
    tags = [
        "notap",  # NOTAP_OK=TODO(wan): Re-enable after the March 2026 TorchTPU release.
        "oss_ready_cpu",
        "wheel_test_excluded",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_cpp_context_test",
    srcs = ["errors_cpp_context_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "booleans_test",
    srcs = ["booleans_test.py"],
    # args = [],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "backwards_test",
    srcs = ["backwards_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "untyped_storage_test",
    srcs = ["untyped_storage_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "view_test",
    srcs = ["view_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "zero_size_test",
    srcs = ["zero_size_test.py"],
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "sync_test",
    srcs = ["sync_test.py"],
    main = "sync_test.py",
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "set_test",
    srcs = ["set_test.py"],
    main = "set_test.py",
    tags = [
        "oss_ready_tpu",
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
    name = "execution_mode_test",
    srcs = ["execution_mode_test.py"],
    main = "execution_mode_test.py",
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dynamism_test",
    srcs = ["dynamism_test.py"],
    main = "dynamism_test.py",
    tags = [
        "oss_ready_tpu",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dynamism_model_test",
    srcs = ["dynamism_model_test.py"],
    main = "dynamism_model_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
