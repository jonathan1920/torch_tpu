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
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "benchmark_utils",
    srcs = ["benchmark_utils.py"],
    deps = [
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/xprof:xprof_analysis_client",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/utils:device_utils",
        "//torch_tpu/api",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "mlcompass_utils",
    srcs = ["mlcompass_utils.py"],
    deps = [
        ":benchmark_utils",
        "//shims/absl_py/logging",
        "//torch_tpu/_internal/shims/mlcompass:benchmark_data",
        "//torch_tpu/_internal/shims/mlcompass:export_lib_borg",
    ],
)

pytype_strict_library(
    name = "performance_utils",
    testonly = 1,
    srcs = ["performance_utils.py"],
    deps = [
        ":benchmark_function_db",
        ":benchmark_utils",
        ":mlcompass_utils",
        ":model_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:gpu_env",
        "//torch_tpu/_internal/distributed:tpu_env",
        "//torch_tpu/_internal/shims/xprof:xprof_analysis_client",
        "//torch_tpu/_internal/utils:device_utils",
    ],
)

pytype_strict_library(
    name = "quality_utils",
    testonly = 1,
    srcs = ["quality_utils.py"],
    deps = [
        ":benchmark_utils",
        ":mlcompass_utils",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//examples/benchmarks/quality_utils/metrics:data_loader",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:gpu_env",
        "//torch_tpu/_internal/distributed:tpu_env",
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        "//shims/absl_py/testing:parameterized",
    ],
)

torch_tpu_py_test(
    name = "model_performance_benchmarks",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_performance_benchmarks.py",
    shard_count = 4,
    strict = True,
    tags = [
        "manual",  # Skip slow tests from local runs by default.
        "cpu:4",
        "notap",  # NOTAP_OK=Tested on guitar workflow.
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "model_quality_benchmarks",
    timeout = "long",
    srcs = ["model_quality_benchmarks.py"],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_quality_benchmarks.py",
    shard_count = 4,
    strict = True,
    tags = [
        "manual",  # Skip slow tests from local runs by default.
        "cpu:4",
        "notap",  # NOTAP_OK=Tested on guitar workflow.
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":quality_utils",
        "//examples/benchmarks/quality_utils/metrics:data_loader",
        "//examples/benchmarks/quality_utils/metrics:perplexity_metric",
        "//examples/llama3/distributed_meta_llama3:meta_llama3_quality_benchmark",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "layer_performance_benchmarks",
    timeout = "long",
    srcs = ["layer_performance_benchmarks.py"],
    main = "layer_performance_benchmarks.py",
    shard_count = 2,
    strict = True,
    tags = [
        "manual",  # Skip slow tests from local runs by default.
        "cpu:4",
        "notap",  # NOTAP_OK=Tested on guitar workflow.
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "layer_performance_benchmarks_cuda",
    timeout = "long",
    srcs = ["layer_performance_benchmarks.py"],
    args = ["--platform=b200_1"],
    main = "layer_performance_benchmarks.py",
    shard_count = 2,
    strict = True,
    tags = [
        "manual",  # Skip slow tests from local runs by default.
        "cpu:4",
        "notap",  # NOTAP_OK=Tested on guitar workflow.
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-gpu-sm100-full-only",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "distributed_benchmark_tests",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    args = [
        "--platform=gfc_2x2x1",
    ],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_performance_benchmarks.py",
    tags = [
        "manual",  # Skip slow tests from local runs by default.
        "cpu:4",
        "notap",  # NOTAP_OK=Tested on guitar workflow.
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x:4",
        "nofastbuild",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "model_benchmark_cuda_tests",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    args = ["--platform=b200_1"],
    main = "model_performance_benchmarks.py",
    shard_count = 2,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "manual",  # Run only manually since it needs --config=cuda
        "nofastbuild",
        "notap",  # NOTAP_OK=frequently causes Forge OOM.
        "requires-accel-hbm",
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

pytype_strict_library(
    name = "benchmark_function_db",
    testonly = 1,
    srcs = ["benchmark_function_db.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils:device_utils",
    ],
)

pytype_strict_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/model_runner_hf:modeling_hf",
        "//torch_tpu/_internal/utils:device_utils",
        "@pypi//fairscale",
        "@pypi//llama_models",
    ],
)

define_cpp_filegroup(name = "all_cpp_files_recursive")
