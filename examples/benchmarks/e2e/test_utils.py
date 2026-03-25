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

"""Utility functions for benchmarks."""

from typing import Any, Sequence

from absl.testing import parameterized
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import performance_utils

# CUDA only has EAGER and COMPILED run modes. Other run modes are applicable to
# TPU only.
CUDA_RUN_MODES = (
    benchmark_utils.RunMode.EAGER,
    benchmark_utils.RunMode.COMPILED,
)


class BenchmarkTest(parameterized.TestCase):
  """Tests for end-to-end performance benchmarks."""

  def run_performance_benchmark_test(
      self,
      config: performance_utils.PerformanceBenchmarkConfig,
      benchmark_name: str,
      microbenchmark_name: str | None = None,
  ) -> None:
    """Runs a benchmark test.

    Args:
      config: The benchmark config.
      benchmark_name: The name of the benchmark. This should match the last part
        of target in the MLCompass config file. See
        go/torchtpu-mlcompass#configuration-structure for more details.
      microbenchmark_name: This is used to export microbenchmark results to
        MLCompass. If a benchmark test is composed of multiple microbenchmarks,
        this should be set to the name of the microbenchmark. For example, when
        testing a linear layer with different tensor shapes, each one should be
        exported to MLCompass as a microbenchmark. See
        go/mlcompass-microbenchmark-guide for more details.
    """

    platform = benchmark_utils.PLATFORM.value
    if platform not in config.supported_platforms:
      self.skipTest(
          f"Platform {benchmark_utils.PLATFORM.value} not in"
          f" {config.supported_platforms}"
      )
    if (
        platform
        in (
            benchmark_utils.Platform.B200_8,
            benchmark_utils.Platform.B200_4,
            benchmark_utils.Platform.B200_1,
        )
        and config.run_mode not in CUDA_RUN_MODES
    ):
      self.skipTest(
          f"Run mode {config.run_mode} not applicable to platform {platform}"
      )

    performance_utils.run_benchmark(
        config=config,
        test_method_name=self._testMethodName,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )


def generate_run_mode_and_train_configs(
    run_modes: Sequence[Any],
    is_training: Sequence[Any],
):
  """Generates test parameters from a list of run modes and training modes.

  Args:
    run_modes: The run modes to generate test parameters for.
    is_training: The training modes to generate test parameters for.

  Yields:
    A dictionary containing the test parameters.
  """
  for training_mode in is_training:
    for run_mode in run_modes:
      name_parts = []
      name_parts.append("train" if training_mode else "eval")
      name_parts.append(f"{run_mode.value}")
      testcase_name = "_".join(name_parts)
      yield dict(
          testcase_name=testcase_name,
          run_mode=run_mode,
          is_training=training_mode,
      )


def generate_run_mode_configs(
    run_modes: Sequence[Any],
):
  """Generates test parameters from a list of run modes.

  Args:
    run_modes: The run modes to generate test parameters for.

  Yields:
    A dictionary containing the test parameters.
  """
  for run_mode in run_modes:
    yield dict(
        testcase_name=run_mode.value,
        run_mode=run_mode,
    )
