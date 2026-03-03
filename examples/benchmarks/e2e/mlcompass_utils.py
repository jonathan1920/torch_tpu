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

"""Utilities for exporting benchmark results to MLCompass."""

from absl import logging
from examples.benchmarks.e2e import benchmark_utils
from torch_tpu._internal.shims.mlcompass import benchmark_data as benchmark_data_lib
from torch_tpu._internal.shims.mlcompass import export_lib_borg


TEAM_NAME = "torch_tpu"


# TODO(b/470090396): Export to MLCompass for quality benchmarking as well.
def export_to_mlcompass(
    platform: benchmark_utils.Platform,
    metrics: benchmark_utils.BenchmarkResultInterface,
    base_cl: str | None,
    mlcompass_tracking_id: str,
    mlcompass_execution_mode: str,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Exports benchmark results to mlcompass.

  Args:
    platform: The platform the benchmark was run on.
    metrics: The performance benchmark results.
    base_cl: The base CL used for the benchmark run.
    mlcompass_tracking_id: The UUID used to track all the generated metrics by a
      single invocation.
    mlcompass_execution_mode: The execution mode of the mlcompass run. This is
      used to distinguish between cbuild and oneshot runs.
    test_method_name: The name of the test method being benchmarked.
    benchmark_name: The name of the benchmark.
    microbenchmark_name: The name of the microbenchmark. This is used to export
      microbenchmark results to MLCompass when a benchmark test method is
      composed of multiple microbenchmarks. See
      go/mlcompass-microbenchmark-guide for more details.
  """

  assert mlcompass_tracking_id is not None
  logging.info(
      "Exporting results to mlcompass for tracking-id: %s, test-name: %s,"
      " benchmark-name: %s, microbenchmark-name: %s",
      mlcompass_tracking_id,
      test_method_name,
      benchmark_name,
      microbenchmark_name,
  )
  # MLCompass expects metrics as mapping from string (metric name) -> float
  # (metric value). It displays a graph for each metric name with metric value
  # across time. We need to make sure that the strings are uniform across all
  # benchmarks for displaying on dashboards but there are no restrictions on
  # what the names should be.
  metric_map = metrics.metric_map()
  test_name = (
      f"{TEAM_NAME}/{platform.value}/{test_method_name}/{benchmark_name}"
  )
  benchmark_data = benchmark_data_lib.BenchmarkData(
      test_name=test_name,
      wall_time=metrics.e2e_wall_time_seconds,
      base_cl=base_cl,
      metrics=metric_map,
      mlcompass_tracking_id=mlcompass_tracking_id,
      mlcompass_execution_mode=mlcompass_execution_mode,
      team_name=TEAM_NAME,
      succeeded=True,
  )
  if microbenchmark_name:
    benchmark_data.micro_result_key = microbenchmark_name
  export_lib_borg.export_results_to_mlcompass(data=benchmark_data)
