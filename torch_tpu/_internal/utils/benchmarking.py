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

"""Utility functions for benchmarking."""

import enum
import operator
from typing import Callable, Dict, List, TypedDict

import numpy as np
from torch.utils import tensorboard


Enum = enum.Enum
SummaryWriter = tensorboard.SummaryWriter


class Stat(Enum):
  MEAN = "Mean"
  MEDIAN = "Median_p50"
  P90 = "P90"
  P95 = "P95"
  P99 = "P99"
  STDDEV = "StdDev"
  VALUE = "Value"


class Comparison(enum.Enum):
  LESS = "LESS"
  GREATER = "GREATER"


# Operation mapping for each comparison type
_COMPARISON_OPERATORS: Dict[Comparison, Callable[[float, float], bool]] = {
    Comparison.LESS: operator.lt,
    Comparison.GREATER: operator.gt,
}


# Default set of statistical measures to compute for a metric profile.
# This list provides a standard, consistent set of statistics (Mean, Median,
# percentiles, etc.) to be used as the default when a metric profile is
# initialized.
_DEFAULT_STATS = [
    Stat.MEAN,
    Stat.MEDIAN,
    Stat.P90,
    Stat.P95,
    Stat.P99,
    Stat.STDDEV,
]


class ProfileData(TypedDict):
  """A type representing the data for a single metric profile.

  Attributes:
    unit: The unit of the metric (e.g., "ms", "s"). This is included in the
      metric tag that appears in TensorBoard and MLCompass.
    stats: A list of statistical measures to compute for the metric. Each stat
      will be recorded as a separate metric in TensorBoard and MLCompass.
    name: The name of the metric profile (e.g., "Inference/Latency"). This is
      used as the base for constructing TensorBoard metric tags, which appear in
      TensorBoard and MLCompass in the format <name>/<Stat>_<unit> (e.g.,
      "Inference/Latency/Mean_ms").
  """

  unit: str
  stats: List[Stat]
  name: str


def _make_profile_map(profiles: List[ProfileData]) -> dict[str, ProfileData]:
  """Creates a dictionary mapping profile names to their ProfileData."""
  profile_map = {}
  for profile in profiles:
    if profile["name"] in profile_map:
      raise ValueError(
          f"Duplicate profile name: '{profile['name']}'. Profile names must be"
          " unique."
      )
    profile_map[profile["name"]] = profile
  return profile_map


# A dictionary mapping metric profile identifiers to their ProfileData.
# The key for each metric profile (e.g., "Inference/Latency") should match
# the name field in its corresponding ProfileData.
METRIC_PROFILES: dict[str, ProfileData] = _make_profile_map([
    # go/keep-sorted start block=yes
    {
        "name": "Compilation/Conv5D",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/DepChain",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/Randint_Float32_fast",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/Randint_Float32_power_of_two",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/Randint_Float32_slow",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/Sort_Float32",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Compilation/Take_Float32",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Inference/Latency",
        "unit": "ms",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Inference/PreheatLatency",
        "unit": "ms",
        "stats": [Stat.VALUE],
    },
    {
        "name": "Training/PreheatOverhead",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Training/StepTime",
        "unit": "s",
        "stats": _DEFAULT_STATS,
    },
    {
        "name": "Training/ValidationLoss",
        "unit": "loss",
        "stats": [Stat.VALUE],
    },
    # go/keep-sorted end
])


def compute_stats(
    metrics: List[float], stats_to_compute: List[Stat]
) -> Dict[Stat, float]:
  """Calculates aggregate statistics (Mean, Median, percentiles) from a list of raw metric values.

  Args:
    metrics: A list of raw metric values.
    stats_to_compute: A list of `Stat` enums to compute.

  Returns:
    A dictionary mapping each `Stat` to its computed float value.
  """
  results: Dict[Stat, float] = {}
  for stat in stats_to_compute:
    if stat == Stat.MEAN:
      results[stat] = np.mean(metrics)
    elif stat == Stat.MEDIAN:
      results[stat] = np.median(metrics)
    elif stat == Stat.P90:
      results[stat] = np.percentile(metrics, 90)
    elif stat == Stat.P95:
      results[stat] = np.percentile(metrics, 95)
    elif stat == Stat.P99:
      results[stat] = np.percentile(metrics, 99)
    elif stat == Stat.STDDEV:
      results[stat] = np.std(metrics)
  return results


def construct_metric_tag(metric_profile: ProfileData, stat: Stat) -> str:
  """Constructs a unique, hierarchical tag for a given metric statistic.

  This tag (Example: "Inference/Latency/P90_ms") serves as the unique
  identifier for a specific stat's time-series.

  Each tag is reported separately and appears as its own graph in
  TensorBoard. This tag is the identifier used to store and query the
  results for that specific stat in MLCompass.

  Args:
    metric_profile: A ProfileData dict containing the metric's name and unit.
    stat: A Stat enum member.

  Returns:
    The formatted metric tag string.
  """
  return f"{metric_profile["name"]}/{stat.value}_{metric_profile["unit"]}"


def record_tensorboard_metrics(
    writer: SummaryWriter,
    metric_profile: ProfileData,
    metrics: list[float],
    record_step_metrics: bool = True,
):
  """Calculates and logs aggregate statistics for a given list of raw metric values.

  This function uses a TensorBoard SummaryWriter to record aggregates
  (Mean, Median, percentiles) based on a metric's profile. It can
  optionally log the raw "step metrics" as well.

  Args:
      writer: A TensorBoard SummaryWriter used to record metrics.
      metric_profile: A ProfileData dict containing the metric's name and unit.
      metrics: A list of raw, un-aggregated metric values (e.g., a list of
        latencies from multiple runs).
      record_step_metrics: If True, logs each raw data point from the metrics
        list individually as a data point for the "Value" stat. Each raw data
        point is logged  using its list index as the TensorBoard step value.
        This is useful for visualizing the raw data in TensorBoard, not just the
        final aggregate statistics (Mean, P90, etc.).
  """

  if record_step_metrics:
    metric_tag = construct_metric_tag(metric_profile, Stat.VALUE)
    for run_id, metric in enumerate(metrics):
      writer.add_scalar(metric_tag, metric, run_id)

  stats_to_compute = metric_profile.get("stats", [])
  if not stats_to_compute:
    return

  calculated_stats = compute_stats(metrics, stats_to_compute)
  for stat, value in calculated_stats.items():
    tag = construct_metric_tag(metric_profile, stat)
    writer.add_scalar(tag, value)


def assert_metric_within_threshold(
    metrics: List[float],
    metric_profile: ProfileData,
    stat_to_check: Stat,
    baseline: float,
    threshold_percent: float,
    comparison: Comparison,
):
  """Asserts that a metric is within a percentage threshold of a baseline value.

  Args:
    metrics: A list of raw, un-aggregated metric values (e.g., a list of
      latencies from multiple runs).
    metric_profile: A ProfileData dict containing the metric's name and unit.
    stat_to_check: The Stat enum specifying which metric to validate.
    baseline: The baseline value to compare against.
    threshold_percent: The allowed percentage deviation from the baseline.
    comparison: The Comparison enum for the assertion logic.
  """

  # Calculate stat value
  calculated_stats = compute_stats(metrics, [stat_to_check])
  stat_value = calculated_stats[stat_to_check]

  # Calculate threshold value
  threshold_value = baseline
  if comparison is Comparison.LESS:
    threshold_value = baseline * (1 + threshold_percent / 100.0)
  elif comparison is Comparison.GREATER:
    threshold_value = baseline * (1 - threshold_percent / 100.0)

  # Perform the assertion
  op_func = _COMPARISON_OPERATORS[comparison]
  assert op_func(stat_value, threshold_value), (
      f"  Benchmark regression failed for '{metric_profile['name']}' "
      f"({stat_to_check.name})!\n"
      f"  Comparison:   '{stat_value:.4f} {comparison.name} "
      f"{threshold_value:.4f}' failed.\n"
      f"  Baseline:     {baseline:.4f}\n"
      f"  Actual Value: {stat_value:.4f}\n"
      "  See go/torch-tpu-benchmarking#updating-metric-baseline "
      "for instructions on how to update the baseline."
  )
