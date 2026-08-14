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

"""Common command-line flags for benchmark tests."""

from absl import flags

SKIP_BEHAVIOR = flags.DEFINE_enum(
    "skip_behavior",
    "skip",
    ["skip", "assert_raise", "ignore", "run_skipped"],
    "Behavior for run modes that are marked as skipped in the config."
    " Options:\n  skip: Skips the test if the benchmark is marked as skipped.\n"
    "  assert_raise: Runs the test and expects it to raise an exception if"
    " skipped.\n  ignore: Runs the test normally even if it is marked as"
    " skipped.\n  run_skipped: Only runs skipped tests (skips those not marked"
    " as skipped).",
)

ENABLE_XPROF = flags.DEFINE_bool(
    "enable_xprof",
    False,
    "Whether to enable xprof profiling.",
    allow_override=True,
)

# Flags passed by launcher for metadata and tracking.
MLCOMPASS_TRACKING_ID = flags.DEFINE_string(
    "mlcompass_tracking_id", "", "MLCompass tracking UUID.", allow_override=True
)
MLCOMPASS_EXECUTION_MODE = flags.DEFINE_string(
    "mlcompass_execution_mode",
    "oneshot",
    "MLCompass execution mode.",
    allow_override=True,
)
BASE_CL = flags.DEFINE_string(
    "base_cl", "", "Base CL used for the benchmark run.", allow_override=True
)
PENDING_CL = flags.DEFINE_string(
    "pending_cl",
    "",
    "Pending CL used for the benchmark run.",
    allow_override=True,
)
BENCHMARK_GROUP = flags.DEFINE_string(
    "benchmark_group", "experiment", "Benchmark group.", allow_override=True
)
CONFIG_PATH = flags.DEFINE_string(
    "config_path", "", "MLCompass config file path.", allow_override=True
)
CELL = flags.DEFINE_string(
    "cell", "", "Cell to run the job in.", allow_override=True
)
PRIORITY = flags.DEFINE_integer(
    "priority", 200, "Priority to launch the job at.", allow_override=True
)
