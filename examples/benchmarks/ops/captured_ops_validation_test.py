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

"""Presubmit validation test suite for captured ATen operator JSON files.

Verifies that all JSON capture files in captured_ops/ directory:
1. Conform to schema structure (root dict, positive counts).
2. Contain valid operator names and namespaces.
3. Can be cleanly deserialized by op_input_loader.deserialize_args without
errors.
"""

import functools
import glob
import json
import os
from typing import List
from absl.testing import absltest
from absl.testing import parameterized
from examples.benchmarks.ops.op_input_loader import deserialize_args


@functools.lru_cache(maxsize=1)
def _get_capture_files() -> List[str]:
  """Locates all captured JSON files in the captured_ops directory."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  captured_ops_dir = os.path.join(script_dir, "captured_ops")

  files = sorted(glob.glob(os.path.join(captured_ops_dir, "*.json")))
  if not files:
    # Also check runfiles if running as blaze test
    candidate_dir = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "google3",
        "third_party",
        "py",
        "torch_tpu",
        "examples",
        "benchmarks",
        "ops",
        "captured_ops",
    )
    if os.path.exists(candidate_dir):
      files = sorted(glob.glob(os.path.join(candidate_dir, "*.json")))
  return files


class CapturedOpsValidationTest(parameterized.TestCase):

  def test_capture_files_exist(self):
    files = _get_capture_files()
    self.assertNotEmpty(
        files, "No captured operator JSON files found in captured_ops/"
    )

  @parameterized.named_parameters(
      [(os.path.basename(f).replace(".", "_"), f) for f in _get_capture_files()]
  )
  def test_validate_captured_file(self, file_path: str):
    """Validates schema, op names, positive counts, and AST deserialization in 1 pass."""
    self.assertTrue(os.path.exists(file_path), f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    self.assertIsInstance(data, dict, f"Root must be a JSON dict: {file_path}")
    self.assertNotEmpty(data, f"JSON file must not be empty: {file_path}")

    total_op_count = 0
    for op_name, entries in data.items():
      if op_name == "CLUSTER":
        self.assertIsInstance(
            entries, list, f"CLUSTER must be an array in {file_path}"
        )
        for cluster_idx, cluster in enumerate(entries):
          self.assertIsInstance(cluster, dict)
          count = cluster.get("count", 0)
          self.assertGreater(
              count,
              0,
              f"Cluster {cluster_idx} count must be > 0 in {file_path}",
          )
          total_op_count += count
          for op_entry in cluster.get("ops", []):
            inputs_str = op_entry.get("inputs", "")
            try:
              args, kwargs = deserialize_args(inputs_str, device="meta")
              self.assertIsInstance(args, list)
              self.assertIsInstance(kwargs, dict)
            except Exception as e:
              self.fail(
                  f"Failed to deserialize CLUSTER op {op_entry.get('op')} at"
                  f" cluster {cluster_idx} in {file_path}:"
                  f" inputs='{inputs_str}', error={e}"
              )
        continue

      self.assertTrue(
          op_name.startswith("aten.") or "." in op_name or "::" in op_name,
          f"Invalid op name format '{op_name}' in {file_path}",
      )
      self.assertIsInstance(
          entries, list, f"Op '{op_name}' entries must be a list in {file_path}"
      )
      self.assertNotEmpty(
          entries, f"Op '{op_name}' entries cannot be empty in {file_path}"
      )

      for idx, entry in enumerate(entries):
        self.assertIsInstance(
            entry,
            dict,
            f"Entry {idx} for '{op_name}' must be dict in {file_path}",
        )
        count = entry.get("count", 0)
        self.assertIsInstance(
            count,
            int,
            f"Count for '{op_name}'[{idx}] must be int in {file_path}",
        )
        self.assertGreater(
            count,
            0,
            f"Count must be > 0 for '{op_name}'[{idx}] in {file_path}",
        )
        total_op_count += count

        inputs_str = entry.get("inputs", "")
        self.assertIsInstance(
            inputs_str,
            str,
            f"inputs for '{op_name}'[{idx}] must be str in {file_path}",
        )
        try:
          args, kwargs = deserialize_args(inputs_str, device="meta")
          self.assertIsInstance(args, list)
          self.assertIsInstance(kwargs, dict)
        except Exception as e:
          self.fail(
              f"Failed to deserialize '{op_name}'[{idx}] in {file_path}:"
              f" inputs='{inputs_str}', error={e}"
          )

    self.assertGreater(
        total_op_count,
        0,
        f"Total operator count must be > 0 in {file_path}",
    )


if __name__ == "__main__":
  absltest.main()
