#!/usr/bin/env python3
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

"""Script to find the latest successful workflow run from GitHub API JSON."""

import json
import sys


def main():
  try:
    data = json.load(sys.stdin)
  except json.JSONDecodeError:
    print("ERROR: Failed to parse JSON from stdin.", file=sys.stderr)
    sys.exit(1)

  runs = data.get("workflow_runs", [])
  print(f"Found {len(runs)} runs", file=sys.stderr)

  if not runs:
    print("ERROR: No runs found for this commit.", file=sys.stderr)
    sys.exit(2)

  for r in runs:
    print(
        f"Run {r['run_number']} (Attempt {r['run_attempt']}):"
        f" status={r['status']}, conclusion={r['conclusion']}",
        file=sys.stderr,
    )
    if r["status"] == "completed" and r["conclusion"] == "success":
      # Print to stdout for bash to capture
      print(f"{r['run_number']} {r['run_attempt']}")
      sys.exit(0)

  print("No successful run found yet.", file=sys.stderr)
  sys.exit(1)


if __name__ == "__main__":
  main()
