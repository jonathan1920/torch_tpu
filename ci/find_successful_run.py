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

import argparse
import json
import sys


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "target_sha",
      help="Enforce matching this head SHA.",
  )
  args = parser.parse_args()

  try:
    data = json.load(sys.stdin)
  except json.JSONDecodeError:
    print("ERROR: Failed to parse JSON from stdin.", file=sys.stderr)
    sys.exit(1)

  runs = data.get("workflow_runs", [])
  print(f"Found {len(runs)} runs", file=sys.stderr)

  if not runs:
    print("ERROR: No runs found for this branch/workflow.", file=sys.stderr)
    sys.exit(2)

  target_sha = args.target_sha
  print(f"Filtering runs for head_sha: {target_sha}", file=sys.stderr)

  matched_run_exists = False
  active_run_exists = False
  failed_runs = []

  for r in runs:
    run_sha = r.get("head_sha")

    if run_sha != target_sha:
      continue

    matched_run_exists = True
    print(
        f"Run {r['run_number']} (Attempt {r['run_attempt']}):"
        f" sha={run_sha[:8] if run_sha else 'None'}, status={r['status']},"
        f" conclusion={r['conclusion']}",
        file=sys.stderr,
    )

    if r["status"] != "completed":
      active_run_exists = True
      continue

    if r["conclusion"] == "success":
      # Print to stdout for bash to capture.
      print(f"{r['run_number']} {r['run_attempt']}")
      sys.exit(0)

    failed_runs.append(r)

  if not matched_run_exists:
    print(f"No runs found yet matching sha {target_sha}.", file=sys.stderr)
    sys.exit(4)

  if active_run_exists:
    print(
        f"Active runs exist for sha {target_sha}. Keep waiting.",
        file=sys.stderr,
    )
    sys.exit(1)

  # If we got here, all matched runs completed but none succeeded.
  for r in failed_runs:
    print(
        f"ERROR: Build run {r['run_number']} completed with conclusion:"
        f" {r['conclusion']}",
        file=sys.stderr,
    )
  sys.exit(3)


if __name__ == "__main__":
  main()
