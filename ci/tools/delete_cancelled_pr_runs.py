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

"""Deletes cancelled GitHub Actions workflow runs for a closed PR once completed."""

import argparse
import json
import logging  # PYTHON_LOGGING_OK=CI utility script running in standard Python environment
import subprocess
import sys
import time
from typing import Sequence


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Delete cancelled workflow runs for a closed PR once completed."
      )
  )
  parser.add_argument(
      "--head-ref",
      required=True,
      help="The head reference/branch name of the closed pull request.",
  )
  parser.add_argument(
      "--current-run-id",
      type=int,
      required=True,
      help="The run ID of the current cleanup workflow to exclude.",
  )
  parser.add_argument(
      "--repository",
      required=True,
      help="The GitHub repository name in 'owner/repo' format.",
  )
  parser.add_argument(
      "--poll-attempts",
      type=int,
      default=8,
      help=(
          "Maximum polling attempts waiting for runs to reach completed status."
      ),
  )
  parser.add_argument(
      "--poll-interval-seconds",
      type=int,
      default=15,
      help="Seconds to wait between polling attempts.",
  )
  return parser.parse_args(args)


def get_target_run_ids(
    repository: str, head_ref: str, current_run_id: int
) -> Sequence[int]:
  """Queries workflow runs to clean up on the branch (in-flight or cancelled)."""
  cmd = [
      "gh",
      "run",
      "list",
      "--repo",
      repository,
      "--branch",
      head_ref,
      "--limit",
      "100",
      "--json",
      "databaseId,status,conclusion",
  ]
  result = subprocess.run(
      cmd,
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  runs = json.loads(result.stdout)

  target_ids = [
      run["databaseId"]
      for run in runs
      if run.get("databaseId") != current_run_id
      and (
          run.get("status") != "completed"
          or run.get("conclusion") == "cancelled"
      )
  ]
  return target_ids


def get_run_status(repository: str, run_id: int) -> str:
  """Fetches the current status of the given run_id."""
  cmd = [
      "gh",
      "run",
      "view",
      str(run_id),
      "--repo",
      repository,
      "--json",
      "status",
      "-q",
      ".status",
  ]
  result = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  if result.returncode == 0:
    return result.stdout.strip()
  logging.warning(
      "Failed to get status for run %s: %s", run_id, result.stderr.strip()
  )
  return "completed"


def delete_run(repository: str, run_id: int) -> bool:
  """Deletes the specified workflow run."""
  cmd = ["gh", "run", "delete", str(run_id), "--repo", repository]
  result = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  if result.returncode != 0:
    logging.warning(
        "Failed to delete run %s: %s", run_id, result.stderr.strip()
    )
    return False
  return True


def delete_cancelled_pr_runs(
    repository: str,
    head_ref: str,
    current_run_id: int,
    poll_attempts: int = 8,
    poll_interval_seconds: int = 15,
) -> None:
  """Deletes cancelled workflow runs on head_ref once they complete."""
  if not head_ref:
    logging.info("No head_ref specified; nothing to clean up.")
    return

  logging.info(
      "Searching for workflow runs to clean up on branch '%s'...", head_ref
  )
  run_ids = get_target_run_ids(repository, head_ref, current_run_id)

  if not run_ids:
    logging.info(
        "No workflow runs found to clean up for branch '%s'.", head_ref
    )
    return

  logging.info(
      "Found workflow runs to clean up for branch '%s': %s",
      head_ref,
      list(run_ids),
  )

  remaining_run_ids = set(run_ids)
  for attempt in range(poll_attempts):
    for run_id in list(remaining_run_ids):
      status = get_run_status(repository, run_id)
      if status == "completed":
        logging.info("Deleting completed run %s...", run_id)
        delete_run(repository, run_id)
        remaining_run_ids.remove(run_id)

    if not remaining_run_ids:
      break

    if attempt < poll_attempts - 1:
      logging.info(
          "Waiting for remaining runs %s to reach completed status (attempt"
          " %d/%d, sleeping %ds)...",
          sorted(remaining_run_ids),
          attempt + 1,
          poll_attempts,
          poll_interval_seconds,
      )
      time.sleep(poll_interval_seconds)

  if remaining_run_ids:
    logging.warning(
        "Did not delete the following jobs because they did not finish"
        " shutting down in time: %s",
        sorted(remaining_run_ids),
    )

  logging.info("Cleanup complete for branch '%s'.", head_ref)


def main(args: Sequence[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parsed = parse_args(args)
  try:
    delete_cancelled_pr_runs(
        repository=parsed.repository,
        head_ref=parsed.head_ref,
        current_run_id=parsed.current_run_id,
        poll_attempts=parsed.poll_attempts,
        poll_interval_seconds=parsed.poll_interval_seconds,
    )
  except Exception as e:
    logging.exception("Failed to clean up workflow runs: %s", e)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
