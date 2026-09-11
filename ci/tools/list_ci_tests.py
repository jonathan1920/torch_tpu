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

"""Tool to inspect and list test targets executed by OSS TorchTPU CI jobs.

Parses .bazelrc test tag filters to identify and list which CI job
configurations are defined and what filters they apply.

Example usage:
    $ ci/tools/list_ci_tests.py
    Available CI jobs from /path/to/.bazelrc:
      ci_v4_8: -requires-gpu-nvidia,-requires-tpu-v5litepod-4
      ci_v5e_4: -requires-gpu-nvidia,-requires-tpu-v4-8
"""

from collections.abc import Mapping, Sequence
import os
import pathlib
import re
import sys

# Matches a Bazel config line that sets --test_tag_filters, e.g.
#   test:ci_cpu --test_tag_filters=-nobuild,-notest
# The filter list may be bare, single-quoted, or double-quoted.
_TEST_TAG_FILTERS_RE = re.compile(
    r"""
    ^(?:common|test|build):        # Bazel command the config applies to.
    (?P<job>[a-zA-Z0-9_-]+)\s+     # CI job (Bazel config) name.
    .*?                            # Any flags preceding --test_tag_filters.
    --test_tag_filters=
    (?:
        "(?P<qfilters>[^"]*)"      # Double-quoted filter list.
      | '(?P<sqfilters>[^']*)'     # Single-quoted filter list.
      | (?P<filters>[^\s#]+)       # Bare filter list, up to space or comment.
    )
    """,
    re.VERBOSE,
)


def find_repo_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path:
  """Locates the OSS repository root containing .bazelrc.

  Resolution strategy:
    1. Checks the TORCH_TPU_REPO_DIR environment variable for an explicit path.
    2. Traverses upward from `start_path` (or this script's directory) towards
       the filesystem root until a directory containing `.bazelrc` is found.
    3. Falls back to the current working directory if no parent contains
       `.bazelrc`.

  Args:
    start_path: Optional path to start searching upwards from. Defaults to the
      location of this script or the current working directory.

  Returns:
    Path to the detected repository root directory.
  """
  env_override = os.environ.get("TORCH_TPU_REPO_DIR")
  if env_override:
    p = pathlib.Path(env_override).resolve()
    if p.is_dir():
      return p

  curr = (start_path or pathlib.Path(__file__)).resolve()
  if curr.is_file():
    curr = curr.parent

  while curr != curr.parent:
    if (curr / ".bazelrc").is_file():
      return curr
    curr = curr.parent

  return pathlib.Path.cwd().resolve()


def get_bazelrc_path(repo_root: pathlib.Path) -> pathlib.Path:
  """Returns the path to the .bazelrc file defining CI test filters.

  Args:
    repo_root: Repository root expected to contain .bazelrc.

  Returns:
    Path to the .bazelrc file.

  Raises:
    FileNotFoundError: If repo_root does not contain a .bazelrc file.
  """
  bazelrc_path = repo_root / ".bazelrc"
  if bazelrc_path.is_file():
    return bazelrc_path
  raise FileNotFoundError(
      f"Could not find .bazelrc defining CI test filters under {repo_root}."
  )


def parse_bazelrc_tag_filters(
    bazelrc_path: pathlib.Path,
) -> Mapping[str, Sequence[str]]:
  """Parses --test_tag_filters configurations for CI jobs from .bazelrc.

  Looks for lines of the form:
    test:<job_name> --test_tag_filters=<filters>

  Args:
    bazelrc_path: Path to the .bazelrc file.

  Returns:
    A mapping from CI job name (e.g. 'ci_cpu_presubmit') to its filter list
    (e.g. ['-nobuild', '-notest', '-requires-gpu', '-requires-tpu']).
  """
  ci_configs: dict[str, list[str]] = {}

  with open(bazelrc_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      m = _TEST_TAG_FILTERS_RE.match(line)
      if m:
        job = m.group("job")
        filters_str = (
            m.group("qfilters")
            or m.group("sqfilters")
            or m.group("filters")
            or ""
        )
        filters = [
            f.strip("\t '\"")
            for f in filters_str.split(",")
            if f.strip("\t '\"")
        ]
        existing = ci_configs.setdefault(job, [])
        for f in filters:
          if f not in existing:
            existing.append(f)

  return ci_configs


def main() -> int:
  repo_root = find_repo_root()
  try:
    bazelrc_path = get_bazelrc_path(repo_root)
  except FileNotFoundError as e:
    sys.stderr.write(f"Error: {e}\n")
    return 1

  ci_configs = parse_bazelrc_tag_filters(bazelrc_path)
  if not ci_configs:
    sys.stderr.write(
        f"Error: No CI test_tag_filters found in {bazelrc_path}.\n"
    )
    return 1

  print(f"Available CI jobs from {bazelrc_path}:")
  for job, filters in sorted(ci_configs.items()):
    print(f"  {job}: {','.join(filters)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
