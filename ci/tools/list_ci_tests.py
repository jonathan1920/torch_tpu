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

Parses .bazelrc test tag filters and invokes Bazel query to inspect which test
targets run in each CI job configuration.

Example usage:
    # List test targets grouped by CI job:
    $ ci/tools/list_ci_tests.py

    # List test targets in count format:
    $ ci/tools/list_ci_tests.py --format=count

    # Check which CI jobs run a specific test target:
    $ ci/tools/list_ci_tests.py --test_target=ops_test_tpu-v5lite
"""

import argparse
from collections.abc import Mapping, Sequence
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree

# Bazelisk release used when bootstrapping. Kept in sync with the version the
# README tells contributors to install. Unlike a pinned Bazel version this is
# not a correctness constraint: whichever Bazelisk we end up with reads the
# repository's `.bazelversion`, so this only affects the launcher itself.
_BAZELISK_VERSION = "1.27.0"


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


def matches_tag_filters(
    target_tags: set[str],
    filter_list: Sequence[str],
) -> bool:
  """Evaluates whether a target's tags satisfy Bazel --test_tag_filters rules.

  Bazel filter semantics reference:
    https://bazel.build/reference/command-line-reference#flag--test_tag_filters

  Semantics summary:
    - Filters can be negative (prefixed with '-', e.g. '-requires-gpu') or
      positive (e.g. 'presubmit-v5').
    - Negative filters: If any excluded tag in `filter_list` is present in
      `target_tags`, the target is excluded immediately (evaluates to False).
    - Positive filters: If any positive tags are specified in `filter_list`,
      the target must possess at least one of them to be included. If no
      positive tags match, the target is excluded (evaluates to False).
    - If no positive filters are specified, the target is included as long as
      it does not match any negative filters.
    - Order does not matter: negative exclusions take precedence regardless of
      position in the filter list.

  Args:
    target_tags: A set of string tag identifiers defined on the test target
      (e.g., {'requires-tpu', 'presubmit-v5'}).
    filter_list: A sequence of tag filter expressions, positive or negative
      (e.g., ['requires-tpu', '-flaky', '-nopresubmit']).

  Returns:
    True if the target satisfies all negative and positive filters; False
    otherwise.
  """
  positive_filters = []
  for f in filter_list:
    if f.startswith("-"):
      excluded_tag = f[1:]
      if excluded_tag in target_tags:
        return False
    else:
      positive_filters.append(f)

  if positive_filters:
    if not any(pos_tag in target_tags for pos_tag in positive_filters):
      return False

  return True


class TestTarget:
  """Represents a discovered test target with its metadata and tags."""

  def __init__(
      self,
      name: str,
      package: str,
      tags: set[str],
      source_file: str,
  ):
    """Initializes a TestTarget instance.

    Args:
      name: The short rule name of the test target (e.g.,
        'ops_test_tpu-v5lite').
      package: The package path containing the rule (e.g., 'tests' or
        'csrc/ops'), without leading or trailing slashes.
      tags: A set of string tags declared on the test rule (e.g.,
        {'requires-tpu', 'presubmit-v5'}).
      source_file: The path or location of the BUILD file where the target was
        declared (e.g., '/workspace/tests/BUILD').
    """
    self.name = name
    self.package = package
    self.tags = tags
    self.source_file = source_file

  @property
  def label(self) -> str:
    pkg = self.package.strip("/")
    if pkg:
      return f"//{pkg}:{self.name}"
    return f"//:{self.name}"

  def __repr__(self) -> str:
    return f"<TestTarget {self.label} tags={sorted(self.tags)}>"


def _get_bazelisk_download_url(version: str = _BAZELISK_VERSION) -> str:
  """Returns the GitHub release download URL for Bazelisk on this machine.

  Args:
    version: Bazelisk release version, without the leading 'v'.

  Returns:
    The download URL for the current OS and CPU architecture.

  Raises:
    RuntimeError: If the OS or CPU architecture is not published by Bazelisk.
  """
  os_names = {"linux": "linux", "darwin": "darwin", "win32": "windows"}
  arch_names = {
      "x86_64": "amd64",
      "amd64": "amd64",
      "aarch64": "arm64",
      "arm64": "arm64",
  }

  os_name = os_names.get(sys.platform)
  if os_name is None:
    raise RuntimeError(f"Unsupported operating system: {sys.platform}.")
  arch_name = arch_names.get(platform.machine().lower())
  if arch_name is None:
    raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}.")

  suffix = ".exe" if os_name == "windows" else ""
  return (
      "https://github.com/bazelbuild/bazelisk/releases/download/"
      f"v{version}/bazelisk-{os_name}-{arch_name}{suffix}"
  )


def _get_bazelisk_cache_path(
    version: str = _BAZELISK_VERSION,
) -> pathlib.Path:
  """Returns the path of the user-space cached Bazelisk binary."""
  cache_dir = pathlib.Path.home() / ".cache" / "torch_tpu" / "bin"
  return cache_dir / f"bazelisk-{version}"


def _verify_bazel_binary(
    bazel_bin: pathlib.Path,
    repo_root: pathlib.Path | None = None,
) -> bool:
  """Verifies that a Bazel binary is executable and runs successfully.

  Args:
    bazel_bin: Path to the candidate Bazel binary executable.
    repo_root: Optional repository root directory. If provided, `bazel
      --version` is executed with `cwd=repo_root`, ensuring any wrapper tools
      (e.g., Bazelisk) respect the repository's `.bazelversion`. If None, `cwd`
      is not set (the current working directory is used).

  Returns:
    True if the binary exists, is executable, and returns exit code 0 when
    invoked with `--version`; False otherwise.
  """
  if not bazel_bin.is_file() or not os.access(bazel_bin, os.X_OK):
    return False
  try:
    proc = subprocess.run(
        [str(bazel_bin), "--version"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0
  except (OSError, subprocess.SubprocessError):
    return False


def _download_bazelisk(
    target_path: pathlib.Path,
    version: str = _BAZELISK_VERSION,
) -> pathlib.Path:
  """Downloads Bazelisk from GitHub Releases into target_path.

  Args:
    target_path: Destination path for the downloaded executable.
    version: Bazelisk release version, without the leading 'v'.

  Returns:
    target_path.

  Raises:
    RuntimeError: If the platform is unsupported or the download fails.
  """
  download_url = _get_bazelisk_download_url(version)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  sys.stderr.write(f"Downloading Bazelisk {version} into {target_path}...\n")
  try:
    with urllib.request.urlopen(download_url, timeout=60) as resp:
      if resp.status != 200:
        raise RuntimeError(
            f"Failed to download Bazelisk: HTTP {resp.status} {resp.reason}"
        )
      # Download to a temporary path and rename, so that an interrupted
      # download cannot leave a truncated binary in the cache.
      temp_path = target_path.with_suffix(".tmp")
      with open(temp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
      temp_path.chmod(0o755)
      temp_path.replace(target_path)
  except (urllib.error.URLError, OSError, RuntimeError) as e:
    raise RuntimeError(
        f"Failed to download Bazelisk {version} from {download_url}: {e}"
    ) from e
  return target_path


def find_bazel_binary(
    repo_root: pathlib.Path,
    explicit_bazel: pathlib.Path | None = None,
) -> pathlib.Path:
  """Locates a Bazel launcher that can run the version pinned by the repo.

  Bootstrapping installs Bazelisk rather than a specific Bazel release, so
  this tool never has to parse `.bazelversion` or reproduce Bazel's per-OS and
  per-CPU release naming. Note that a plain `bazel` on PATH is not necessarily
  sufficient: the apt-installed wrapper dispatches on `.bazelversion` but
  cannot fetch a missing version, which is why every candidate is verified by
  running `bazel --version` from `repo_root`.

  Checks in order:
    1. Explicit path passed as argument.
    2. Environment variable $BAZEL_BIN.
    3. `bazel` on PATH.
    4. Cached Bazelisk in `~/.cache/torch_tpu/bin/`.
    5. Bazelisk downloaded from its official GitHub releases.

  Args:
    repo_root: Path to repository root. Candidates are verified with this as the
      working directory so that `.bazelversion` is honoured.
    explicit_bazel: Optional explicit path provided by the caller.

  Returns:
    Path to the verified Bazel executable.

  Raises:
    FileNotFoundError: If no working Bazel binary can be located or
      bootstrapped.
  """
  if explicit_bazel:
    candidate = explicit_bazel.resolve()
    if _verify_bazel_binary(candidate, repo_root):
      return candidate
    raise FileNotFoundError(
        f"Specified Bazel binary at {candidate} is not executable or failed"
        " verification."
    )

  env_bin = os.environ.get("BAZEL_BIN")
  if env_bin:
    candidate = pathlib.Path(env_bin).resolve()
    if _verify_bazel_binary(candidate, repo_root):
      return candidate

  which_path = shutil.which("bazel")
  if which_path:
    candidate = pathlib.Path(which_path)
    if _verify_bazel_binary(candidate, repo_root):
      return candidate

  cached_path = _get_bazelisk_cache_path()
  if _verify_bazel_binary(cached_path, repo_root):
    return cached_path

  try:
    downloaded = _download_bazelisk(cached_path)
  except RuntimeError as e:
    raise FileNotFoundError(
        "Bazel is required to discover test targets, but no working `bazel`"
        f" was found and Bazelisk could not be downloaded:\n{e}\nInstall"
        " Bazelisk as described in the repository README, or point $BAZEL_BIN"
        " at a Bazel binary."
    ) from e

  if not _verify_bazel_binary(downloaded, repo_root):
    raise FileNotFoundError(
        f"Downloaded Bazelisk at {downloaded} failed verification. Install"
        " Bazelisk as described in the repository README, or point $BAZEL_BIN"
        " at a Bazel binary."
    )
  return downloaded


def parse_bazel_query_xml(xml_content: str) -> list[TestTarget]:
  """Parses Bazel query XML output into TestTarget objects.

  Bazel query XML output format (relevant subsets):
    The root element is `<query version="2">` containing `<rule>` elements for
    matching build targets.
    - Each `<rule>` has attributes:
        - `class`: The rule class type (e.g., `py_test`, `cc_test`).
        - `name`: The full target label (e.g., `//tests:ops_test_tpu-v5lite`).
        - `location`: The declaration source file and line (e.g.,
          `/workspace/tests/BUILD:10:1`).
    - Inside `<rule>`, tag metadata is stored in a `<list name="tags">` element
      containing zero or more `<string value="...">` elements.

  Example XML:
    ```xml
    <query version="2">
      <rule class="py_test" location="/workspace/tests/BUILD:10:1"
            name="//tests:my_test">
        <string name="name" value="my_test"/>
        <list name="tags">
          <string value="requires-tpu"/>
          <string value="presubmit-v5"/>
        </list>
      </rule>
    </query>
    ```

  Args:
    xml_content: String containing the raw XML output from `bazel query
      --output=xml`.

  Returns:
    A list of TestTarget objects extracted from the XML, sorted by target label.

  Raises:
    ValueError: If `xml_content` cannot be parsed as valid XML.
  """
  targets: list[TestTarget] = []
  try:
    root = xml.etree.ElementTree.fromstring(xml_content)
  except xml.etree.ElementTree.ParseError as e:
    raise ValueError(f"Failed to parse Bazel query XML output: {e}") from e

  for rule in root.findall("rule"):
    label = rule.get("name")
    if not label:
      continue

    if ":" in label:
      pkg_part, name = label.split(":", 1)
      package = pkg_part.lstrip("/")
    else:
      package = ""
      name = label.lstrip("/")

    tags: set[str] = set()
    for list_elem in rule.findall("list"):
      if list_elem.get("name") == "tags":
        for str_elem in list_elem.findall("string"):
          val = str_elem.get("value")
          if val:
            tags.add(val)

    location = rule.get("location", "")
    source_file = location.split(":")[0] if location else ""

    targets.append(
        TestTarget(
            name=name,
            package=package,
            tags=tags,
            source_file=source_file,
        )
    )

  targets.sort(key=lambda t: t.label)
  return targets


def discover_test_targets(
    repo_root: pathlib.Path,
    bazel_bin: pathlib.Path | None = None,
) -> list[TestTarget]:
  """Queries all test targets and tags defined in repo_root using Bazel query.

  Args:
    repo_root: Root directory of the repository where the Bazel query will be
      executed.
    bazel_bin: Optional explicit path to the Bazel binary executable. If None,
      the binary is located via `find_bazel_binary(repo_root)`.

  Returns:
    A list of TestTarget objects discovered in the repository.

  Raises:
    RuntimeError: If Bazel query fails or exits with a non-zero returncode.
    FileNotFoundError: If no working Bazel binary can be located.
  """
  resolved_bazel = find_bazel_binary(repo_root, explicit_bazel=bazel_bin)
  cmd = [
      str(resolved_bazel),
      "query",
      'kind(".*_test rule", //...)',
      "--output=xml",
  ]
  proc = subprocess.run(
      cmd,
      cwd=repo_root,
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    err_msg = proc.stderr.strip() or proc.stdout.strip()
    raise RuntimeError(
        f"Bazel query failed (exit code {proc.returncode}):\n{err_msg}"
    )

  return parse_bazel_query_xml(proc.stdout)


def map_jobs_to_tests(
    targets: Sequence[TestTarget],
    ci_configs: Mapping[str, Sequence[str]],
    selected_jobs: Sequence[str] | None = None,
) -> dict[str, list[TestTarget]]:
  """Maps each CI job to the test targets it will execute.

  Args:
    targets: Sequence of candidate TestTarget objects to evaluate.
    ci_configs: Mapping from CI job name (e.g. 'ci_cpu_presubmit') to its
      sequence of tag filters (e.g. ['-nobuild', '-notest', '-requires-tpu']).
    selected_jobs: Optional sequence of CI job names to restrict evaluation to.
      If None or empty, all jobs in `ci_configs` are evaluated in sorted order.

  Returns:
    A dictionary mapping each evaluated CI job name to the list of TestTarget
    objects matching that job's filter configuration.
  """
  job_to_targets: dict[str, list[TestTarget]] = {}
  jobs_to_eval = selected_jobs if selected_jobs else sorted(ci_configs.keys())

  for job in jobs_to_eval:
    if job not in ci_configs:
      continue
    filters = ci_configs[job]
    matched = [t for t in targets if matches_tag_filters(t.tags, filters)]
    job_to_targets[job] = matched

  return job_to_targets


def map_test_to_jobs(
    target_query: str,
    targets: Sequence[TestTarget],
    ci_configs: Mapping[str, Sequence[str]],
    selected_jobs: Sequence[str] | None = None,
) -> tuple[TestTarget | None, list[str]]:
  """Queries which CI jobs execute a specific test target.

  Args:
    target_query: The test target label or name to search for (e.g.,
      'ops_test_tpu-v5lite', '//tests:ops_test_tpu-v5lite', or
      ':ops_test_tpu-v5lite').
    targets: Sequence of available TestTarget objects to match against.
    ci_configs: Mapping from CI job name to its sequence of tag filters.
    selected_jobs: Optional sequence of CI job names to restrict evaluation to.
      If None or empty, all jobs in `ci_configs` are evaluated in sorted order.

  Returns:
    A tuple of:
      - The matched TestTarget object, or None if no match was found.
      - A list of CI job names that will execute this test target based on tag
        filters.
  """
  clean_query = target_query.strip()
  matched_target: TestTarget | None = None

  for t in targets:
    if clean_query in (t.name, t.label, f":{t.name}"):
      matched_target = t
      break

  if not matched_target:
    return None, []

  jobs_to_eval = selected_jobs if selected_jobs else sorted(ci_configs.keys())
  matching_jobs: list[str] = []
  for job in jobs_to_eval:
    if job not in ci_configs:
      continue
    filters = ci_configs[job]
    if matches_tag_filters(matched_target.tags, filters):
      matching_jobs.append(job)

  return matched_target, matching_jobs


def format_text_output(
    job_to_targets: Mapping[str, Sequence[TestTarget]],
) -> str:
  """Formats test targets grouped by CI job in human-readable text."""
  lines = []
  for job, targets in sorted(job_to_targets.items()):
    lines.append(f"CI Job: {job} ({len(targets)} tests)")
    if targets:
      for t in targets:
        lines.append(f"  {t.label}")
    else:
      lines.append("  (No tests match this job's filter)")
    lines.append("")
  return "\n".join(lines).rstrip()


def format_count_output(
    job_to_targets: Mapping[str, Sequence[TestTarget]],
) -> str:
  """Formats test count summary grouped by CI job."""
  lines = ["CI Job Test Counts:", "--------------------"]
  for job, targets in sorted(job_to_targets.items()):
    lines.append(f"{job:30} : {len(targets)} tests")
  return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
  """Builds CLI argument parser for list_ci_tests."""
  parser = argparse.ArgumentParser(
      description="List test targets included in TorchTPU CI jobs.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--jobs",
      type=str,
      default=None,
      help=(
          "Comma-separated list of CI job names to include (e.g."
          " --jobs=ci_cpu_presubmit,ci_tpu_v5_presubmit). Defaults to all jobs."
      ),
  )
  parser.add_argument(
      "--test_target",
      type=str,
      default=None,
      help=(
          "Specify a test target name or label to list all CI jobs that execute"
          " it (e.g. --test_target=ops_test_tpu-v5lite or"
          " --test_target=//tests:ops_test_tpu-v5lite)."
      ),
  )
  parser.add_argument(
      "--format",
      choices=["text", "count"],
      default="text",
      help="Output format: 'text' (default) or 'count'.",
  )
  parser.add_argument(
      "--list_jobs",
      action="store_true",
      help="List all defined CI jobs and their tag filters, then exit.",
  )
  parser.add_argument(
      "--bazel_bin",
      type=str,
      default=None,
      help="Explicit path to the Bazel binary executable to use.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = _build_arg_parser()
  args = parser.parse_args(argv)

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

  if args.list_jobs:
    print(f"Available CI jobs from {bazelrc_path}:")
    for job, filters in sorted(ci_configs.items()):
      print(f"  {job}: {','.join(filters)}")
    return 0

  selected_jobs: list[str] | None = None
  if args.jobs:
    selected_jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    invalid_jobs = [j for j in selected_jobs if j not in ci_configs]
    if invalid_jobs:
      sys.stderr.write(
          f"Warning: Unknown CI job(s): {', '.join(invalid_jobs)}.\n"
          f"Available jobs: {', '.join(sorted(ci_configs.keys()))}\n\n"
      )

  bazel_path = pathlib.Path(args.bazel_bin) if args.bazel_bin else None
  try:
    targets = discover_test_targets(repo_root, bazel_bin=bazel_path)
  except Exception as e:  # pylint: disable=broad-exception-caught
    sys.stderr.write(f"Error discovering test targets: {e}\n")
    return 1

  if args.test_target:
    matched_target, matching_jobs = map_test_to_jobs(
        args.test_target, targets, ci_configs, selected_jobs
    )
    if not matched_target:
      sys.stderr.write(
          f"Error: Test target '{args.test_target}' was not found in"
          f" {repo_root}.\n"
      )
      return 1

    print(f"Test Target: {matched_target.label}")
    print(f"Tags: {', '.join(sorted(matched_target.tags))}")
    print(f"Source: {matched_target.source_file}")
    print(f"\nRuns in {len(matching_jobs)} CI job(s):")
    if matching_jobs:
      for j in sorted(matching_jobs):
        print(f"  - {j}")
    else:
      print("  (None: excluded from all queried CI jobs)")
    return 0

  job_to_targets = map_jobs_to_tests(targets, ci_configs, selected_jobs)

  if args.format == "count":
    print(format_count_output(job_to_targets))
  else:
    print(format_text_output(job_to_targets))

  return 0


if __name__ == "__main__":
  sys.exit(main())
