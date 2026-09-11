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

"""Unit tests for the open-source list_ci_tests tool."""

import io
import os
import pathlib
import platform
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

try:
  from ci.tools import list_ci_tests  # pylint: disable=g-import-not-at-top
except ImportError:
  import list_ci_tests  # pylint: disable=g-import-not-at-top

_SAMPLE_BAZELRC = textwrap.dedent("""\
    # Sample CI test filter definitions
    test:ci_cpu --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu
    test:ci_cpu_presubmit --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-nopresubmit
    test:ci_tpu_v5 --test_tag_filters=requires-tpu,-notest,-nobuild,-fails-on-tpu-v5
    test:wheel_test_tpu_v5 --test_tag_filters=-notest,-nobuild,-nopresubmit,-wheel_test_excluded,-fails-on-tpu-v5,presubmit-v5
    """)

_SAMPLE_BAZEL_XML = textwrap.dedent("""\
    <?xml version="1.1" encoding="UTF-8" standalone="no"?>
    <query version="2">
        <rule class="py_test" location="/workspace/tests/BUILD:10:1" name="//tests:ops_test_tpu-v5lite">
            <string name="name" value="ops_test_tpu-v5lite"/>
            <list name="tags">
                <string value="presubmit-v5"/>
                <string value="requires-tpu"/>
                <string value="requires-tpu-v5lite"/>
                <string value="fails-on-tpu-v6"/>
            </list>
        </rule>
        <rule class="cc_test" location="/workspace/csrc/ops/BUILD:20:1" name="//csrc/ops:kernel_test">
            <string name="name" value="kernel_test"/>
            <list name="tags">
                <string value="cpu"/>
            </list>
        </rule>
    </query>
    """)


class ListCiTestsTest(unittest.TestCase):  # UNITTEST_OK=testing tools
  """Test suite for OSS list_ci_tests."""

  def setUp(self):
    super().setUp()
    self.repo_root = list_ci_tests.find_repo_root(pathlib.Path(__file__))

  def test_get_bazelrc_path(self):
    """Verifies that .bazelrc is discovered correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      bazelrc = root_path / ".bazelrc"
      bazelrc.write_text("test:ci --test_tag_filters=cpu\n", encoding="utf-8")

      discovered = list_ci_tests.get_bazelrc_path(root_path)
      self.assertEqual(discovered, bazelrc)

  def test_get_bazelrc_path_missing(self):
    """Verifies FileNotFoundError when .bazelrc is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tools
        list_ci_tests.get_bazelrc_path(root_path)

  def test_find_repo_root_env_override(self):
    """Verifies TORCH_TPU_REPO_DIR environment variable override."""
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch.dict(os.environ, {"TORCH_TPU_REPO_DIR": temp_dir}):
        resolved = list_ci_tests.find_repo_root()
        self.assertEqual(resolved, pathlib.Path(temp_dir).resolve())

  def test_parse_bazelrc_tag_filters(self):
    """Verifies parsing of --test_tag_filters from .bazelrc content."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".bazelrc", delete=False
    ) as f:
      f.write(_SAMPLE_BAZELRC)
      f.flush()
      f_path = pathlib.Path(f.name)

    try:
      configs = list_ci_tests.parse_bazelrc_tag_filters(f_path)
      self.assertIn("ci_cpu", configs)
      self.assertEqual(
          configs["ci_cpu"],
          ["-nobuild", "-notest", "-requires-gpu", "-requires-tpu"],
      )
      self.assertIn("ci_tpu_v5", configs)
      self.assertEqual(
          configs["ci_tpu_v5"],
          ["requires-tpu", "-notest", "-nobuild", "-fails-on-tpu-v5"],
      )
    finally:
      f_path.unlink()

  def test_parse_bazelrc_tag_filters_multiline_and_quotes(self):
    """Verifies that multiple filter lines for the same job accumulate and strip quotes."""
    content = textwrap.dedent("""\
        test:ci_multi --test_tag_filters="foo, bar"
        test:ci_multi --test_tag_filters='baz, bar'
        """)
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".bazelrc", delete=False
    ) as f:
      f.write(content)
      f.flush()
      f_path = pathlib.Path(f.name)

    try:
      configs = list_ci_tests.parse_bazelrc_tag_filters(f_path)
      self.assertEqual(configs["ci_multi"], ["foo", "bar", "baz"])
    finally:
      f_path.unlink()

  def test_matches_tag_filters(self):
    """Verifies tag filtering matching positive and negative filters."""
    tags = {"requires-tpu", "presubmit-v5"}
    self.assertTrue(list_ci_tests.matches_tag_filters(tags, ["presubmit-v5"]))
    self.assertTrue(
        list_ci_tests.matches_tag_filters(tags, ["-fails-on-tpu-v5"])
    )
    self.assertFalse(list_ci_tests.matches_tag_filters(tags, ["-requires-tpu"]))
    self.assertFalse(list_ci_tests.matches_tag_filters(tags, ["presubmit-v6"]))

  def test_parse_bazel_query_xml(self):
    """Verifies parsing of Bazel query XML output."""
    targets = list_ci_tests.parse_bazel_query_xml(_SAMPLE_BAZEL_XML)
    self.assertEqual(len(targets), 2)

    target_map = {t.name: t for t in targets}
    self.assertIn("ops_test_tpu-v5lite", target_map)
    self.assertIn("kernel_test", target_map)

    v5_target = target_map["ops_test_tpu-v5lite"]
    self.assertEqual(v5_target.label, "//tests:ops_test_tpu-v5lite")
    self.assertEqual(v5_target.package, "tests")
    self.assertEqual(v5_target.source_file, "/workspace/tests/BUILD")
    self.assertIn("presubmit-v5", v5_target.tags)
    self.assertIn("requires-tpu", v5_target.tags)

    kernel_target = target_map["kernel_test"]
    self.assertEqual(kernel_target.label, "//csrc/ops:kernel_test")
    self.assertEqual(kernel_target.package, "csrc/ops")
    self.assertIn("cpu", kernel_target.tags)

  def test_find_bazel_binary_explicit(self):
    """Verifies using an explicit Bazel binary."""
    fake_bin = pathlib.Path("/bin/true")
    with mock.patch.object(
        list_ci_tests, "_verify_bazel_binary", return_value=True
    ):
      resolved = list_ci_tests.find_bazel_binary(
          self.repo_root, explicit_bazel=fake_bin
      )
      self.assertEqual(resolved, fake_bin.resolve())

  def test_find_bazel_binary_env(self):
    """Verifies resolution via $BAZEL_BIN environment variable."""
    fake_bin = "/custom/bin/bazel"
    with mock.patch.dict(os.environ, {"BAZEL_BIN": fake_bin}):
      with mock.patch.object(
          list_ci_tests, "_verify_bazel_binary", return_value=True
      ):
        resolved = list_ci_tests.find_bazel_binary(self.repo_root)
        self.assertEqual(resolved, pathlib.Path(fake_bin).resolve())

  def test_find_bazel_binary_from_path(self):
    """Verifies resolution via `bazel` on PATH."""
    fake_bin = "/usr/bin/bazel"
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=fake_bin):
        with mock.patch.object(
            list_ci_tests, "_verify_bazel_binary", return_value=True
        ):
          resolved = list_ci_tests.find_bazel_binary(self.repo_root)
          self.assertEqual(resolved, pathlib.Path(fake_bin))

  def test_find_bazel_binary_cached_bazelisk(self):
    """Verifies fallback to a cached Bazelisk when PATH has no usable Bazel."""
    cached = list_ci_tests._get_bazelisk_cache_path()
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests,
            "_verify_bazel_binary",
            side_effect=lambda p, root=None: p == cached,
        ):
          resolved = list_ci_tests.find_bazel_binary(self.repo_root)
          self.assertEqual(resolved, cached)

  def test_find_bazel_binary_downloads_bazelisk(self):
    """Verifies Bazelisk is downloaded when no usable Bazel is available."""
    downloaded = pathlib.Path("/fake/cache/bazelisk-1.27.0")
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests,
            "_download_bazelisk",
            return_value=downloaded,
        ) as mock_download:
          # Every candidate fails until the freshly downloaded Bazelisk is
          # checked, so the download path is the one exercised here.
          with mock.patch.object(
              list_ci_tests,
              "_verify_bazel_binary",
              side_effect=lambda p, root=None: p == downloaded,
          ):
            resolved = list_ci_tests.find_bazel_binary(self.repo_root)
            self.assertEqual(resolved, downloaded)
            mock_download.assert_called_once()

  def test_find_bazel_binary_rejects_unverifiable_download(self):
    """Verifies a downloaded Bazelisk that does not run is not returned."""
    downloaded = pathlib.Path("/fake/cache/bazelisk-1.27.0")
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests, "_download_bazelisk", return_value=downloaded
        ):
          with mock.patch.object(
              list_ci_tests, "_verify_bazel_binary", return_value=False
          ):
            with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tools
              list_ci_tests.find_bazel_binary(self.repo_root)

  def test_get_bazelisk_download_url_is_platform_specific(self):
    """Verifies the Bazelisk URL encodes the OS and CPU architecture."""
    with mock.patch.object(sys, "platform", "linux"):
      with mock.patch.object(platform, "machine", return_value="x86_64"):
        self.assertEqual(
            list_ci_tests._get_bazelisk_download_url("1.27.0"),
            "https://github.com/bazelbuild/bazelisk/releases/download/"
            "v1.27.0/bazelisk-linux-amd64",
        )
    with mock.patch.object(sys, "platform", "darwin"):
      with mock.patch.object(platform, "machine", return_value="arm64"):
        self.assertIn(
            "bazelisk-darwin-arm64",
            list_ci_tests._get_bazelisk_download_url("1.27.0"),
        )

  def test_get_bazelisk_download_url_unsupported_arch(self):
    """Verifies an unsupported CPU architecture is reported clearly."""
    with mock.patch.object(sys, "platform", "linux"):
      with mock.patch.object(platform, "machine", return_value="mips"):
        with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=tools
          list_ci_tests._get_bazelisk_download_url("1.27.0")

  def test_discover_test_targets_executes_query(self):
    """Verifies discover_test_targets executes Bazel query and parses XML."""
    mock_proc = mock.Mock(returncode=0, stdout=_SAMPLE_BAZEL_XML, stderr="")
    with mock.patch("subprocess.run", return_value=mock_proc) as mock_run:
      with mock.patch.object(
          list_ci_tests,
          "find_bazel_binary",
          return_value=pathlib.Path("/bin/bazel"),
      ):
        targets = list_ci_tests.discover_test_targets(self.repo_root)
        self.assertEqual(len(targets), 2)
        mock_run.assert_called_once_with(
            [
                "/bin/bazel",
                "query",
                'kind(".*_test rule", //...)',
                "--output=xml",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

  def test_map_jobs_to_tests(self):
    """Verifies mapping of test targets to CI jobs with tag filters."""
    targets = [
        list_ci_tests.TestTarget(
            name="cpu_test",
            package="tests",
            tags=set(),
            source_file="BUILD",
        ),
        list_ci_tests.TestTarget(
            name="tpu_v5_test",
            package="tests",
            tags={"requires-tpu", "presubmit-v5"},
            source_file="BUILD",
        ),
    ]

    configs = {
        "ci_cpu": ["-requires-tpu"],
        "ci_tpu_v5_presubmit": ["presubmit-v5"],
    }

    job_map = list_ci_tests.map_jobs_to_tests(targets, configs)
    self.assertEqual(len(job_map["ci_cpu"]), 1)
    self.assertEqual(job_map["ci_cpu"][0].name, "cpu_test")
    self.assertEqual(len(job_map["ci_tpu_v5_presubmit"]), 1)
    self.assertEqual(job_map["ci_tpu_v5_presubmit"][0].name, "tpu_v5_test")

  def test_map_test_to_jobs(self):
    """Verifies target querying across multiple jobs."""
    targets = [
        list_ci_tests.TestTarget(
            name="ops_test_tpu-v5lite",
            package="tests",
            tags={"requires-tpu", "presubmit-v5"},
            source_file="tests/BUILD",
        ),
    ]

    configs = {
        "ci_cpu": ["-requires-tpu"],
        "ci_tpu_v5": ["requires-tpu"],
        "ci_tpu_v5_presubmit": ["presubmit-v5"],
    }

    target, jobs = list_ci_tests.map_test_to_jobs(
        "ops_test_tpu-v5lite", targets, configs
    )
    self.assertIsNotNone(target)
    self.assertCountEqual(jobs, ["ci_tpu_v5", "ci_tpu_v5_presubmit"])

  def test_format_text_output(self):
    """Verifies that format_text_output formats targets with indentation and no bullet."""
    targets = [
        list_ci_tests.TestTarget(
            name="cpu_test",
            package="tests",
            tags=set(),
            source_file="BUILD",
        ),
    ]
    job_to_targets = {"ci_cpu": targets}
    output = list_ci_tests.format_text_output(job_to_targets)
    self.assertIn("CI Job: ci_cpu (1 tests)", output)
    self.assertIn("  //tests:cpu_test", output)
    self.assertNotIn("  - //tests:cpu_test", output)

  def test_cli_execution_list_jobs(self):
    """Verifies the --list_jobs CLI flag."""
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    try:
      sys.stdout = captured_stdout
      ret = list_ci_tests.main(["--list_jobs"])
      self.assertEqual(ret, 0)
      output = captured_stdout.getvalue()
      self.assertIn("Available CI jobs", output)
    finally:
      sys.stdout = old_stdout

  def test_cli_execution_test_target_text(self):
    """Verifies querying a test target with text output format."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="my_target",
            package="tests",
            tags={"requires-tpu"},
            source_file="tests/BUILD",
        )
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          with mock.patch.object(
              list_ci_tests, "discover_test_targets", return_value=mock_targets
          ):
            ret = list_ci_tests.main(["--test_target=my_target"])
            self.assertEqual(ret, 0)
            output = captured_stdout.getvalue()
            self.assertIn("Test Target: //tests:my_target", output)
            self.assertIn("Runs in 1 CI job(s):", output)
            self.assertIn("ci_tpu_v5", output)
      finally:
        sys.stdout = old_stdout

  def test_cli_execution_jobs_filter_count(self):
    """Verifies filtering by --jobs with count format."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="my_target",
            package="tests",
            tags=set(),
            source_file="tests/BUILD",
        )
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          with mock.patch.object(
              list_ci_tests, "discover_test_targets", return_value=mock_targets
          ):
            ret = list_ci_tests.main([
                "--jobs=ci_cpu,ci_cpu_presubmit",
                "--format=count",
            ])
            self.assertEqual(ret, 0)
            output = captured_stdout.getvalue()
            self.assertIn("ci_cpu", output)
            self.assertIn("ci_cpu_presubmit", output)
      finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
  unittest.main()
