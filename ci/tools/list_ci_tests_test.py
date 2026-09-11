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

"""Unit tests for the open-source list_ci_tests tool."""

import io
import os
import pathlib
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

  def test_real_ci_filters_integrity(self):
    """Verifies that the real repository .bazelrc is valid and complete."""
    bazelrc_path = list_ci_tests.get_bazelrc_path(self.repo_root)
    configs = list_ci_tests.parse_bazelrc_tag_filters(bazelrc_path)

    self.assertIn("ci_cpu", configs)
    self.assertIn("ci_cpu_presubmit", configs)
    self.assertIn("ci_tpu_v5_presubmit", configs)
    self.assertIn("ci_tpu_v7_presubmit", configs)

  def test_cli_execution_in_oss_repo(self):
    """Verifies default CLI execution in an OSS repository layout."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      bazelrc = root_path / ".bazelrc"
      bazelrc.write_text(_SAMPLE_BAZELRC, encoding="utf-8")

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          ret = list_ci_tests.main()
          self.assertEqual(ret, 0)
          output = captured_stdout.getvalue()
          self.assertIn("ci_cpu", output)
          self.assertIn("ci_tpu_v5", output)
      finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
  unittest.main()
