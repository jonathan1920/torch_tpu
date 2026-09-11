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

"""Unit tests for TorchTPU environment and libtpu version handling."""

import enum

from absl import flags
from absl.testing import absltest
from torch_tpu import _loader
from torch_tpu._internal import env
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  LIBTPU_VERSION = "libtpu_version"
  LIBTPU_VERSION_NOT_LOADED = "libtpu_version_not_loaded"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


class EnvTest(seed_test_utils.RepeatableTest):
  """Worker test class running individual test cases in isolated subprocesses."""

  def setUp(self) -> None:
    super().setUp()
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")

  def test_libtpu_version(self) -> None:
    _loader.load()
    try:
      import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

      expected_version = libtpu.__version__
    except ImportError:
      expected_version = ""

    self.assertEqual(env.get_libtpu_version(), expected_version)

  def test_libtpu_version_not_loaded(self) -> None:
    self.assertIsNone(env.get_libtpu_version())


class ParentEnvTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.RepeatableTest,
):
  """Parent test running isolated worker subprocesses for each test case."""

  WORKER_TEST_METHOD_TEMPLATE = "EnvTest.test_{mode.value}"

  def setUp(self) -> None:
    super().setUp()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def test_libtpu_version(self) -> None:
    self.run_sub_test(TestMode.LIBTPU_VERSION)

  def test_libtpu_version_not_loaded(self) -> None:
    self.run_sub_test(TestMode.LIBTPU_VERSION_NOT_LOADED)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
