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

"""Tests for `error_testing.py` functions."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from tests import error_testing as et
from tests import seed_test_utils


class ErrorTestingTest(seed_test_utils.RepeatableTest):

  def test_append_error_test_failure_protocol(self):
    """Tests that the error test failure protocol is appended to the error."""

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        Exception,
        # Check that the appended note does not have leading spaces.
        r"\nNOTE: This test might fail depending on which PyTorch version is"
        " being used.",
    ):
      with et.assert_raises_message(
          RuntimeError,
          tpu="""TPU dummy error message for errors_test.py failure protocol""",
          gpu="""GPU dummy error message for errors_test.py failure protocol""",
          message_reviewed_by="wan",
      ):
        raise RuntimeError("Not the dummy error message")

  def test_why_tpu_only_validation(self):
    """Tests that why_tpu_only decorator requires a non-empty reason string."""
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        "non-empty",
    ):
      et.why_tpu_only("")

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        "non-empty",
    ):
      et.why_tpu_only("   ")

  def test_tpu_only_error_test_base_enforces_decorator(self):
    """Tests that TpuOnlyErrorTestBase enforces @why_tpu_only decorator."""

    class DummyTest(et.TpuOnlyErrorTestBase):

      def test_missing(self):
        pass

    test_instance = DummyTest("test_missing")
    old_mode = flags.FLAGS.test_mode
    flags.FLAGS.test_mode = "tpu"
    try:
      with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
          AssertionError,
          "TpuOnlyErrorTestBase",
      ):
        test_instance.setUp()
    finally:
      flags.FLAGS.test_mode = old_mode

  def test_why_tpu_only_works_for_parameterized_tests(self):
    """Tests that @why_tpu_only decorator works for parameterized tests."""

    class DummyParameterizedTest(
        et.TpuOnlyErrorTestBase, parameterized.TestCase
    ):

      @parameterized.named_parameters(
          {"testcase_name": "param1", "value": 1},
          {"testcase_name": "param2", "value": 2},
      )
      @et.why_tpu_only("Valid reason for parameterized test")
      def test_param(self, value):
        pass

    test_instance1 = DummyParameterizedTest("test_param_param1")
    test_instance2 = DummyParameterizedTest("test_param_param2")
    old_mode = flags.FLAGS.test_mode
    flags.FLAGS.test_mode = "tpu"
    try:
      test_instance1.setUp()
      test_instance2.setUp()
    finally:
      flags.FLAGS.test_mode = old_mode

  def test_standalone_method_in_parameterized_test_enforces_decorator(self):
    """Tests standalone method in parameterized test enforces decorator."""

    class DummyParameterizedTest(
        et.TpuOnlyErrorTestBase, parameterized.TestCase
    ):

      def test_standalone_missing(self):
        pass

    test_standalone = DummyParameterizedTest("test_standalone_missing")
    old_mode = flags.FLAGS.test_mode
    flags.FLAGS.test_mode = "tpu"
    try:
      with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
          AssertionError,
          "TpuOnlyErrorTestBase",
      ):
        test_standalone.setUp()
    finally:
      flags.FLAGS.test_mode = old_mode


if __name__ == "__main__":
  absltest.main()
