# Copyright 2025 Google LLC
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

"""Utilities for testing error messages."""

import contextlib
import enum
import functools
import re

from absl import flags
from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import testing as tt_testing


TEST_MODE = flags.DEFINE_string(
    name="test_mode",
    required=True,
    default=None,
    help=(
        "Mode of the test. Valid options are 'tpu' (verify the error messages"
        " on TPU), 'cpu' (verify the error messages on CPU), and 'cov' (measure"
        " the error message coverage on TPU). Note that the 'cov' mode does"
        " NOT verify the correctness of the error messages (it only collects"
        " the coverage stats). Therefore running the test in the 'tpu' mode "
        " is still necessary."
    ),
)


@functools.lru_cache(maxsize=1)
def device():
  """Returns the device to use for testing the error messages."""

  if TEST_MODE.value in ("tpu", "cov"):
    return api.tpu_device()
  if TEST_MODE.value == "cpu":
    return torch.device("cpu")
  raise ValueError(f"Unsupported test mode: {TEST_MODE.value}")


def set_up_module(init_torch_tpu: bool = True):
  """Called by absltest after flags are parsed and before tests are run.

  Args:
    init_torch_tpu: Whether to initialize torch_tpu. This should be set to False
      for distributed tests, which need to initialize torch_tpu by itself.
  """

  if TEST_MODE.value in ("tpu", "cov") and init_torch_tpu:
    api.tpu_device()  # Initialize torch_tpu.


# Matches a source location in the format of "file:line: ...".
_SOURCE_LOC_RE = re.compile(r"([/\w\.]+:\d+):.*")

# Matches a line in the format of "Exception raised from ... at file:line ...".
_EXCEPTION_RAISED_RE = re.compile(
    r"Exception raised from .* at ([/\w\.]+:\d+).*"
)


def _parse_error(msg: str):
  """Parses the message and prints the error macro locations it covers."""
  # An error message looks like:
  #
  #   empty(): product of dimension sizes ...
  #
  #   C++ error trace (starting from the origin):
  #   third_party/py/torch_tpu/common/error_utils.cc:53: SafeMultiply()
  #   third_party/py/torch_tpu/common/error_utils.cc:101: ValidateTensorByteSize()
  #   third_party/py/torch_tpu/device_buffer.cc:361: CreateEmpty()
  #   third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:90: MakeEmptyBuffer()
  #
  #   Exception raised from operator() at third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:155 (most recent call first):
  #   ...
  #
  # We need to extract the source locations from the C++ error trace and the
  # "Exception raised from ..." line.
  for line in msg.splitlines():
    m = _SOURCE_LOC_RE.match(line)
    if m:
      print(f"\nCovered {m.group(1)}")
    else:
      m = _EXCEPTION_RAISED_RE.match(line)
      if m:
        print(f"\nCovered {m.group(1)}")
        break


class _MatchType(enum.Enum):
  """The type of match to perform on the error message."""

  EXACT = "exact"  # Verify that the message is exactly equal to the expected.
  SUFFIX = "suffix"  # Verify that the message ends with the expected message.


def _check_error_message(
    msg: str,
    expected_msg: str | re.Pattern[str],
    device_type: str,
    match_type: _MatchType = _MatchType.EXACT,
):
  """Checks that the error message matches the expected message.

  Or, in the coverage mode and when `device_type` is "tpu", parses the error
  message and prints the covered error macro locations to stdout.

  Args:
    msg: The error message to check.
    expected_msg: The expected error message. Can be either a string or a
      regular expression (the result of re.compile()). Prefer using a string. A
      regular expression should only be used when the error message contains
      unstable information such as a source line number, which should be
      extremely rare as we strive at not leaking implementation details in the
      error messages.
    device_type: The device the error message is for (either "tpu" or "cpu").
    match_type: The type of match to perform on the error message.
  """

  if TEST_MODE.value == "cov":
    if device_type == "tpu":
      _parse_error(msg)
    return

  if match_type == _MatchType.SUFFIX:
    assert isinstance(
        expected_msg, str
    ), "The expected message must be a string for suffix match."
    assert msg.endswith(expected_msg), (
        f"Expected the {device_type} message to end with:\n{expected_msg}\nbut"
        f" got:\n{msg}"
    )
    return

  if isinstance(expected_msg, str):
    assert (
        expected_msg == msg
    ), f"Expected the {device_type} message:\n{expected_msg}\nbut got:\n{msg}"
  else:  # `expected_msg` is a regex.
    # We don't use assertRegex as it does a substring match, while we want
    # a full match.
    assert expected_msg.fullmatch(msg), (
        f"Expected the {device_type} message to match:\n{expected_msg}\nbut"
        f" got:\n{msg}"
    )


@contextlib.contextmanager
def assert_raises_message(
    exception_type,
    tpu: str | re.Pattern[str],
    cpu: str | re.Pattern[str] | None = None,
    message_reviewed_by: str | None = None,
):
  """Asserts that a specific exception and message is raised.

  The expected error message can be either a string or a regular expression
  (the result of re.compile()). Prefer using a string. A regular expression
  should only be used when the error message contains unstable information
  such as a source line number, which should be extremely rare as we strive
  at not leaking implementation details in the error messages.

  Args:
    exception_type: The expected exception type.
    tpu: The expected error message on TPU.
    cpu: The expected error message on CPU. If not provided, it is assumed to be
      the same as the TPU error message.
    message_reviewed_by: The LDAP of the engineer who reviewed the error
      message. This will be used by a future tool to find all error messages not
      reviewed by an engineer.
  """

  del message_reviewed_by  # Unused for now.

  if cpu is None:
    cpu = tpu

  try:
    yield
  except exception_type as e:
    msg = str(e)
    if TEST_MODE.value == "cpu":
      _check_error_message(msg, cpu, "cpu")
    else:
      _check_error_message(msg, tpu, "tpu")
  except BaseException as e:
    raise AssertionError(
        f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}"
    ) from e
  else:
    raise AssertionError(
        f"Expected {exception_type.__name__} to be raised, but no exception"
        " was raised."
    )


@contextlib.contextmanager
def assert_subprocess_raises_message(exception_type, expected_msg: str):
  """Asserts that a specific exception and message is raised by a subprocess.

  Args:
    exception_type: The expected exception type.
    expected_msg: The expected error message.
  """
  try:
    yield
  # When a subprocess raises an exception of type FooError, the parent process
  # translates it to a ProcessRaisedException with a message like:
  #
  #   -- Process 0 terminated with the following error:
  #   Traceback (most recent call last):
  #     ...
  #   FooError: <exception message>
  #
  # Therefore we verify that the parent process message ends with
  #   FooError: <expected message>
  except mp.ProcessRaisedException as e:
    msg = str(e)
    expected_msg_suffix = f"\n{exception_type.__name__}: {expected_msg}"
    _check_error_message(
        msg.rstrip(), expected_msg_suffix, "tpu", _MatchType.SUFFIX
    )
  except BaseException as e:
    raise AssertionError(
        f"Expected ProcessRaisedException but got {type(e).__name__}: {e}"
    ) from e
  else:
    raise AssertionError(
        "Expected ProcessRaisedException to be raised, but no exception"
        " was raised."
    )


class ErrorTestBase(absltest.TestCase):
  """Base class for error tests."""

  def setUp(self):
    super().setUp()
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")

  def tearDown(self):
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")
    super().tearDown()


class TpuOnlyErrorTestBase(ErrorTestBase):
  """Base class for error tests that are only relevant for TPU."""

  def setUp(self):
    super().setUp()
    if TEST_MODE.value == "cpu":
      self.skipTest("Skipping test on CPU as it is only relevant for TPU.")
