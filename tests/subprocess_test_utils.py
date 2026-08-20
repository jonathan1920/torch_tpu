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

"""Utilities for two-stage parent-child subprocess test execution.

Several TorchTPU test files use a two-stage execution model where a "parent"
test class spawns isolated child subprocesses to run individual test methods.
This is needed when the code under test relies on process-global state (e.g.,
C++ static caches, environment variables read once at startup) that cannot be
safely reset between test cases within a single process.

This module extracts the common scaffolding shared by those test files:

* `QueueWriter` — a file-like object that forwards writes to a
  `multiprocessing.Queue`, used to redirect child stdout/stderr back to the
  parent process for logging.

* `sub_test_worker_entry` — the generic child-process entry point that applies
  environment mutations, installs `QueueWriter`, rewrites `sys.argv` to target
  a single test method, and calls `absltest.main()`.

* `SubprocessTestMixin` — a mixin providing `run_sub_test()`, which spawns an
  isolated child process, drains its output queue, and asserts a zero exit code.
"""

import enum
import io
import os
import queue
import sys
from typing import Mapping, Sequence

from absl import logging
from absl.testing import absltest
from torch_tpu._internal.distributed import multiprocessing


class QueueWriter:
  """File-like object forwarding writes to a multiprocessing queue.

  Used to redirect stdout/stderr in child subprocesses so that output appears
  in the parent process's test logs.
  """

  def __init__(self, q: queue.Queue[str]):
    self.q = q

  def write(self, s: str | None) -> None:
    if s:
      self.q.put(s)

  def flush(self) -> None:
    pass

  def fileno(self) -> None:
    raise io.UnsupportedOperation("QueueWriter does not have a fileno")


def sub_test_worker_entry(
    q: queue.Queue[str],
    mode: enum.Enum,
    test_method_template: str,
    env_updates: Mapping[str, str] | None = None,
    env_removals: Sequence[str] | None = None,
) -> None:
  """Generic child-process entry point for two-stage subprocess tests.

  This function executes inside a newly spawned child process. It applies the
  requested environment mutations, redirects stdout/stderr to the parent via
  `q`, and rewrites `sys.argv` so that `absltest.main()` runs only the
  single test method identified by `test_method_template`.

  Args:
    q: A multiprocessing queue for forwarding stdout/stderr to the parent.
    mode: The enum member selecting which test mode / method to run.
    test_method_template: A format string producing the fully qualified test
      method name. It is formatted with `mode=mode`, so it can contain
      `{mode.name}` or `{mode.value}` placeholders. For example:
      `"EnvVarsTest.test_{mode.value}"` or
      `"CacheTest.test_persistent_cache_{mode.value}"`.
    env_updates: Environment variables to set in the child process.
    env_removals: Environment variable keys to remove in the child process.
  """
  if env_removals:
    for key in env_removals:
      os.environ.pop(key, None)
  if env_updates:
    os.environ.update(env_updates)

  qw = QueueWriter(q)
  sys.stdout = qw
  sys.stderr = qw

  sys.argv = [
      sys.argv[0],
      f"--test_mode={mode.name}",
      test_method_template.format(mode=mode),
  ]

  absltest.main()


class SubprocessTestMixin:
  """Mixin providing `run_sub_test` for parent test classes.

  The mixin is intentionally not a full `TestCase` subclass so it can be
  freely composed with any test base class (e.g., `RepeatableTest`).

  Subclasses must set the class attribute `WORKER_TEST_METHOD_TEMPLATE` to
  a format string that produces the target test method name when formatted
  with `mode=<TestMode enum member>`.

  Example::

      class ParentCacheTest(
          subprocess_test_utils.SubprocessTestMixin,
          seed_test_utils.RepeatableTest,
      ):
        WORKER_TEST_METHOD_TEMPLATE = (
            "CacheTest.test_persistent_cache_{mode.value}"
        )
  """

  # Subclasses must override with a format string, e.g.
  # "CacheTest.test_persistent_cache_{mode.value}"
  WORKER_TEST_METHOD_TEMPLATE: str

  def fail(self, msg: str) -> None:
    # Provided by the concrete test class.
    ...  # pragma: no cover

  def run_sub_test(
      self,
      mode: enum.Enum,
      env_updates: Mapping[str, str] | None = None,
      env_removals: Sequence[str] | None = None,
  ) -> None:
    """Spawns an isolated child subprocess to run a single test method.

    Args:
      mode: The test-mode enum member selecting the child test method.
      env_updates: Environment variables to set in the child process.
      env_removals: Environment variable keys to remove in the child process.
    """
    logging.info("Running subtest %s in subprocess for isolation.", mode.name)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=sub_test_worker_entry,
        args=(
            q,
            mode,
            self.WORKER_TEST_METHOD_TEMPLATE,
            env_updates,
            env_removals,
        ),
    )
    p.start()

    while p.is_alive() or not q.empty():
      try:
        output = q.get(timeout=0.1)
        sys.stderr.write(output)
        sys.stderr.flush()
      except queue.Empty:
        continue

    p.join()
    if p.exitcode != 0:
      self.fail(f"Subtest {mode.name} failed with exit code {p.exitcode}")
