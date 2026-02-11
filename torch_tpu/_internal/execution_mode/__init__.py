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

from contextlib import contextmanager
from typing import Set, TypeAlias

import torch
from torch_tpu._internal import execution_mode_impl
from torch_tpu._internal.execution_mode_impl import set_defer_mode

DeferMode: TypeAlias = execution_mode_impl.DeferMode


@contextmanager
def defer_mode(mode: DeferMode):
  """Context manager for setting the execution mode."""
  old_defer_mode = execution_mode_impl.get_defer_mode()
  try:
    execution_mode_impl.set_defer_mode(mode)
    yield
  finally:
    execution_mode_impl.set_defer_mode(old_defer_mode)


@contextmanager
def cpu_fallback_mode(enabled: bool):
  """Context manager for setting the fallback mode."""
  old_enabled = execution_mode_impl.is_cpu_fallback_enabled()
  try:
    execution_mode_impl.enable_cpu_fallback(enabled)
    yield
  finally:
    execution_mode_impl.enable_cpu_fallback(old_enabled)


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "DeferMode",
    "cpu_fallback_mode",
    "defer_mode",
    "set_defer_mode",
    # go/keep-sorted end
]
