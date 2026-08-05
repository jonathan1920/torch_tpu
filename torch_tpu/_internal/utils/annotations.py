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

"""TorchTPU API lifecycle stage annotations and decorators.

This module provides decorators (@experimental, @stable, @deprecated) to mark
the lifecycle stage of TorchTPU Python APIs. These decorators attach stage
metadata (__tt_api_stage__, __tt_stage_reason__, __tt_deprecated_version__) to
functions, methods, or classes for automatic inspection by scanner tools, and
emit standard Python runtime warnings (UserWarning, DeprecationWarning) on the
first invocation per process.
"""

import functools
from typing import Any, Callable, TypeVar
import warnings

_F = TypeVar("_F", bound=Callable[..., Any])


# TODO(yilingyuan): Refactor to share common implementation between the 3
# decorators (@experimental, @stable, @deprecated).
def experimental(reason: str) -> Callable[[_F], _F]:
  """Marks a TorchTPU Python API as experimental.

  Experimental APIs are exposed under public namespaces (e.g. torch.tpu.*) for
  early trial, but their signature and behavior are subject to change.

  Calling an experimental API will trigger a one-time UserWarning per process.

  Args:
    reason: Non-empty explanation for why the API is experimental.

  Returns:
    The decorated function with experimental stage metadata attached.
  """
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("@experimental requires a non-empty reason string.")

  def decorator(func: _F) -> _F:
    func.__tt_api_stage__ = "Experimental"  # pylint: disable=protected-access
    func.__tt_stage_reason__ = reason  # pylint: disable=protected-access

    _warned = False

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      nonlocal _warned
      if not _warned:
        func_name = getattr(func, "__name__", str(func))
        warnings.warn(
            f"{func_name} is experimental and subject to change: {reason}",
            category=UserWarning,
            stacklevel=2,
        )
        _warned = True
      return func(*args, **kwargs)

    wrapper.__tt_api_stage__ = "Experimental"  # pylint: disable=protected-access
    wrapper.__tt_stage_reason__ = reason  # pylint: disable=protected-access
    return wrapper

  return decorator


def stable(reason: str = "") -> Callable[[_F], _F]:
  """Marks a TorchTPU Python API as stable.

  Stable APIs are production-ready and backward compatibility is guaranteed. No
  runtime warnings are triggered.

  Args:
    reason: Optional description or justification for stable status.

  Returns:
    The decorated function with stable stage metadata attached.
  """

  def decorator(func: _F) -> _F:
    func.__tt_api_stage__ = "Stable"  # pylint: disable=protected-access
    func.__tt_stage_reason__ = reason  # pylint: disable=protected-access
    return func

  return decorator


def deprecated(version: str, reason: str) -> Callable[[_F], _F]:
  """Marks a TorchTPU Python API as deprecated.

  Deprecated APIs will be removed in a future release. Calling a deprecated API
  will trigger a one-time DeprecationWarning per process, pointing users to the
  replacement API or migration guide.

  Args:
    version: The version string when the API was deprecated (e.g. "2.13").
    reason: Guidance on replacement APIs and planned removal timeline.

  Returns:
    The decorated function with deprecated stage metadata attached.
  """
  if not isinstance(version, str) or not version.strip():
    raise ValueError(
        "@deprecated requires a non-empty version string (e.g. version='2.13')."
    )
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("@deprecated requires a non-empty reason string.")

  def decorator(func: _F) -> _F:
    func.__tt_api_stage__ = "Deprecated"  # pylint: disable=protected-access
    func.__tt_deprecated_version__ = version  # pylint: disable=protected-access
    func.__tt_stage_reason__ = reason  # pylint: disable=protected-access

    _warned = False

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      nonlocal _warned
      if not _warned:
        func_name = getattr(func, "__name__", str(func))
        warnings.warn(
            f"{func_name} is deprecated as of TorchTPU {version} and will be"
            f" removed in a future release: {reason}",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
      return func(*args, **kwargs)

    wrapper.__tt_api_stage__ = "Deprecated"  # pylint: disable=protected-access
    wrapper.__tt_deprecated_version__ = version  # pylint: disable=protected-access
    wrapper.__tt_stage_reason__ = reason  # pylint: disable=protected-access
    return wrapper

  return decorator
