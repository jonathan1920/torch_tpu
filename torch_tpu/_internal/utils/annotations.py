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
metadata (__tt_api_stage__, __tt_api_stage_reason__,
__tt_api_deprecated_version__) to
functions, methods, or classes for automatic inspection by scanner tools, and
emit standard Python runtime warnings (UserWarning, DeprecationWarning) on the
first invocation per process.
"""

import enum
import functools
from typing import Any, Callable, Optional, TypeVar
import warnings

_T = TypeVar("_T")

TT_API_STAGE = "__tt_api_stage__"
TT_API_STAGE_REASON = "__tt_api_stage_reason__"
TT_API_DEPRECATED_VERSION = "__tt_api_deprecated_version__"


class Stage(str, enum.Enum):
  """Lifecycle stages for TorchTPU APIs."""

  STABLE = "Stable"
  EXPERIMENTAL = "Experimental"
  DEPRECATED = "Deprecated"

  @property
  def warning_category(self) -> Optional[type[Warning]]:
    """The Warning category for this stage."""
    if self == Stage.EXPERIMENTAL:
      return UserWarning
    if self == Stage.DEPRECATED:
      return DeprecationWarning
    return None

  def format_warning_message(
      self, name: str, reason: str, version: Optional[str] = None
  ) -> str:
    """Formats the warning message for this stage."""
    if self == Stage.EXPERIMENTAL:
      return f"{name} is experimental and subject to change: {reason}"
    if self == Stage.DEPRECATED:
      return (
          f"{name} is deprecated as of TorchTPU {version} and will be"
          f" removed in a future release: {reason}"
      )
    return ""


def _safe_setattr(target: Any, name: str, value: Any) -> None:
  """Safely sets an attribute on target, swallowing AttributeError and TypeError.

  Args:
    target: The target object to modify.
    name: The attribute name to set.
    value: The attribute value to assign.
  """
  try:
    setattr(target, name, value)  # pylint: disable=protected-access
  except (AttributeError, TypeError):
    pass


def _annotate_target(
    target: _T,
    stage: Stage,
    reason: str,
    version: Optional[str] = None,
) -> _T:
  """Attaches stage metadata and warning wrapper to a target function.

  Args:
    target: The function, method, or class to annotate.
    stage: The lifecycle stage of the target.
    reason: Explanation for the stage. Required to be non-empty for experimental
      and deprecated stages.
    version: The version string when the target was deprecated (e.g. "2.13").
      Only used for deprecated targets.

  Returns:
    The annotated target.
  """
  if stage.warning_category:
    _warned = False

    @functools.wraps(target)  # type: ignore[arg-type]
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      nonlocal _warned
      if not _warned:
        func_name = getattr(target, "__name__", str(target))
        msg = stage.format_warning_message(func_name, reason, version)
        cat = stage.warning_category
        if cat:
          warnings.warn(msg, category=cat, stacklevel=2)
        _warned = True
      return target(*args, **kwargs)  # type: ignore[operator]

    _safe_setattr(wrapper, TT_API_STAGE, stage.value)
    _safe_setattr(wrapper, TT_API_STAGE_REASON, reason)
    if version:
      _safe_setattr(wrapper, TT_API_DEPRECATED_VERSION, version)
    return wrapper  # type: ignore[return-value]

  _safe_setattr(target, TT_API_STAGE, stage.value)
  _safe_setattr(target, TT_API_STAGE_REASON, reason)
  if version:
    _safe_setattr(target, TT_API_DEPRECATED_VERSION, version)
  return target


def experimental(reason: str) -> Callable[[_T], _T]:
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

  return lambda target: _annotate_target(
      target, stage=Stage.EXPERIMENTAL, reason=reason
  )


def stable(reason: str = "") -> Callable[[_T], _T]:
  """Marks a TorchTPU Python API as stable.

  Stable APIs are production-ready and backward compatibility is guaranteed. No
  runtime warnings are triggered.

  Args:
    reason: Optional description or justification for stable status.

  Returns:
    The decorated function with stable stage metadata attached.
  """

  return lambda target: _annotate_target(
      target, stage=Stage.STABLE, reason=reason
  )


def deprecated(version: str, reason: str) -> Callable[[_T], _T]:
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

  return lambda target: _annotate_target(
      target, stage=Stage.DEPRECATED, reason=reason, version=version
  )
