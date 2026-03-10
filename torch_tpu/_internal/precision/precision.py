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

"""Op precision context manager."""

import contextlib
import contextvars
import enum
from typing import Generator

import torch_tpu._internal.precision.precision_impl as precision_impl  # pylint: disable=consider-using-from-import

# pylint: disable=protected-access

__all__ = ["precision", "Precision"]


class Precision(enum.Enum):
  """StableHLO PrecisionConfig.

  This enum provides a direct binding to the StableHLO precision configuration.
  For explicit definitions of 'DEFAULT', 'HIGH', and 'HIGHEST', please see the
  StableHLO documentation for 'dot_general':
  https://openxla.org/stablehlo/spec#dot_general
  """

  DEFAULT = "DEFAULT"
  HIGH = "HIGH"
  HIGHEST = "HIGHEST"


_precision_var = contextvars.ContextVar("precision", default=Precision.DEFAULT)

_PRECISION_CONFIG_TO_IMPL = {
    Precision.DEFAULT: precision_impl.Precision.default,
    Precision.HIGH: precision_impl.Precision.high,
    Precision.HIGHEST: precision_impl.Precision.highest,
}


@contextlib.contextmanager
def precision(mode: Precision) -> Generator[None, None, None]:
  """Context manager to set op precisions.

  This provides a direct binding to the StableHLO precision configuration. It is
  primarily used for operations involving matrix multiplication (e.g.,
  matmul and convolution). On TPU, float32 matrix multiplications are natively
  executed in bfloat16. This context manager allows you to increase the
  precision of these operations, mapping directly to StableHLO precision
  settings.

  WARNING: This has different behavior compared to
  `torch.set_float32_matmul_precision` which has unpredictable precision
  behavior across varying GPUs. While StableHLO ops are generally stable, their
  lowering to device-specific ISA is not fully documented, and as a result
  precision may vary across different hardware.

  Args:
      mode: The precision to use. Must be a Precision enum. See the StableHLO
        documentation for details on the numerical behavior of each mode:
        https://openxla.org/stablehlo/spec#dot_general

  Yields:
      None
  """
  if mode not in _PRECISION_CONFIG_TO_IMPL:
    raise ValueError(
        "expected to be one of 'PrecisionConfig.DEFAULT',"
        f" 'PrecisionConfig.HIGH', or 'PrecisionConfig.HIGHEST', got '{mode}'"
    )

  precision_config = _precision_var.set(mode)
  # TODO: b/489872437 - ContextVar should be the source of truth. We shouldn't
  # maintain another SoT in C++. When we need the value in C++, we should get it
  # from the Python ContextVar.
  precision_impl._set_precision(_PRECISION_CONFIG_TO_IMPL[mode])
  try:
    yield
  finally:
    _precision_var.reset(precision_config)
    precision_impl._set_precision(
        _PRECISION_CONFIG_TO_IMPL[_precision_var.get()]
    )
