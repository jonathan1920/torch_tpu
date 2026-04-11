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

"""Determine bounds of symbolic integers."""

from __future__ import annotations
from absl import logging
import sympy
import torch
from torch.utils._sympy.numbers import int_oo


def _lookup_known_bounds(sym: torch.SymInt) -> tuple[int, int] | None:
  """Looks up known bounds for a symbol, i.e. via mark_dynamic, else None."""
  shape_env = sym.node.shape_env
  expr = sym.node.expr
  if not isinstance(expr, sympy.Symbol):
    # var_to_range only contains bounds for individual sympy.Symbol instances,
    # not for complex expressions.
    return None
  vr = shape_env.var_to_range.get(expr, None)
  if not vr:
    return None

  def is_valid_bound(s: sympy.Expr) -> bool:
    return s.is_integer and s.is_constant() and (s not in (int_oo, -int_oo))

  if not is_valid_bound(vr.lower) or not is_valid_bound(vr.upper):
    return None
  return (int(vr.lower), int(vr.upper))


def get_symint_bounds(sym_int: torch.SymInt) -> tuple[int, int]:
  """Gets lower and upper bounds for a given SymInt.

  Args:
    sym_int: The SymInt to get the bounds for.

  Returns:
    A tuple of (lower_bound, upper_bound) for the SymInt.

  Raises:
    RuntimeError: If bounds are not found via shape_env and no hint is
      provided for the symbol.

  If bounds are not found via shape env, current algorithm determines:
    - lower_bound using the hint, if hint is present.
    - upper_bound as 2 * lower_bound.

  Note: Bound selection algorithms will be explored separately.

  If a shape dimension is dependent on tensor data (e.g., nonzero(), item()),
  PyTorch generates an "unbacked" SymInt which has no concrete backing value
  at trace time, resulting in hint = None.
  """

  if vr := _lookup_known_bounds(sym_int):
    lower_bound, upper_bound = vr
  else:
    hint = sym_int.node.hint
    if hint is None:
      raise RuntimeError(
          f"Cannot determine bounds for dynamic symbol: {sym_int}."
      )
    lower_bound = hint
    upper_bound = lower_bound * 2

  logging.debug("symint: %s, lb: %s, ub: %s", sym_int, lower_bound, upper_bound)
  return lower_bound, upper_bound
