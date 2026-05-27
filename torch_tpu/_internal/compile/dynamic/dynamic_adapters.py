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

"""Maintain a linear hypothesis for the dynamic inputs passed to an executable."""

from collections.abc import Iterator
from collections.abc import Sequence
from typing import NamedTuple


class ShapeBoundInfo(NamedTuple):
  """Shape bound information for dynamic dimensions of a tensor.

  Attributes:
    dynamic_dims: A list of indices of the dynamic dimensions.
    upper_bounds: A list of upper bounds for each corresponding dynamic
      dimension.
  """

  dynamic_dims: list[int]
  upper_bounds: list[int]


class DynamicAdapterLinearHypothesis:
  """Linear hypothesis for runtime input shapes of a dynamic executable.

  Attributes:
    is_linear: A boolean indicating whether the hypothesis is linear.
  """

  def __init__(self, bounds_list: Sequence[ShapeBoundInfo]):
    # --- Public API ---
    self.is_linear: bool = True
    # --- Internal Linear Hypothesis State ---
    self._origin: dict[tuple[int, int], int] | None = None
    self._base_step: dict[tuple[int, int], int] | None = None
    self._upper_bounds: dict[tuple[int, int], int] = self._extract_upper_bounds(
        bounds_list
    )
    # --- Internal Step Counters & Caps ---
    self._last_t_seen: int | None = None
    self._last_t_produced: int | None = None
    self._max_t: int | None = None
    self._num_inputs: int = len(bounds_list)

  def _extract_upper_bounds(
      self, bounds_list: Sequence[ShapeBoundInfo]
  ) -> dict[tuple[int, int], int]:
    """Extracts the upper bounds for each dynamic dimension of each input tensor."""
    upper_bounds: dict[tuple[int, int], int] = {}
    for tensor_idx, bounds in enumerate(bounds_list):
      for dim_idx, upper_bound in zip(bounds.dynamic_dims, bounds.upper_bounds):
        upper_bounds[tensor_idx, dim_idx] = upper_bound
    return upper_bounds

  def _is_colinear(self, query: dict[tuple[int, int], int]) -> None | int:
    """Returns t, where query = origin + base_step * t, or None.

    Returns None if the query point is not colinear with the hypothesis or if we
    don't have enough information to determine colinearity.

    Args:
      query: The query point to check for colinearity.

    Returns:
      The t value if the query point is colinear with the hypothesis, otherwise
      None.
    """
    if self._base_step is None or self._origin is None:
      return None
    t = None
    for tensor_idx, dim_idx in self._origin.keys():
      step = self._base_step[tensor_idx, dim_idx]
      diff = query[tensor_idx, dim_idx] - self._origin[tensor_idx, dim_idx]
      if diff % step != 0:
        return None
      if t is None:
        t = diff // step
      elif t != diff // step:
        return None
    return t

  def _to_sparse(self, shapes: list[list[int]]) -> dict[tuple[int, int], int]:
    """Converts raw shape lists to a sparse dictionary of dynamic dimensions."""
    sparse_shapes: dict[tuple[int, int], int] = {}
    for tensor_idx, dim_idx in self._upper_bounds:
      sparse_shapes[tensor_idx, dim_idx] = shapes[tensor_idx][dim_idx]
    return sparse_shapes

  def update(self, shapes: list[list[int]]) -> None:
    """Updates the hypothesis based on the given input tensor shapes."""
    if not self.is_linear:
      return
    assert (
        len(shapes) == self._num_inputs
    ), f"Expected {self._num_inputs} shapes, but got {len(shapes)}."
    sparse_shapes = self._to_sparse(shapes)
    # Phase 1: Establish origin
    if self._origin is None:
      for (tensor_idx, dim_idx), dim_size in sparse_shapes.items():
        assert dim_size <= self._upper_bounds[tensor_idx, dim_idx]
      self._origin = sparse_shapes
      return
    # Phase 2: Establish linear slope (base_step) and max_t cap
    if self._base_step is None:
      diff = {}
      max_t = None
      for (tensor_idx, dim_idx), dim_size in sparse_shapes.items():
        assert dim_size <= self._upper_bounds[tensor_idx, dim_idx]
        if dim_size != self._origin[tensor_idx, dim_idx]:
          step = dim_size - self._origin[tensor_idx, dim_idx]
          diff[tensor_idx, dim_idx] = step
          # Calculate max_t only for dimensions that actually grow
          num_steps = (
              self._upper_bounds[tensor_idx, dim_idx]
              - self._origin[tensor_idx, dim_idx]
          ) // step
          if max_t is None:
            max_t = num_steps
          else:
            max_t = min(max_t, num_steps)
      if diff:
        self._base_step = diff
        self._last_t_seen = 1
        self._max_t = max_t
      return
    # Phase 3: Verify colinearity for subsequent calls
    t = self._is_colinear(sparse_shapes)
    if t is None:
      self.is_linear = False
      return
    self._last_t_seen = t

  def get_shape_updates(
      self, num_steps: int
  ) -> Iterator[dict[tuple[int, int], int]]:
    """Returns updates to the input tensor shapes for the next num_steps."""
    if (
        not self.is_linear
        or self._origin is None
        or self._base_step is None
        or self._last_t_seen is None
    ):
      return
    next_t = self._last_t_seen + 1
    last_t = self._last_t_seen + num_steps
    if self._max_t is not None:
      last_t = min(last_t, self._max_t)
    if self._last_t_produced is not None:
      next_t = max(next_t, self._last_t_produced + 1)
    if next_t > last_t:
      return
    for t in range(next_t, last_t + 1):
      diff: dict[tuple[int, int], int] = {}
      for (tensor_idx, dim_idx), step in self._base_step.items():
        diff[tensor_idx, dim_idx] = self._origin[tensor_idx, dim_idx] + step * t
      yield diff
    self._last_t_produced = last_t
