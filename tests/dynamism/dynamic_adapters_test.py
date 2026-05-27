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

"""Unit tests for DynamicAdapterLinearHypothesis."""

from absl.testing import absltest
from torch_tpu._internal.compile.dynamic.dynamic_adapters import DynamicAdapterLinearHypothesis
from torch_tpu._internal.compile.dynamic.dynamic_adapters import ShapeBoundInfo


class DynamicAdaptersTest(absltest.TestCase):

  def test_linear_hypothesis_standard_flow(self):
    # Test setup: 2 tensors.
    # tensor 0: static shape [10]
    # tensor 1: dynamic shape [B, 10] where B <= 16, dynamic dimension is
    # index 0
    bounds_list = [
        ShapeBoundInfo(dynamic_dims=[], upper_bounds=[]),
        ShapeBoundInfo(dynamic_dims=[0], upper_bounds=[16]),
    ]
    hypothesis = DynamicAdapterLinearHypothesis(bounds_list)

    # Step 1: First call (S=3)
    # Sets the origin
    hypothesis.update([[10], [3, 10]])
    self.assertTrue(hypothesis.is_linear)
    self.assertEqual(hypothesis._origin, {(1, 0): 3})
    self.assertIsNone(hypothesis._base_step)

    # Step 2: Second call (S=5)
    # Establishes the line: slope = 5 - 3 = 2
    hypothesis.update([[10], [5, 10]])
    self.assertTrue(hypothesis.is_linear)
    self.assertEqual(hypothesis._base_step, {(1, 0): 2})
    self.assertEqual(hypothesis._last_t_seen, 1)

    # Test precompilation predictions for next 2 steps (t=2, t=3)
    # Expected S values:
    # t=2 -> 3 + 2 * 2 = 7
    # t=3 -> 3 + 2 * 3 = 9
    updates = list(hypothesis.get_shape_updates(num_steps=2))
    self.assertLen(updates, 2)
    self.assertEqual(updates[0], {(1, 0): 7})
    self.assertEqual(updates[1], {(1, 0): 9})
    self.assertEqual(hypothesis._last_t_produced, 3)

    # Step 3: Third call (S=7)
    # Verifies colinearity (7 = 3 + 2 * 2 -> t=2)
    hypothesis.update([[10], [7, 10]])
    self.assertTrue(hypothesis.is_linear)
    self.assertEqual(hypothesis._last_t_seen, 2)

    # Test precompilation deduplication:
    # Since last_t_produced is 3, calling get_shape_updates(2) at last_t_seen=2
    # wants shapes up to last_t_seen + 2 = 4.
    # It should only yield t=4 (S = 3 + 2 * 4 = 11)
    new_updates = list(hypothesis.get_shape_updates(num_steps=2))
    self.assertLen(new_updates, 1)
    self.assertEqual(new_updates[0], {(1, 0): 11})
    self.assertEqual(hypothesis._last_t_produced, 4)

  def test_non_linear_detection(self):
    bounds_list = [
        ShapeBoundInfo(dynamic_dims=[0], upper_bounds=[16]),
    ]
    hypothesis = DynamicAdapterLinearHypothesis(bounds_list)

    # Step 1: S=3
    hypothesis.update([[3]])

    # Step 2: S=5 (slope = 2)
    hypothesis.update([[5]])

    # Step 3: S=6 (non-colinear: 6 - 3 = 3 which is not divisible by 2)
    hypothesis.update([[6]])

    self.assertFalse(hypothesis.is_linear)
    # get_shape_updates should return empty generator
    self.assertEqual(list(hypothesis.get_shape_updates(2)), [])

  def test_bounds_violation(self):
    bounds_list = [
        ShapeBoundInfo(dynamic_dims=[0], upper_bounds=[10]),
    ]
    hypothesis = DynamicAdapterLinearHypothesis(bounds_list)

    # S=12 exceeds upper bound 10 -> should raise AssertionError
    with self.assertRaises(AssertionError):
      hypothesis.update([[12]])

  def test_bounds_enforcement_in_shape_updates(self):
    # Dynamic dimension index 0, upper bound is 10
    bounds_list = [
        ShapeBoundInfo(dynamic_dims=[0], upper_bounds=[10]),
    ]
    hypothesis = DynamicAdapterLinearHypothesis(bounds_list)

    # Origin S=6
    hypothesis.update([[6]])
    # Step 2: S=8 (slope = 2)
    hypothesis.update([[8]])

    # We want to precompile 2 steps ahead:
    # t=2 -> 6 + 2 * 2 = 10 (Equal to upper bound -> OK!)
    # t=3 -> 6 + 2 * 3 = 12 (Exceeds upper bound 10 -> Should NOT be produced!)
    # So get_shape_updates should ONLY yield t=2 (S=10).
    updates = list(hypothesis.get_shape_updates(num_steps=2))
    self.assertLen(updates, 1)
    self.assertEqual(updates[0], {(0, 0): 10})
    self.assertEqual(hypothesis._last_t_produced, 2)


if __name__ == "__main__":
  absltest.main()
