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
# pylint: skip-file

"""Tests for deferred_execution.py — verifies each executable cell runs without error."""

"""Tests for deferred_execution.py — verifies each executable cell runs without error."""
import torch
from torch_tpu._internal import execution_mode as em


def test_factory_demo(device):
  """Cell: factory_demo — creates tensors, defers operations, materializes."""
  # This records a "recipe" to create ones — no physical TPU memory is used yet
  t1 = torch.ones((10, 10), device=device, dtype=torch.bfloat16)
  assert t1.shape == (10, 10)
  assert t1.dtype == torch.bfloat16

  # This also records a recipe — still no hardware execution
  t2 = torch.arange(
      100, device=device, dtype=torch.bfloat16
  )  # StableHLO Constant
  t3 = t1 + t2.reshape(10, 10)  # Records a reshape (DeferredOp) and add node
  assert t3.shape == (10, 10)

  # NOW materialization happens — the whole graph is compiled and executed
  result = t3.cpu()
  assert result.shape == (10, 10)
  assert result.device.type == "cpu"


def test_item_demo(device):
  """Cell: item_demo — verifies .item() and implicit materialization."""
  # Create two scalar tensors
  v1 = torch.tensor(5.0, device=device, dtype=torch.bfloat16)
  v2 = torch.tensor(3.0, device=device, dtype=torch.bfloat16)

  # Perform a comparison
  cond = v1 > v2

  # Using .item() forces materialization to get the python boolean
  is_greater = cond.item()
  assert is_greater is True

  # Using the tensor in an `if` statement implicitly calls .item()
  if cond:
    pass  # Should not raise error


def test_eager_mode_debug(device):
  """Verifies the EagerMode.DEFER_NEVER debug snippet."""
  # Force every operation to materialize immediately
  with em.eager_mode(em.EagerMode.DEFER_NEVER):
    # In DEFER_NEVER mode, ops should execute immediately
    x = torch.ones((5, 5), device=device)
    y = x * 2
    assert y.sum().item() == 50.0
