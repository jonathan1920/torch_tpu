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

"""Unit tests for PyTorch torch.jagged layout and jagged tensors on TPU.

Tests cover:
- NestedTensor construction, layout queries, and device transfers on TPU.
- Elementwise and linear operations on jagged nested tensors.
- Eager ATen conversion kernels:
    aten._jagged_to_padded_dense_forward
    aten._padded_dense_to_jagged_forward
- Parity against CPU ATen kernels across all supported dtypes and padding modes.
- Complex corner cases: truncation, 1D scalar tokens, zero-length items,
  all-empty batches, strided/non-contiguous offsets, and compilation cache keys.
"""

from absl.testing import absltest
import torch
from torch.nested._internal import nested_tensor as nt_internal
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class JaggedTensorTest(seed_test_utils.RepeatableTest):

  def _assert_jagged_to_padded(
      self, values_cpu, offsets_cpu, max_lengths, padding_value=0.0
  ):
    """Helper to verify jagged-to-padded parity between TPU and CPU.

    Executes aten._jagged_to_padded_dense_forward on both CPU and TPU, verifies
    that device, shape, and dtype match, and asserts numerical equality.

    Args:
      values_cpu: CPU tensor of shape (total_L, *features).
      offsets_cpu: 1D int64 CPU tensor of shape (batch_size + 1,).
      max_lengths: Target sequence length list [max_length].
      padding_value: Scalar constant for padding unoccupied elements.

    Returns:
      A tuple of (padded_tpu, padded_cpu).
    """
    padded_cpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_cpu, [offsets_cpu], max_lengths, padding_value
    )
    padded_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_cpu.to("tpu"),
        [offsets_cpu.to("tpu")],
        max_lengths,
        padding_value,
    )
    self.assertEqual(padded_tpu.device.type, "tpu")
    self.assertEqual(padded_tpu.shape, padded_cpu.shape)
    self.assertEqual(padded_tpu.dtype, padded_cpu.dtype)
    utils.assert_close(padded_tpu.cpu(), padded_cpu)
    return padded_tpu, padded_cpu

  def _assert_padded_to_jagged(self, dense_cpu, offsets_cpu, total_L=None):
    """Helper to verify padded-to-jagged parity between TPU and CPU.

    Executes aten._padded_dense_to_jagged_forward on both CPU and TPU, verifies
    that device, shape, and dtype match, and asserts numerical equality.

    Args:
      dense_cpu: CPU dense tensor of shape (batch_size, max_length, *features).
      offsets_cpu: 1D int64 CPU tensor of shape (batch_size + 1,).
      total_L: Optional expected number of elements in output values.

    Returns:
      A tuple of (jagged_tpu, jagged_cpu).
    """
    kwargs = {} if total_L is None else {"total_L": total_L}
    jagged_cpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_cpu, [offsets_cpu], **kwargs
    )
    jagged_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_cpu.to("tpu"), [offsets_cpu.to("tpu")], **kwargs
    )
    self.assertEqual(jagged_tpu.device.type, "tpu")
    self.assertEqual(jagged_tpu.shape, jagged_cpu.shape)
    self.assertEqual(jagged_tpu.dtype, jagged_cpu.dtype)
    utils.assert_close(jagged_tpu.cpu(), jagged_cpu)
    return jagged_tpu, jagged_cpu

  # ---------------------------------------------------------------------------
  # Core Jagged Layout & NestedTensor Tests
  # ---------------------------------------------------------------------------

  def test_empty_jagged(self):
    """Tests creating an empty tensor with layout=torch.jagged on TPU.

    Calling torch.empty with layout=torch.jagged allocates the underlying flat
    strided buffer that PyTorch uses to back jagged nested tensors.
    """
    t = torch.empty(5, layout=torch.jagged, device="tpu")
    self.assertEqual(t.device.type, "tpu")
    self.assertEqual(t.shape, (5,))
    self.assertEqual(t.layout, torch.strided)

  def test_nested_tensor_construction(self):
    """Tests constructing a nested tensor with layout=torch.jagged on TPU."""
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="tpu")
    t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.shape[0], 2)  # batch dimension
    self.assertEqual(nt.shape[2], 2)  # feature dimension

    # Verify that underlying values and offsets buffers live on TPU.
    values, offsets = nt.values(), nt.offsets()
    self.assertEqual(values.device.type, "tpu")
    self.assertEqual(offsets.device.type, "tpu")
    self.assertEqual(values.shape, (5, 2))
    utils.assert_close(
        offsets.cpu(), torch.tensor([0, 2, 5], dtype=torch.int64)
    )

  def test_as_nested_tensor(self):
    """Tests torch.nested.as_nested_tensor with layout=torch.jagged on TPU."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(5, 4, device="tpu")
    nt = torch.nested.as_nested_tensor([t1, t2], layout=torch.jagged)

    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.values().shape, (8, 4))
    utils.assert_close(
        nt.offsets().cpu(), torch.tensor([0, 3, 8], dtype=torch.int64)
    )

  def test_pointwise_operations_parity(self):
    """Tests that pointwise operations on jagged tensors match CPU."""
    t1_cpu = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    t2_cpu = torch.tensor([[5.0, 6.0], [-7.0, 8.0], [9.0, -10.0]])
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )

    # Scalar arithmetic operations
    utils.assert_close((nt_tpu + 2.5).values().cpu(), (nt_cpu + 2.5).values())
    utils.assert_close((nt_tpu * 1.5).values().cpu(), (nt_cpu * 1.5).values())

    # Unary activations
    utils.assert_close(
        torch.relu(nt_tpu).values().cpu(), torch.relu(nt_cpu).values()
    )
    utils.assert_close(
        torch.sin(nt_tpu).values().cpu(), torch.sin(nt_cpu).values()
    )

    # Binary tensor addition
    utils.assert_close(
        (nt_tpu + nt_tpu).values().cpu(), (nt_cpu + nt_cpu).values()
    )

  def test_linear_projection(self):
    """Tests linear layer projection on a jagged tensor on TPU."""
    t1_cpu = torch.randn(4, 8)
    t2_cpu = torch.randn(6, 8)
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)
    linear_cpu = torch.nn.Linear(8, 16)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )
    linear_tpu = torch.nn.Linear(8, 16).to("tpu")
    linear_tpu.load_state_dict(linear_cpu.state_dict())

    out_cpu = linear_cpu(nt_cpu)
    out_tpu = linear_tpu(nt_tpu)

    self.assertEqual(out_tpu.layout, torch.jagged)
    self.assertEqual(out_tpu.device.type, "tpu")
    self.assertEqual(out_tpu.values().shape, (10, 16))
    utils.assert_close(
        out_tpu.values().cpu(), out_cpu.values(), atol=1e-2, rtol=1e-2
    )

  def test_device_transfer(self):
    """Tests moving a jagged tensor between CPU and TPU."""
    t1 = torch.randn(2, 3)
    t2 = torch.randn(4, 3)
    nt_cpu = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Transfer CPU -> TPU
    nt_tpu = nt_cpu.to("tpu")
    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)

    # Transfer TPU -> CPU
    nt_back_to_cpu = nt_tpu.to("cpu")
    self.assertEqual(nt_back_to_cpu.device.type, "cpu")
    self.assertEqual(nt_back_to_cpu.layout, torch.jagged)
    utils.assert_close(nt_back_to_cpu.values(), nt_cpu.values())
    utils.assert_close(nt_back_to_cpu.offsets(), nt_cpu.offsets())

  # ---------------------------------------------------------------------------
  # Autograd & Backward Operator Derivative Tests
  # ---------------------------------------------------------------------------

  def test_to_padded_tensor_autograd_analytic_gradients(self):
    """Verifies analytical gradients of to_padded_tensor across dtypes."""
    for dtype in [torch.float32, torch.bfloat16, torch.float64]:
      values = torch.randn(5, 4, dtype=dtype, device="tpu", requires_grad=True)
      offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

      nt = nt_internal.nested_view_from_values_offsets(
          values, offsets, min_seqlen=2, max_seqlen=3
      )
      padded = nt.to_padded_tensor(0.0)

      self.assertEqual(padded.shape, (2, 3, 4))
      self.assertEqual(padded.dtype, dtype)
      self.assertEqual(padded.device.type, "tpu")

      # Backpropagate arbitrary weight matrix
      weights = torch.randn(2, 3, 4, dtype=dtype, device="tpu")
      loss = (padded * weights).sum()
      loss.backward()

      self.assertIsNotNone(values.grad)
      self.assertEqual(values.grad.shape, values.shape)
      self.assertEqual(values.grad.dtype, dtype)

      # Analytical expected grad: gather non-padded elements from weights
      expected_grad = torch.cat(
          [weights[0, :2, :], weights[1, :3, :]], dim=0
      ).cpu()
      utils.assert_close(values.grad.cpu(), expected_grad, atol=1e-3, rtol=1e-3)

  def test_to_padded_tensor_with_output_size_autograd(self):
    """Verifies autograd when padding to an explicit larger output_size."""
    values = torch.randn(
        5, 3, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )
    # Pad to explicit larger sequence length = 4 (batch 0 has 2 pads, batch 1 has 1 pad)
    padded = nt.to_padded_tensor(0.0, output_size=(2, 4, 3))

    self.assertEqual(padded.shape, (2, 4, 3))
    loss = padded.sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    grad_cpu = values.grad.cpu()

    # All active tokens must have gradient 1.0
    self.assertTrue(torch.all(grad_cpu == 1.0))

  def test_nested_from_padded_tensor_autograd(self):
    """Verifies gradient flow through _nested_from_padded_tensor."""
    dense = torch.randn(
        2, 4, 3, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch.ops.aten._nested_from_padded_tensor(
        dense, offsets, dummy, sum_S=5
    )
    self.assertEqual(nt.values().shape, (5, 3))

    # Compute loss on flat values
    target_weights = torch.arange(
        1, 16, dtype=torch.float32, device="tpu"
    ).reshape(5, 3)
    loss = (nt.values() * target_weights).sum()
    loss.backward()

    self.assertIsNotNone(dense.grad)
    dense_grad_cpu = dense.grad.cpu()

    # Active sequence elements receive the upstream weights
    utils.assert_close(dense_grad_cpu[0, :2, :], target_weights[:2].cpu())
    utils.assert_close(dense_grad_cpu[1, :3, :], target_weights[2:].cpu())

    # Padding positions in the dense tensor MUST receive zero gradient
    self.assertTrue(torch.all(dense_grad_cpu[0, 2:, :] == 0.0))
    self.assertTrue(torch.all(dense_grad_cpu[1, 3:, :] == 0.0))

  def test_roundtrip_jagged_to_padded_to_jagged_autograd(self):
    """Verifies identity gradient propagation through jagged -> padded -> jagged."""
    values = torch.randn(
        6, 4, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device="tpu")

    nt_in = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=3, max_seqlen=3
    )
    padded = nt_in.to_padded_tensor(0.0)
    dummy = nt_internal._nt_view_dummy()
    nt_out = torch.ops.aten._nested_from_padded_tensor(
        padded, offsets, dummy, sum_S=6
    )

    loss = (nt_out.values() * 3.5).sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 3.5))

  def test_multidimensional_features_autograd(self):
    """Verifies autograd on jagged tensors with multidimensional feature shapes."""
    # (total_L=4, feature_dim_0=3, feature_dim_1=5)
    values = torch.randn(
        4, 3, 5, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 1, 4], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=1, max_seqlen=3
    )
    padded = nt.to_padded_tensor(0.0)

    self.assertEqual(padded.shape, (2, 3, 3, 5))
    loss = padded.pow(2).sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    expected_grad = (2.0 * values).cpu()
    utils.assert_close(values.grad.cpu(), expected_grad)

  def test_autograd_retain_graph_and_accumulation(self):
    """Verifies gradient accumulation across multiple backward passes."""
    values = torch.randn(
        5, 2, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )
    padded = nt.to_padded_tensor(0.0)

    loss1 = (padded * 1.5).sum()
    loss1.backward(retain_graph=True)

    loss2 = (padded * 2.5).sum()
    loss2.backward()

    # Accumulated grad must be 1.5 + 2.5 = 4.0
    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 4.0))

  # ---------------------------------------------------------------------------
  # _nested_view_from_jagged Dedicated Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_view_from_jagged_basic_and_cpu_parity(self):
    """Tests _nested_view_from_jagged tensor construction and CPU parity."""
    values_cpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()

    nt_cpu = torch._nested_view_from_jagged(values_cpu, offsets_cpu, dummy_cpu)

    values_tpu = values_cpu.to("tpu")
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    nt_tpu = torch._nested_view_from_jagged(values_tpu, offsets_tpu, dummy_tpu)

    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)
    self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
    self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])
    self.assertEqual(nt_tpu.dtype, nt_cpu.dtype)
    utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
    utils.assert_close(nt_tpu.offsets().cpu(), nt_cpu.offsets())

  def test_nested_view_from_jagged_with_lengths(self):
    """Tests _nested_view_from_jagged with explicit lengths parameter."""
    values = torch.randn(6, 4, device="tpu")
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="tpu")
    lengths = torch.tensor([2, 4], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch._nested_view_from_jagged(values, offsets, dummy, lengths=lengths)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.shape[0], 2)
    self.assertEqual(nt.shape[2], 4)
    utils.assert_close(nt.values().cpu(), values.cpu())
    utils.assert_close(nt.offsets().cpu(), offsets.cpu())

  def test_nested_view_from_jagged_with_min_max_seqlen(self):
    """Tests _nested_view_from_jagged with min_seqlen and max_seqlen tensors."""
    values = torch.randn(7, 3, device="tpu")
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()
    min_seqlen = torch.tensor(3, device="tpu")
    max_seqlen = torch.tensor(4, device="tpu")

    nt = torch._nested_view_from_jagged(
        values,
        offsets,
        dummy,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    utils.assert_close(nt.values().cpu(), values.cpu())

  def test_nested_view_from_jagged_dtypes(self):
    """Tests _nested_view_from_jagged across multiple supported dtypes."""
    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ]:
      if dtype == torch.bool:
        values = torch.tensor([True, False, True, True, False], device="tpu")
      elif dtype in (torch.int32, torch.int64):
        values = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="tpu")
      else:
        values = torch.randn(5, 2, dtype=dtype, device="tpu")

      offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
      dummy = nt_internal._nt_view_dummy()

      nt = torch._nested_view_from_jagged(values, offsets, dummy)
      self.assertEqual(nt.dtype, dtype)
      self.assertEqual(nt.device.type, "tpu")
      self.assertEqual(nt.layout, torch.jagged)
      utils.assert_close(nt.values().cpu(), values.cpu())

  def test_nested_view_from_jagged_mutation_aliasing(self):
    """Verifies that _nested_view_from_jagged creates an aliased view."""
    values = torch.zeros(5, 2, device="tpu")
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch._nested_view_from_jagged(values, offsets, dummy)

    # In-place add on values buffer should be reflected in the view
    values.add_(5.0)
    utils.assert_close(nt.values().cpu(), torch.full((5, 2), 5.0))

  # ---------------------------------------------------------------------------
  # _nested_from_padded_tensor Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_from_padded_tensor_eager_conversion_and_cpu_parity(self):
    """Tests _nested_from_padded_tensor conversion and CPU parity."""
    # Input dense tensor shape: (B=2, max_seqlen=3, D=2)
    # Batch item 0: [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]] (padded with [0, 0] at index 2)
    # Batch item 1: [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]] (full length 3)
    dense_cpu = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    # Offsets [0, 2, 5]:
    # - Batch 0 takes length 2 - 0 = 2 tokens: dense[0, 0:2, :] -> [[1.0, 2.0], [3.0, 4.0]]
    # - Batch 1 takes length 5 - 2 = 3 tokens: dense[1, 0:3, :] -> [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    # Total extracted tokens: sum_S = 2 + 3 = 5
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()

    nt_cpu = torch.ops.aten._nested_from_padded_tensor(
        dense_cpu, offsets_cpu, dummy_cpu, sum_S=5
    )

    dense_tpu = dense_cpu.to("tpu")
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    nt_tpu = torch.ops.aten._nested_from_padded_tensor(
        dense_tpu, offsets_tpu, dummy_tpu, sum_S=5
    )

    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)
    self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
    self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])

    # Explicitly verify against manually constructed expected values buffer (5, 2)
    expected_values = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    utils.assert_close(nt_tpu.values().cpu(), expected_values)
    utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
    utils.assert_close(nt_tpu.offsets().cpu(), offsets_cpu)

  def test_nested_from_padded_tensor_with_min_max_seqlen(self):
    """Tests _nested_from_padded_tensor with min/max seqlen metadata."""
    # Input dense tensor shape: (B=2, max_seqlen=4, D=3)
    # Offsets [0, 2, 6]:
    # - Batch 0 takes length 2 - 0 = 2 tokens (dense[0, :2, :])
    # - Batch 1 takes length 6 - 2 = 4 tokens (dense[1, :4, :])
    # - Total unpadded tokens = 2 + 4 = 6, yielding a flat values shape of (6, 3).
    dense = torch.randn(2, 4, 3, device="tpu")
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()
    min_seqlen = torch.tensor(2, device="tpu")
    max_seqlen = torch.tensor(4, device="tpu")

    nt = torch.ops.aten._nested_from_padded_tensor(
        dense,
        offsets,
        dummy,
        ragged_idx=1,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        sum_S=6,
    )
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.values().shape, (6, 3))

  def test_nested_from_padded_tensor_dtypes(self):
    """Tests _nested_from_padded_tensor across multiple dtypes with CPU parity."""
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ]:
      if dtype == torch.bool:
        dense_cpu = torch.tensor(
            [
                [[True, False], [True, True], [False, False]],
                [[False, True], [True, False], [True, True]],
            ],
            dtype=dtype,
        )
      elif dtype in (torch.int32, torch.int64):
        dense_cpu = torch.tensor(
            [[[1, 2], [3, 4], [0, 0]], [[5, 6], [7, 8], [9, 10]]],
            dtype=dtype,
        )
      else:
        dense_cpu = torch.randn(2, 3, 4, dtype=dtype)

      dense_tpu = dense_cpu.to("tpu")

      nt_cpu = torch.ops.aten._nested_from_padded_tensor(
          dense_cpu, offsets_cpu, dummy_cpu, sum_S=5
      )
      nt_tpu = torch.ops.aten._nested_from_padded_tensor(
          dense_tpu, offsets_tpu, dummy_tpu, sum_S=5
      )

      self.assertEqual(nt_tpu.dtype, dtype)
      self.assertEqual(nt_tpu.device.type, "tpu")
      self.assertEqual(nt_tpu.layout, torch.jagged)
      self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
      self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])
      utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
      utils.assert_close(nt_tpu.offsets().cpu(), nt_cpu.offsets())

  # ---------------------------------------------------------------------------
  # _nested_get_* Accessor Primitives Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_get_accessors(self):
    """Tests ATen _nested_get_* accessor primitives on jagged nested tensors."""
    values = torch.randn(5, 3, device="tpu", requires_grad=True)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )

    extracted_values = torch.ops.aten._nested_get_values(nt)
    extracted_offsets = torch.ops.aten._nested_get_offsets(nt)
    extracted_min_seqlen = torch.ops.aten._nested_get_min_seqlen(nt)
    extracted_max_seqlen = torch.ops.aten._nested_get_max_seqlen(nt)
    extracted_ragged_idx = torch.ops.aten._nested_get_ragged_idx(nt)
    dummy = torch.ops.aten._nested_get_jagged_dummy(nt)

    self.assertEqual(extracted_values.device.type, "tpu")
    self.assertEqual(extracted_offsets.device.type, "tpu")
    utils.assert_close(extracted_values.cpu(), values.cpu())
    utils.assert_close(extracted_offsets.cpu(), offsets.cpu())
    if extracted_min_seqlen.numel() > 0:
      self.assertEqual(extracted_min_seqlen.item(), 2)
    if extracted_max_seqlen.numel() > 0:
      self.assertEqual(extracted_max_seqlen.item(), 3)
    self.assertEqual(extracted_ragged_idx, 1)
    self.assertIsNotNone(dummy)

    # Verify autograd flows through _nested_get_values
    loss = (extracted_values * 2.0).sum()
    loss.backward()
    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 2.0))

  # ---------------------------------------------------------------------------
  # Conversion Operators: Jagged <-> Padded Dense Tests
  # ---------------------------------------------------------------------------

  def test_jagged_to_padded_dense_forward_basic(self):
    """Tests basic jagged-to-padded conversion."""
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_jagged_to_padded(
        values, offsets, max_lengths=[3], padding_value=0.0
    )

  def test_jagged_to_padded_dense_padding_values_and_dtypes(self):
    """Tests various padding values and dtypes for jagged-to-padded conversion."""
    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.bool,
    ]:
      for pad_val in [0.0, 1.0, -1.0, float("-inf"), float("inf")]:
        if dtype == torch.bool:
          values = torch.tensor([True, False, True, True, False])
        elif dtype in (torch.int64, torch.int32):
          values = torch.tensor([10, 20, 30, 40, 50], dtype=dtype)
        else:
          values = torch.tensor([1.5, -2.5, 3.5, -4.5, 5.5], dtype=dtype)

        offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
        self._assert_jagged_to_padded(
            values, offsets, max_lengths=[4], padding_value=pad_val
        )

  def test_jagged_to_padded_dense_truncation(self):
    """Tests jagged-to-padded truncation when actual sequence length > max_length."""
    # Batch sequence lengths: [5, 2, 8, 0, 3], with max_length = 4
    offsets = torch.tensor([0, 5, 7, 15, 15, 18], dtype=torch.int64)
    values = torch.randn(18, 3)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4], padding_value=-99.0
    )
    self.assertEqual(padded_tpu.shape, (5, 4, 3))

    # Verify specific batch rows:
    # Row 0: truncated from length 5 to 4
    utils.assert_close(padded_tpu[0].cpu(), values[0:4])
    # Row 1: length 2 preserved + 2 padding positions
    utils.assert_close(padded_tpu[1, :2].cpu(), values[5:7])
    self.assertTrue(torch.all(padded_tpu[1, 2:].cpu() == -99.0))
    # Row 3: empty sequence is 100% padding
    self.assertTrue(torch.all(padded_tpu[3].cpu() == -99.0))

  def test_jagged_1d_tokens(self):
    """Tests 1D jagged values (scalar tokens with no trailing feature dimensions)."""
    values = torch.tensor([101, 2054, 2003, 1037, 3231, 102], dtype=torch.int64)
    offsets = torch.tensor([0, 3, 4, 6], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4]
    )
    self.assertEqual(padded_tpu.shape, (3, 4))

    # Convert back to 1D flat values
    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=6
    )
    self.assertEqual(recovered_tpu.shape, (6,))
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_padded_dense_to_jagged_forward_basic(self):
    """Tests basic padded-to-jagged conversion with and without total_L."""
    padded = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_padded_to_jagged(padded, offsets)
    self._assert_padded_to_jagged(padded, offsets, total_L=5)

  def test_roundtrip_conversion(self):
    """Tests round-trip conversion: jagged -> padded dense -> jagged."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(1, 4, device="tpu")
    t3 = torch.randn(5, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2, t3], layout=torch.jagged)

    values, offsets = nt.values(), nt.offsets()
    padded = torch.ops.aten._jagged_to_padded_dense_forward(
        values, [offsets], [5], 0.0
    )
    self.assertEqual(padded.shape, (3, 5, 4))

    recovered_values = torch.ops.aten._padded_dense_to_jagged_forward(
        padded, [offsets], total_L=values.size(0)
    )
    utils.assert_close(recovered_values.cpu(), values.cpu())

  def test_jagged_to_padded_dense_multidim(self):
    """Tests jagged conversion with higher-dimensional features."""
    values = torch.randn(6, 3, 5)
    offsets = torch.tensor([0, 1, 4, 6], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[3]
    )
    self.assertEqual(padded_tpu.shape, (3, 3, 3, 5))
    self._assert_padded_to_jagged(padded_tpu.cpu(), offsets, total_L=6)

  # ---------------------------------------------------------------------------
  # Edge Cases & Stress Scenarios
  # ---------------------------------------------------------------------------

  def test_all_empty_batch(self):
    """Tests entirely empty batch (offsets = [0, 0, 0, 0], total_L = 0)."""
    values = torch.empty((0, 4), dtype=torch.float32)
    offsets = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[5], padding_value=42.0
    )
    self.assertEqual(padded_tpu.shape, (3, 5, 4))
    self.assertTrue(torch.all(padded_tpu.cpu() == 42.0))

    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=0
    )
    self.assertEqual(recovered_tpu.shape, (0, 4))

  def test_zero_padded_length(self):
    """Tests max_lengths = [0] with non-empty input."""
    values = torch.randn(5, 4)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[0]
    )
    self.assertEqual(padded_tpu.shape, (2, 0, 4))

  def test_zero_batch_size(self):
    """Tests zero batch size (offsets = [0])."""
    values = torch.empty((0, 4))
    offsets = torch.tensor([0], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[5]
    )
    self.assertEqual(padded_tpu.shape, (0, 5, 4))

  def test_exact_fit_boundary(self):
    """Tests boundary case where max_length equals the longest sequence."""
    offsets = torch.tensor([0, 4, 5, 9], dtype=torch.int64)
    values = torch.randn(9, 4)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4]
    )
    self.assertEqual(padded_tpu.shape, (3, 4, 4))
    self._assert_padded_to_jagged(padded_tpu.cpu(), offsets, total_L=9)

  def test_compilation_cache_varying_offsets(self):
    """Tests compilation cache isolation when offsets change across executions."""
    values = torch.randn(6, 4)
    for offsets_list in [
        [0, 2, 4, 6],
        [0, 4, 5, 6],
        [0, 0, 0, 6],
        [0, 6, 6, 6],
        [0, 1, 3, 6],
    ]:
      offsets = torch.tensor(offsets_list, dtype=torch.int64)
      padded_tpu, _ = self._assert_jagged_to_padded(
          values, offsets, max_lengths=[6]
      )
      recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
          padded_tpu, [offsets.to("tpu")], total_L=6
      )
      utils.assert_close(recovered_tpu.cpu(), values)

  def test_special_values_and_complex(self):
    """Tests complex64, nan, and inf padding values on TPU."""
    # Complex64
    values_complex = torch.tensor(
        [
            [1.0 + 2.0j, 3.0 - 4.0j],
            [5.0 + 6.0j, -7.0 + 8.0j],
            [9.0 - 10.0j, 11.0 + 12.0j],
        ],
        dtype=torch.complex64,
    )
    offsets_tpu = torch.tensor([0, 1, 3], dtype=torch.int64, device="tpu")
    padded_complex_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_complex.to("tpu"), [offsets_tpu], [2], 0.0
    )
    self.assertEqual(padded_complex_tpu.shape, (2, 2, 2))
    utils.assert_close(padded_complex_tpu.cpu()[0, 0], values_complex[0])
    utils.assert_close(padded_complex_tpu.cpu()[1, 0], values_complex[1])
    utils.assert_close(padded_complex_tpu.cpu()[1, 1], values_complex[2])

    recovered_complex_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_complex_tpu, [offsets_tpu], total_L=3
    )
    utils.assert_close(recovered_complex_tpu.cpu(), values_complex)

    # NaN / Inf padding
    offsets_tpu = torch.tensor([0, 1, 3], dtype=torch.int64, device="tpu")
    values_float = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="tpu"
    )
    padded_nan = torch.ops.aten._jagged_to_padded_dense_forward(
        values_float, [offsets_tpu], [2], float("nan")
    )
    self.assertTrue(torch.isnan(padded_nan[0, 1, 0]).item())
    self.assertFalse(torch.isnan(padded_nan[0, 0, 0]).item())

    padded_inf = torch.ops.aten._jagged_to_padded_dense_forward(
        values_float, [offsets_tpu], [2], float("inf")
    )
    self.assertTrue(torch.isinf(padded_inf[0, 1, 0]).item())

  def test_nan_inf_token_payload_preservation(self):
    """Tests that NaNs and Infs inside valid tokens are preserved during dense-to-jagged."""
    dense = torch.tensor(
        [
            [[float("nan"), 1.0], [float("inf"), float("-inf")], [0.0, 0.0]],
            [[2.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        device="tpu",
    )
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64, device="tpu")

    jagged_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense, [offsets], total_L=3
    )
    self.assertTrue(torch.isnan(jagged_tpu[0, 0]).item())
    self.assertEqual(jagged_tpu[0, 1].item(), 1.0)
    self.assertTrue(torch.isposinf(jagged_tpu[1, 0]).item())
    self.assertTrue(torch.isneginf(jagged_tpu[1, 1]).item())
    self.assertEqual(jagged_tpu[2, 0].item(), 2.0)
    self.assertEqual(jagged_tpu[2, 1].item(), 3.0)

  def test_heterogeneous_device_offsets(self):
    """Tests conversion when values is on TPU but offsets is on CPU."""
    values = torch.randn(5, 4)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    padded_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values.to("tpu"), [offsets], [3], 0.0
    )
    self.assertEqual(padded_tpu.device.type, "tpu")
    self.assertEqual(padded_tpu.shape, (2, 3, 4))
    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets], total_L=5
    )
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_non_contiguous_inputs(self):
    """Tests non-contiguous strided dense and offsets inputs."""
    # Transposed dense input
    dense = torch.randn(2, 3, 4, 5)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_padded_to_jagged(dense.transpose(2, 3), offsets)

    # Strided offsets input
    full_offsets = torch.tensor([0, 999, 2, 999, 5], dtype=torch.int64)
    strided_offsets = full_offsets[::2]
    self._assert_jagged_to_padded(
        torch.randn(5, 4), strided_offsets, max_lengths=[3]
    )

  def test_sparse_and_asymmetric_sequences(self):
    """Tests heavily sparse batch with huge length disparity."""
    offsets = torch.tensor([0, 0, 1, 1, 129, 129, 132], dtype=torch.int64)
    values = torch.randn(132, 8)
    self._assert_jagged_to_padded(
        values, offsets, max_lengths=[64], padding_value=-1.0
    )

  def test_large_batch_stress(self):
    """Stress test with large batch size and uneven sequence lengths."""
    batch_size = 64
    lengths = torch.randint(0, 32, (batch_size,), dtype=torch.int64)
    offsets = torch.cat(
        [torch.tensor([0], dtype=torch.int64), torch.cumsum(lengths, dim=0)]
    )
    total_l = offsets[-1].item()

    values = torch.randn(total_l, 16)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[32], padding_value=-1.0
    )
    self.assertEqual(padded_tpu.shape, (batch_size, 32, 16))

    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=total_l
    )
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_end_to_end_model_pipeline(self):
    """Tests end-to-end forward pipeline combining linear, conversion, and dense layers."""
    t1 = torch.randn(5, 8, device="tpu")
    t2 = torch.randn(3, 8, device="tpu")
    t3 = torch.randn(7, 8, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2, t3], layout=torch.jagged)

    # 1. Pointwise linear projection on jagged tensor
    linear = torch.nn.Linear(8, 16).to("tpu")
    projected_nt = linear(nt)

    # 2. Convert to padded dense for attention / layer norm
    padded = torch.ops.aten._jagged_to_padded_dense_forward(
        projected_nt.values(), [projected_nt.offsets()], [8], 0.0
    )
    self.assertEqual(padded.shape, (3, 8, 16))

    # 3. Dense operation on padded tensor
    layernorm = torch.nn.LayerNorm(16).to("tpu")
    normed = layernorm(padded)

    # 4. Convert back to jagged
    recovered_values = torch.ops.aten._padded_dense_to_jagged_forward(
        normed, [projected_nt.offsets()], total_L=projected_nt.values().size(0)
    )
    self.assertEqual(recovered_values.shape, (15, 16))

  def test_invalid_inputs_error_handling(self):
    """Tests error checking on invalid offsets, dimensions, and parameters."""
    values = torch.randn(5, 4, device="tpu")
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
    dense = torch.randn(2, 3, 4, device="tpu")

    # Negative max_length check
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [-1], 0.0
      )

    # Multiple jagged dimensions rejected
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets, offsets], [3], 0.0
      )

    # 2D offsets tensor rejected
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets.unsqueeze(0)], [3], 0.0
      )

    # 1D dense tensor in padded_dense_to_jagged rejected
    with self.assertRaises(RuntimeError):
      torch.ops.aten._padded_dense_to_jagged_forward(
          torch.randn(5, device="tpu"), [offsets]
      )

    # total_L mismatch rejected
    with self.assertRaises(RuntimeError):
      torch.ops.aten._padded_dense_to_jagged_forward(
          dense, [offsets], total_L=999
      )

    # Offset sequence length exceeding dense sequence length rejected
    invalid_offsets = torch.tensor([0, 10, 20], dtype=torch.int64, device="tpu")
    with self.assertRaises(RuntimeError):
      torch.ops.aten._padded_dense_to_jagged_forward(dense, [invalid_offsets])

    # Non-int64 offsets rejected
    int32_offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device="tpu")
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [int32_offsets], [3], 0.0
      )

    # Non-zero start offsets rejected
    nonzero_start_offsets = torch.tensor(
        [1, 2, 5], dtype=torch.int64, device="tpu"
    )
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [nonzero_start_offsets], [3], 0.0
      )

    # Non-monotonic offsets rejected
    decreasing_offsets = torch.tensor(
        [0, 5, 2], dtype=torch.int64, device="tpu"
    )
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [decreasing_offsets], [3], 0.0
      )

    # 0D values tensor rejected
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          torch.tensor(1.0, device="tpu"), [offsets], [3], 0.0
      )

    # Offsets referencing more elements than available in values rejected
    out_of_bounds_offsets = torch.tensor(
        [0, 10], dtype=torch.int64, device="tpu"
    )
    with self.assertRaises(RuntimeError):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [out_of_bounds_offsets], [10], 0.0
      )

    # Offsets batch size mismatching dense batch size rejected
    mismatched_batch_offsets = torch.tensor(
        [0, 1, 2, 3], dtype=torch.int64, device="tpu"
    )
    with self.assertRaises(RuntimeError):
      torch.ops.aten._padded_dense_to_jagged_forward(
          dense, [mismatched_batch_offsets]
      )


if __name__ == "__main__":
  absltest.main()
