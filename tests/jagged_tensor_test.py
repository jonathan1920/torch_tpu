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

"""Unit tests for PyTorch torch.jagged layout and jagged tensors on TPU."""

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class JaggedTensorTest(seed_test_utils.RepeatableTest):

  def test_empty_jagged(self):
    """Tests creating an empty tensor with layout=torch.jagged on TPU.

    Calling torch.empty with layout=torch.jagged allocates the underlying flat
    strided buffer that PyTorch uses to back jagged nested tensors.
    """
    t = torch.empty(5, layout=torch.jagged, device="tpu")
    self.assertEqual(t.device.type, "tpu")
    self.assertEqual(t.shape, (5,))
    # The raw allocated buffer is a flat strided tensor before being wrapped
    # into a NestedTensor.
    self.assertEqual(t.layout, torch.strided)

  def test_nested_tensor_construction(self):
    """Tests constructing a nested tensor with layout=torch.jagged on TPU."""
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="tpu")
    t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.shape[0], 2)  # batch size
    self.assertEqual(nt.shape[2], 2)  # feature dim

    # Check values and offsets on TPU
    values = nt.values()
    offsets = nt.offsets()
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
    """Tests that pointwise operations on jagged tensors on TPU match CPU."""
    t1_cpu = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    t2_cpu = torch.tensor([[5.0, 6.0], [-7.0, 8.0], [9.0, -10.0]])
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)

    t1_tpu = t1_cpu.to("tpu")
    t2_tpu = t2_cpu.to("tpu")
    nt_tpu = torch.nested.nested_tensor([t1_tpu, t2_tpu], layout=torch.jagged)

    # Addition with scalar
    out_cpu = nt_cpu + 2.5
    out_tpu = nt_tpu + 2.5
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

    # Multiplication with scalar
    out_cpu = nt_cpu * 1.5
    out_tpu = nt_tpu * 1.5
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

    # Pointwise ReLU
    out_cpu = torch.relu(nt_cpu)
    out_tpu = torch.relu(nt_tpu)
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

    # Pointwise Sin
    out_cpu = torch.sin(nt_cpu)
    out_tpu = torch.sin(nt_tpu)
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

    # Binary addition between two jagged tensors
    out_cpu = nt_cpu + nt_cpu
    out_tpu = nt_tpu + nt_tpu
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

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

    nt_tpu = nt_cpu.to("tpu")
    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)

    nt_back_to_cpu = nt_tpu.to("cpu")
    self.assertEqual(nt_back_to_cpu.device.type, "cpu")
    self.assertEqual(nt_back_to_cpu.layout, torch.jagged)
    utils.assert_close(nt_back_to_cpu.values(), nt_cpu.values())
    utils.assert_close(nt_back_to_cpu.offsets(), nt_cpu.offsets())


if __name__ == "__main__":
  absltest.main()
