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

from absl.testing import absltest
import torch
from torch_tpu._internal.testing import get_memory_kind
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

try:
  # pylint: disable=g-import-not-at-top
  from torch._functorch._activation_offloading import offload_ops

  _HAS_AO_OPS = offload_ops is not None
except (ImportError, AttributeError):
  _HAS_AO_OPS = False


@absltest.skipIf(
    not _HAS_AO_OPS,
    "`torch._functorch._activation_offloading.offload_ops` is not available in "
    "this PyTorch version",
)
class AoOffloadTest(seed_test_utils.RepeatableTest):

  def test_offload(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    self.assertEqual(get_memory_kind(x), "device")

    y = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(y), "pinned_host")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)

  def test_reload(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32).flatten()
    offloaded = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    y = torch.ops.ao.reload(
        offloaded,
        torch.device("tpu"),
        [2, 3],  # original_size
        [3, 1],  # original_stride
    )
    self.assertEqual(list(y.shape), [2, 3])
    self.assertEqual(list(y.stride()), [3, 1])
    self.assertEqual(get_memory_kind(y), "device")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)

  def test_wait_tensor(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    self.assertEqual(get_memory_kind(x), "device")

    y = torch.ops.ao.wait_tensor(x)
    self.assertEqual(x.data_ptr(), y.data_ptr())
    self.assertEqual(get_memory_kind(y), "device")

    y_kwargs = torch.ops.ao.wait_tensor(x, keepalive=x, last_use_of_storage=x)
    self.assertEqual(x.data_ptr(), y_kwargs.data_ptr())
    self.assertEqual(get_memory_kind(y_kwargs), "device")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)
    test_utils.assert_close(y_kwargs.cpu(), expected)

  def test_tensor_with_view(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    offloaded = torch.ops.ao.offload(x)
    offloaded_view = offloaded.view(6)

    self.assertEqual(get_memory_kind(offloaded_view), "pinned_host")

    y = torch.ops.ao.reload(
        offloaded_view,
        torch.device("tpu"),
        [2, 3],  # original_size
        [3, 1],  # original_stride
    )
    self.assertEqual(list(y.shape), [2, 3])
    self.assertEqual(list(y.stride()), [3, 1])
    self.assertEqual(get_memory_kind(y), "device")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)

  def test_offload_non_contiguous(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32).t()
    self.assertFalse(x.is_contiguous())

    y = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(y), "pinned_host")
    expected = torch.tensor(data, dtype=torch.float32).t()
    test_utils.assert_close(y.cpu(), expected)

  def test_reload_non_contiguous(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32).t()
    self.assertFalse(x.is_contiguous())
    offloaded = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    y = torch.ops.ao.reload(
        offloaded,
        torch.device("tpu"),
        [3, 2],  # original_size
        [1, 3],  # non-contiguous transposed stride
    )
    self.assertEqual(list(y.shape), [3, 2])
    self.assertEqual(list(y.stride()), [1, 3])
    self.assertFalse(y.is_contiguous())
    self.assertEqual(get_memory_kind(y), "device")
    test_utils.assert_close(y.cpu(), x.cpu())

  def test_offload_scalar(self):
    x = torch.tensor(3.14, device="tpu", dtype=torch.float32)
    y = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(y), "pinned_host")

    reloaded = torch.ops.ao.reload(y, torch.device("tpu"))
    self.assertEqual(get_memory_kind(reloaded), "device")
    test_utils.assert_close(reloaded.cpu(), torch.tensor(3.14))

  def test_offload_zero_element(self):
    x = torch.empty((2, 0, 3), device="tpu", dtype=torch.float32)
    y = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(y), "pinned_host")

    reloaded = torch.ops.ao.reload(y, torch.device("tpu"))
    self.assertEqual(get_memory_kind(reloaded), "device")
    self.assertEqual(list(reloaded.shape), [2, 0, 3])

  def test_compile_offload_and_reload(self):
    @torch.compile
    def fwd(t):
      return torch.ops.ao.offload(t)

    @torch.compile
    def bwd(offloaded):
      reloaded = torch.ops.ao.reload(offloaded, torch.device("tpu"))
      return reloaded + 1.0

    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)

    offloaded = fwd(x)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    out = bwd(offloaded)
    self.assertEqual(get_memory_kind(out), "device")
    expected = torch.tensor(data, dtype=torch.float32) + 1.0
    test_utils.assert_close(out.cpu(), expected)

  def test_compile_offload_and_reload_with_views(self):
    @torch.compile
    def fwd(t):
      v = t.view(1, 6).squeeze(0)
      offloaded = torch.ops.ao.offload(v)
      return offloaded.view(3, 2).unsqueeze(0)

    @torch.compile
    def bwd(offloaded):
      v = offloaded.squeeze(0).view(1, 6).squeeze(0)
      reloaded = torch.ops.ao.reload(v, torch.device("tpu"))
      return reloaded.view(2, 3) + 1.0

    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)

    offloaded = fwd(x)
    self.assertEqual(list(offloaded.shape), [1, 3, 2])
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    out = bwd(offloaded)
    self.assertEqual(list(out.shape), [2, 3])
    self.assertEqual(get_memory_kind(out), "device")
    expected = torch.tensor(data, dtype=torch.float32) + 1.0
    test_utils.assert_close(out.cpu(), expected)

  def test_reload_on_device_tensor(self):
    data = [1.0, 2.0, 3.0]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    self.assertEqual(get_memory_kind(x), "device")

    y = torch.ops.ao.reload(x, torch.device("tpu"))
    self.assertEqual(get_memory_kind(x), "device")
    self.assertEqual(get_memory_kind(y), "device")
    test_utils.assert_close(y.cpu(), torch.tensor(data, dtype=torch.float32))

  def test_offload_on_pinned_host_tensor(self):
    data = [1.0, 2.0, 3.0]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    offloaded1 = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(offloaded1), "pinned_host")

    offloaded2 = torch.ops.ao.offload(offloaded1)
    self.assertEqual(get_memory_kind(offloaded2), "pinned_host")
    test_utils.assert_close(
        offloaded2.cpu(), torch.tensor(data, dtype=torch.float32)
    )

  def test_compile_sibling_views_before_reload(self):
    @torch.compile
    def fn(offloaded):
      v1 = offloaded.view(1, 6)
      v2 = offloaded.view(2, 3)
      reloaded = torch.ops.ao.reload(v1, torch.device("tpu"))
      return reloaded, v2

    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)

    offloaded = torch.ops.ao.offload(x)
    reloaded, v2 = fn(offloaded)
    self.assertEqual(get_memory_kind(reloaded), "device")
    self.assertEqual(get_memory_kind(v2), "pinned_host")
    test_utils.assert_close(
        reloaded.cpu(), torch.tensor(data, dtype=torch.float32).view(1, 6)
    )
    test_utils.assert_close(v2.cpu(), torch.tensor(data, dtype=torch.float32))

  def test_offload_sliced_tensor_with_storage_offset(self):
    data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    sliced = x[2:]
    self.assertEqual(sliced.storage_offset(), 2)

    offloaded = torch.ops.ao.offload(sliced)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    reloaded = torch.ops.ao.reload(
        offloaded,
        torch.device("tpu"),
        [4],
        [1],
    )
    self.assertEqual(get_memory_kind(reloaded), "device")
    test_utils.assert_close(
        reloaded.cpu(), torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    )

  def test_compile_unused_offload(self):
    @torch.compile
    def fn(t):
      _ = torch.ops.ao.offload(t)
      return t * 2.0

    data = [[1.0, 2.0], [3.0, 4.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    y = fn(x)
    self.assertEqual(get_memory_kind(y), "device")
    expected = torch.tensor(data, dtype=torch.float32) * 2.0
    test_utils.assert_close(y.cpu(), expected)

  def test_compile_offloaded_identity(self):
    @torch.compile
    def fn(offloaded):
      return offloaded

    data = [[1.0, 2.0], [3.0, 4.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    offloaded = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")
    y = fn(offloaded)
    self.assertEqual(get_memory_kind(y), "pinned_host")
    test_utils.assert_close(y.cpu(), torch.tensor(data, dtype=torch.float32))

  def test_compile_mixed_device_and_offloaded_inputs(self):
    @torch.compile
    def fn(dev_a, offloaded_b, dev_c):
      reloaded_b = torch.ops.ao.reload(offloaded_b, torch.device("tpu"))
      return dev_a + reloaded_b + dev_c

    data_a = [[1.0, 2.0], [3.0, 4.0]]
    data_b = [[10.0, 20.0], [30.0, 40.0]]
    data_c = [[100.0, 200.0], [300.0, 400.0]]

    a = torch.tensor(data_a, device="tpu", dtype=torch.float32)
    b = torch.ops.ao.offload(
        torch.tensor(data_b, device="tpu", dtype=torch.float32)
    )
    c = torch.tensor(data_c, device="tpu", dtype=torch.float32)

    self.assertEqual(get_memory_kind(a), "device")
    self.assertEqual(get_memory_kind(b), "pinned_host")
    self.assertEqual(get_memory_kind(c), "device")

    y = fn(a, b, c)
    expected = torch.tensor([[111.0, 222.0], [333.0, 444.0]])
    test_utils.assert_close(y.cpu(), expected)

  def test_compile_autograd_offload_and_reload(self):
    class OffloadFunction(torch.autograd.Function):

      @staticmethod
      def forward(ctx, x):
        offloaded = torch.ops.ao.offload(x)
        ctx.save_for_backward(offloaded)
        return x * x

      @staticmethod
      def backward(ctx, grad_output):
        (offloaded,) = ctx.saved_tensors
        reloaded = torch.ops.ao.reload(offloaded, torch.device("tpu"))
        return 2.0 * reloaded * grad_output

    @torch.compile
    def fn(x):
      return OffloadFunction.apply(x)

    data = [[1.0, 2.0], [3.0, 4.0]]
    x = torch.tensor(
        data, device="tpu", dtype=torch.float32, requires_grad=True
    )

    out = fn(x)
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x.grad)
    # For f(x) = x^2, df/dx = 2*x; grad_output is 1.0 from loss = out.sum().
    expected_grad = torch.tensor(data, dtype=torch.float32) * 2.0
    test_utils.assert_close(x.grad.cpu(), expected_grad)


if __name__ == "__main__":
  absltest.main()
