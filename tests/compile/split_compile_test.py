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

import unittest.mock
from absl.testing import absltest
import torch
from torch_tpu._internal.compile import collective_ops
from torch_tpu._internal.compile._backend import TpuBackend
from torch_tpu._internal.utils import utils


class SplitCompileTest(absltest.TestCase):

  def test_split_graph_with_dynamic_tensor(self):
    backend = TpuBackend(debug=True, dynamism=True)

    def f(x):
      y = x * 2
      z = y + 3
      return z + 4

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul' (x * 2)
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(f, backend=backend)

      x = torch.ones((2, 2), device="tpu")
      torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=8)

      res = compiled_f(x)

    expected = torch.full((2, 2), 9.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_split_graph_with_direct_symint_usage(self):
    backend = TpuBackend(debug=True, dynamism=True)

    def f(x):
      s1 = x.shape[0]
      s2 = x.shape[1]
      y = x * 2
      z = y.reshape(s1, 1, s2, 1)
      return z + 3

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul' (x * 2)
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(f, backend=backend)

      x = torch.ones((8, 6), device="tpu")
      torch._dynamo.mark_dynamic(x, 0, min=2, max=16)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=16)

      res = compiled_f(x)

    expected = torch.full((8, 1, 6, 1), 5.0, device="cpu")
    utils.assert_close(res.cpu(), expected)


if __name__ == "__main__":
  absltest.main()
