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

"""Unit tests for SymShapeManager."""

from __future__ import annotations
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import sym_shape_manager
from tests import seed_test_utils


class SymShapeManagerTest(seed_test_utils.RepeatableTest):

  def test_symbol_manager_extraction(self):
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertEqual(captured_sm.get_num_dynamic_dims(1), 1)
    self.assertEqual(captured_sm.input_tensors_metadata[1].static_shape, [-1])
    self.assertEqual(captured_sm.get_bounded_shape(1), [8])

  def test_symbol_manager_static_shape(self):
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def f(x):
      return x + 1

    t = torch.ones(2, 3)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertEqual(captured_sm.get_num_dynamic_dims(0), 0)
    self.assertEqual(captured_sm.input_tensors_metadata[0].static_shape, [2, 3])
    self.assertEqual(captured_sm.get_bounded_shape(0), [2, 3])


if __name__ == "__main__":
  absltest.main()
