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

"""Unit tests for graph transformations."""

from __future__ import annotations
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import graph_transformations
from torch_tpu._internal.compile.dynamic import sym_shape_manager


class GraphTransformationsTest(absltest.TestCase):

  def test_handle_dynamic_input_tensor_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
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

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    # Apply the pass
    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    graph_transformations.HandleDynamicInputTensorPass()(
        captured_gm, captured_sm, placeholders
    )

    # Verify that a new placeholder was added for the dynamic dimension
    new_placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    self.assertLen(new_placeholders, len(placeholders) + 1)

    # Verify that set_dimension_logical_size was inserted
    set_dim_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.torch_tpu.set_dimension_logical_size,
        )
    )
    self.assertLen(set_dim_nodes, 1)

  def test_handle_generative_ops_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return torch.arange(0, x.shape[0])

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    # Apply the pass
    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )

    graph_transformations.HandleGenerativeOpsPass()(
        captured_gm, captured_sm, placeholders
    )

    # Verify that set_dimension_logical_size was inserted for arange
    set_dim_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.torch_tpu.set_dimension_logical_size,
        )
    )
    self.assertLen(set_dim_nodes, 1)

    # Verify that arange now takes the static upper bound as input.
    arange_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.arange.start,
        )
    )
    self.assertLen(arange_nodes, 1)
    arange_node = arange_nodes[0]

    # The original args were (0, arg0_1). After the pass, the second arg
    # should be the upper bound, which is 8.
    self.assertEqual(arange_node.args[0], 0)
    self.assertEqual(arange_node.args[1], 8)


if __name__ == "__main__":
  absltest.main()
