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

"""View operations transformation pass for handling dynamic shapes."""

from __future__ import annotations
import torch
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager


class HandleViewOpsPass:
  """View ops transformation pass.

  Detects reshape-like view operations (such as aten.view, aten.reshape,
  aten._unsafe_view) that have dynamic dimensions (SymInt or SymInt expressions)
  as input and replaces them with torch.ops.tpu.dynamic_reshape.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.view.default: self._process_view_op,
        torch.ops.aten._unsafe_view.default: self._process_view_op,
        torch.ops.aten.reshape.default: self._process_view_op,
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the view ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_view_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes view/reshape op node and converts to dynamic_reshape if dynamic."""
    inp = node.args[0]
    shape = node.args[1]
    if not isinstance(shape, (list, tuple)):
      shape = [shape]

    target_shape = (
        node.meta["val"].shape
        if "val" in node.meta and hasattr(node.meta["val"], "shape")
        else shape
    )

    num_dynamic_dims = sum(
        1
        for arg in target_shape
        if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt)
    )
    if num_dynamic_dims == 0:
      return

    shape_tensors = []
    static_shape = []
    is_dynamic = []

    for arg in target_shape:
      if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt):
        is_dynamic.append(True)
        symint = arg.meta["val"] if sym_utils.is_symint_node(arg) else arg
        _, upper = symbol_bounds.get_symint_bounds(symint)
        static_shape.append(upper)
        tensor_node = self._sym_shape_manager.ensure_tensor(
            graph_module, arg, node, dtype=torch.int32
        )
        shape_tensors.append(tensor_node)
      else:
        is_dynamic.append(False)
        val = (
            arg.meta["val"]
            if isinstance(arg, torch.fx.Node) and "val" in arg.meta
            else arg
        )
        assert isinstance(
            val, int
        ), f"Expected int for static shape arg, got {val}"
        static_shape.append(val)
        tensor_node = self._sym_shape_manager.ensure_tensor(
            graph_module, arg, node, dtype=torch.int32
        )
        shape_tensors.append(tensor_node)

    with graph_module.graph.inserting_after(node):
      dynamic_reshape_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_reshape,
          args=(inp, shape_tensors, static_shape, is_dynamic),
      )
      dynamic_reshape_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_reshape_node)
    graph_module.graph.erase_node(node)
