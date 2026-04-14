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

"""Graph transformation passes for handling dynamic shapes.

This file contains the following passes:
- HandleDynamicInputTensorPass: Handles input tensors with dynamic dimensions.
- HandleGenerativeOpsPass: Handles generative ops (e.g., torch.arange) that
  have dynamic scalar inputs.
"""

from __future__ import annotations
from typing import Any
import torch
from torch.fx.passes import graph_transform_observer
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager
from torch_tpu._internal.compile.dynamic.symbol_bounds import get_symint_bounds

GraphTransformObserver = graph_transform_observer.GraphTransformObserver


class HandleDynamicInputTensorPass:
  """Dynamic input tensor transformation pass.

  This pass does the following:
  - Identifies input tensors with dynamic dimensions (tensors with SymInt in
    their shape).
  - Inserts a new placeholder for each dynamic dimension to represent the
    runtime size of the dynamic dimension.
  - Replaces all usage of the input tensor with the output of a chain of
    `set_dimension_logical_size` ops, each operating on the output of the
    previous one, and taking the new placeholder as the size input.
  - Modifies the graph in-place.

  Note: set_dimension_logical_size lowers down to
    stablehlo.set_dimension_size.

  Example:
    Model:
      def forward(x: torch.Tensor):
          z = x + 10
          return z

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s27)", arg1_1: "i64[1, s27, s53]"):
          add: "f32[1, s27, s53]" = torch.ops.aten.add.Tensor(arg1_1, 10)
          return (add,)

    Modified FX Graph:
      def forward(
          self,
          arg0_1: "Sym(s27)",
          arg1_1: "i64[1, s27, s53]",
          dyn_size_1_dim1: "i64[]",
          dyn_size_1_dim2: "i64[]",
      ):
        set_dimension_logical_size_1: "i64[1, s27, s53]" =
            torch.ops.torch_tpu.set_dimension_logical_size(
                arg1_1, 1, dyn_size_1_dim1)

        set_dimension_logical_size_2: "i64[1, s27, s53]" =
            torch.ops.torch_tpu.set_dimension_logical_size(
                set_dimension_logical_size_1, 2, dyn_size_1_dim2)

        add: "f32[1, s27, s53]" = torch.ops.aten.add.Tensor(
            set_dimension_logical_size_2, 10)

        return (add,)
  """

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ) -> None:
    """Runs the op insertion pass."""

    for idx, node in enumerate(placeholders):
      tensor_metadata = sym_shape_manager.input_tensors_metadata.get(idx)
      if tensor_metadata is None:
        continue

      self._process_node(graph_module, node, idx, tensor_metadata.dynamic_dims)

  def _process_node(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      idx: int,
      dynamic_dims: list[int],
  ) -> None:
    """Processes a single placeholder node and inserts set_dimension_size chains."""
    original_users = set(node.users.keys())
    current_tensor_node = node

    for dim in dynamic_dims:
      size_ph = self._insert_size_placeholder(
          graph_module, current_tensor_node, idx, dim
      )
      set_dim_size_node = self._insert_set_dimension_logical_size_node(
          graph_module, current_tensor_node, size_ph, dim, node.meta
      )
      current_tensor_node = set_dim_size_node

    node.replace_all_uses_with(
        current_tensor_node, delete_user_cb=lambda u: u in original_users
    )

  def _insert_size_placeholder(
      self,
      graph_module: torch.fx.GraphModule,
      current_tensor_node: torch.fx.Node,
      idx: int,
      dim: int,
  ) -> torch.fx.Node:
    """Inserts a size placeholder after the current tensor node."""
    with graph_module.graph.inserting_after(current_tensor_node):
      size_ph = graph_module.graph.placeholder(f"dyn_size_{idx}_dim_{dim}")
      size_ph.meta = {
          "tensor_meta": TensorMetadata(
              shape=torch.Size([]),
              dtype=torch.int32,
              requires_grad=False,
              stride=(),
              memory_format=torch.contiguous_format,
              is_quantized=False,
              qparams={},
          ),
      }
    return size_ph

  def _insert_set_dimension_logical_size_node(
      self,
      graph_module: torch.fx.GraphModule,
      current_tensor_node: torch.fx.Node,
      size_ph: torch.fx.Node,
      dim: int,
      meta: dict[str, Any],
  ) -> torch.fx.Node:
    """Inserts a set_dimension_logical_size node after the size placeholder."""
    with graph_module.graph.inserting_after(size_ph):
      set_dim_size_node = graph_module.graph.call_function(
          torch.ops.torch_tpu.set_dimension_logical_size,
          args=(current_tensor_node, dim, size_ph),
      )
      set_dim_size_node.meta = meta.copy()
    return set_dim_size_node


class HandleGenerativeOpsPass:
  """Generative ops transformation pass.

  Generative operations (like torch.arange, torch.zeros, etc.) construct new
  tensors without taking a tensor as input. When their shape or size arguments
  are dynamic (represented by SymInts), they cannot be statically compiled with
  fixed shapes. This pass handles them by replacing the dynamic scalar inputs
  with their static upper bounds to allow compilation, and then dynamically
  adjusting output tensor size at runtime using `set_dimension_logical_size`.

  This pass does the following:
  - Identifies generative ops with dynamic scalar inputs (SymInts),
  - Replaces the dynamic scalar inputs with their static upper bounds.
  - Adds a new placeholder for the runtime size of the dynamic scalar input.
  - Adds a `set_dimension_logical_size` op on the output of the generative op
    to adjust the output tensor size at runtime.
  - Replaces all usage of the generative op's output tensor in the graph with
    the output of the `set_dimension_logical_size` operation.
  - Modifies the graph in-place.

  Note: set_dimension_logical_size lowers down to
    stablehlo.set_dimension_size.

  Example:
    Model:
      def forward(x: torch.Tensor):
          z = torch.arange(0, x.shape[0])
          return z

    Test case: x = torch.tensor([8])

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s77)"):
          arange: "i64[s77]" = torch.ops.aten.arange.start(0, arg0_1)
          return (arange,)

    Modified FX Graph:
      def forward(
          self, arg0_1: "Sym(s77)", dyn_size_0: "i32[]"
      ):
        # Upper bound = 2 * arg0_1 = 2 * 8 = 16
        arange: "i64[s77]" = torch.ops.aten.arange.start(0, 16)
        set_dimension_logical_size: "i64[s77]" =
            torch.ops.torch_tpu.set_dimension_logical_size(
                arange, 0, dyn_size_0)
        return (set_dimension_logical_size,)
  """

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ) -> None:
    """Runs the op insertion pass."""

    nodes = list(graph_module.graph.nodes)
    for node in nodes:
      if node.op == "call_function" and node.target in [
          torch.ops.aten.arange.default,
          torch.ops.aten.arange.start,
      ]:
        self._process_generative_op(
            graph_module, node, sym_shape_manager, placeholders
        )

  def _process_generative_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ) -> None:
    """Processes generative op node."""
    new_args = list(node.args)
    modified_args = False
    last_sym_int = None

    for index, arg in enumerate(new_args):
      if (
          isinstance(arg, torch.fx.Node)
          and "val" in arg.meta
          and isinstance(arg.meta["val"], torch.SymInt)
      ):
        sym_int = arg.meta["val"]
        lower, upper = get_symint_bounds(sym_int)
        if lower != upper:
          new_args[index] = upper
          modified_args = True
          last_sym_int = sym_int

    if modified_args and last_sym_int is not None:
      node.args = tuple(new_args)

      sym_str = str(last_sym_int)
      size_ph = sym_shape_manager.symint_to_placeholder.get(sym_str)
      if size_ph is None:
        size_ph = self._create_tensor_placeholder(
            graph_module, sym_str, sym_shape_manager, placeholders
        )
        sym_shape_manager.symint_to_placeholder[sym_str] = size_ph

      set_dim_size_node = self._insert_set_dimension_logical_size(
          graph_module, node, size_ph
      )

      node.replace_all_uses_with(
          set_dim_size_node, delete_user_cb=lambda u: u != set_dim_size_node
      )

  def _create_tensor_placeholder(
      self,
      graph_module: torch.fx.GraphModule,
      sym_str: str,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ) -> torch.fx.Node:
    """Creates a tensor placeholder for the given SymInt."""
    idx = -1
    for i, arg in enumerate(sym_shape_manager.example_inputs):
      if isinstance(arg, torch.SymInt) and str(arg) == sym_str:
        idx = i
        break

    assert idx != -1, f"Could not find SymInt {sym_str} in example inputs"
    assert idx < len(placeholders), f"Index {idx} out of range for placeholders"

    symint_placeholder = placeholders[idx]
    size_ph = self._insert_size_placeholder(
        graph_module, symint_placeholder, idx
    )
    return size_ph

  def _insert_size_placeholder(
      self,
      graph_module: torch.fx.GraphModule,
      symint_placeholder: torch.fx.Node,
      idx: int,
  ) -> torch.fx.Node:
    """Inserts a size placeholder node after the current symint placeholder node."""
    with graph_module.graph.inserting_after(symint_placeholder):
      size_ph = graph_module.graph.placeholder(f"dyn_size_{idx}")
      size_ph.meta = {
          "tensor_meta": TensorMetadata(
              shape=torch.Size([]),
              dtype=torch.int32,
              requires_grad=False,
              stride=(),
              memory_format=torch.contiguous_format,
              is_quantized=False,
              qparams={},
          ),
      }
    return size_ph

  def _insert_set_dimension_logical_size(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      tensor_size_node: torch.fx.Node,
  ) -> torch.fx.Node:
    """Inserts a set_dimension_logical_size node after the current node."""
    with graph_module.graph.inserting_after(node):
      set_dim_size_node = graph_module.graph.call_function(
          torch.ops.torch_tpu.set_dimension_logical_size,
          args=(node, 0, tensor_size_node),
      )
      set_dim_size_node.meta = node.meta.copy()
    return set_dim_size_node


def apply_dynamism_transformations(
    graph_module: torch.fx.GraphModule, sym_shape_manager: SymShapeManager
) -> None:
  """Runs all FX graph transforms for dynamic shapes."""

  # Fetch original placeholders once in argument order
  original_placeholders = list(
      graph_module.graph.find_nodes(op="placeholder", sort=True)
  )

  # Updates ops that have input tensors with dynamic dimensions.
  GraphTransformObserver(
      graph_module, "handle_dynamic_inputs"
  ).apply_graph_pass(
      lambda _: HandleDynamicInputTensorPass()(
          graph_module, sym_shape_manager, original_placeholders
      )
  )

  # Updates the generative ops that have dynamic scalar inputs.
  GraphTransformObserver(
      graph_module, "handle_generative_ops"
  ).apply_graph_pass(
      lambda _: HandleGenerativeOpsPass()(
          graph_module, sym_shape_manager, original_placeholders
      )
  )

  graph_module.recompile()
  graph_module.graph.lint()
