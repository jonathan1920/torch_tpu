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
from collections.abc import Sequence
from typing import Any
import torch
from torch.fx.passes import graph_transform_observer
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager


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

  def __init__(
      self,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ):
    self._sym_shape_manager = sym_shape_manager
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""

    for idx, node in enumerate(self._placeholders):
      tensor_metadata = self._sym_shape_manager.input_tensors_metadata.get(idx)
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


class ScanInputsCreatePlaceholdersPass:
  """Pass to scan inputs and create placeholders for SymInt inputs.

  This pass scans the inputs and if an input is a SymInt, it creates a new
  placeholder for it.
  """

  def __init__(
      self,
      sym_shape_manager: SymShapeManager,
      placeholders: Sequence[torch.fx.Node],
  ):
    self._sym_shape_manager = sym_shape_manager
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the create placeholders pass."""
    for node in self._placeholders:
      if sym_utils.is_symint_node(node):
        sym_str = str(node.meta["val"])
        # Create a new placeholder next to it
        with graph_module.graph.inserting_after(node):
          size_ph = graph_module.graph.placeholder(f"{sym_str}_size")
        self._sym_shape_manager.symint_to_placeholder[sym_str] = size_ph


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
          arange: "i64[s77]" = arange.start(0, arg0_1)
          return (arange,)

    Modified FX Graph:
      def forward(self, arg0_1: "Sym(s77)", s77_size: "i32[]"):
        arange: "i64[s77]" = arange.start(0, arg0_1)
        arange_bounded: "i64[s77]" = set_dimension_logical_size(
            arange, 0, s77_size
        )
        return (arange_bounded,)
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""

    for node in graph_module.graph.nodes:
      if node.op == "call_function" and node.target in [
          torch.ops.aten.arange.default,
          torch.ops.aten.arange.start,
      ]:
        self._process_generative_op(graph_module, node)

  def _process_generative_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes generative op node."""
    new_args = list(node.args)
    modified_args = False
    tensor_node = None

    for arg in new_args:
      if sym_utils.is_symint_node(arg):
        tensor_node = self._sym_shape_manager.get_or_create_tensor_node(
            arg, node
        )

        assert tensor_node is not None, f"tensor node for {arg} not found"
        modified_args = True
        break

    if modified_args and tensor_node is not None:
      set_dim_size_node = self._insert_set_dimension_logical_size(
          graph_module, node, tensor_node
      )

      node.replace_all_uses_with(
          set_dim_size_node, delete_user_cb=lambda u: u != set_dim_size_node
      )

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
      set_dim_size_node.name = f"{node.name}_bounded"
    return set_dim_size_node


def apply_dynamism_transformations(
    graph_module: torch.fx.GraphModule, sym_shape_manager: SymShapeManager
) -> None:
  """Runs all FX graph transforms for dynamic shapes."""

  # Fetch original placeholders once in argument order
  original_placeholders = list(
      graph_module.graph.find_nodes(op="placeholder", sort=True)
  )

  # Scan inputs and create placeholders for SymInt inputs.
  GraphTransformObserver(
      graph_module, "scan_inputs_create_placeholders"
  ).apply_gm_pass(
      ScanInputsCreatePlaceholdersPass(sym_shape_manager, original_placeholders)
  )

  # Updates ops that have input tensors with dynamic dimensions.
  GraphTransformObserver(graph_module, "handle_dynamic_inputs").apply_gm_pass(
      HandleDynamicInputTensorPass(sym_shape_manager, original_placeholders)
  )

  # Updates the generative ops that have dynamic scalar inputs.
  GraphTransformObserver(graph_module, "handle_generative_ops").apply_gm_pass(
      HandleGenerativeOpsPass(sym_shape_manager)
  )

  graph_module.recompile()
  graph_module.graph.lint()
