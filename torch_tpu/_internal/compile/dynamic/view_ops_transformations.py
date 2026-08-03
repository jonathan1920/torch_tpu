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
from collections.abc import Sequence
import sys
from typing import Any
from absl import logging
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager


def _get_node_val(arg: Any) -> Any:
  """Unwraps an FX Node to its metadata value ('val') if present."""
  if isinstance(arg, torch.fx.Node) and "val" in arg.meta:
    return arg.meta["val"]
  return arg


def _extract_shape_tensors_and_bounds(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    target_shape: Sequence[Any],
    sym_shape_manager: SymShapeManager,
) -> tuple[list[torch.fx.Node], list[int], list[bool]]:
  """Extracts shape tensor nodes, static bounds, and dynamic flags for a target shape."""
  shape_tensors = []
  static_shape = []
  is_dynamic = []

  for arg in target_shape:
    if sym_utils.is_symint(arg):
      is_dynamic.append(True)
      symint = arg.meta["val"] if sym_utils.is_symint_node(arg) else arg
      _, upper = symbol_bounds.get_symint_bounds(symint)
      static_shape.append(upper)
      tensor_node = sym_shape_manager.ensure_tensor(
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
      tensor_node = sym_shape_manager.ensure_tensor(
          graph_module, arg, node, dtype=torch.int32
      )
      shape_tensors.append(tensor_node)

  return shape_tensors, static_shape, is_dynamic


class HandleReshapeLikeOpsPass:
  """Reshape-like ops transformation pass.

  Detects reshape-like view operations (such as aten.view, aten.reshape,
  aten._unsafe_view) that have dynamic dimensions (SymInt or SymInt
  expressions) as input and replaces them with torch.ops.tpu.dynamic_reshape.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.view.default: self._process_view_op,
        torch.ops.aten._unsafe_view.default: self._process_view_op,
        torch.ops.aten.reshape.default: self._process_view_op,
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the reshape-like ops transformation pass."""
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
        1 for arg in target_shape if sym_utils.is_symint(arg)
    )
    if num_dynamic_dims == 0:
      return

    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, target_shape, self._sym_shape_manager
    )

    with graph_module.graph.inserting_after(node):
      dynamic_reshape_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_reshape,
          args=(inp, shape_tensors, static_shape, is_dynamic),
      )
      dynamic_reshape_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_reshape_node)
    graph_module.graph.erase_node(node)


class HandleBroadcastLikeOpsPass:
  """Broadcast-like ops transformation pass.

  Detects broadcast operations (such as aten.expand, aten.expand_copy) that
  have dynamic dimensions (SymInt or SymInt expressions) as input and replaces
  them with torch.ops.tpu.dynamic_broadcast.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.expand.default: self._process_broadcast_op,
        torch.ops.aten.expand_copy.default: self._process_broadcast_op,
        torch.ops.aten.broadcast_to.default: self._process_broadcast_op,
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the broadcast-like ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_broadcast_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes broadcast op node and converts to dynamic_broadcast if dynamic."""
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
        1 for arg in target_shape if sym_utils.is_symint(arg)
    )
    if num_dynamic_dims == 0:
      return

    if (
        isinstance(inp, torch.fx.Node)
        and "val" in inp.meta
        and hasattr(inp.meta["val"], "shape")
    ):
      inp_shape = inp.meta["val"].shape
    else:
      inp_shape = []
    r_in = len(inp_shape)
    r_out = len(target_shape)
    broadcast_dims = list(range(r_out - r_in, r_out))

    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, target_shape, self._sym_shape_manager
    )

    with graph_module.graph.inserting_after(node):
      dynamic_broadcast_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_broadcast,
          args=(inp, shape_tensors, broadcast_dims, static_shape, is_dynamic),
      )
      dynamic_broadcast_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_broadcast_node)
    graph_module.graph.erase_node(node)


class HandleSliceLikeOpsPass:
  """Slice-like ops transformation pass.

  Detects slice operations (such as aten.slice.Tensor) that have dynamic
  dimensions (SymInt or SymInt expressions) as end or step input (with start=0)
  and calculates the dynamic size of the sliced dimension using tensor
  arithmetic, then applies set_dimension_logical_size.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.slice.Tensor: self._process_slice_op,
        torch.ops.aten.slice_backward.default: self._process_slice_backward_op,
    }

  def _extract_slice_args(
      self, node: torch.fx.Node, offset: int = 0
  ) -> tuple[int, Any, Any, Any]:
    """Extracts (dim, start, end, step) from slice or slice_backward node args/kwargs."""
    dim = (
        node.args[1 + offset]
        if len(node.args) > 1 + offset
        else node.kwargs.get("dim", 0)
    )
    start = _get_node_val(
        node.args[2 + offset]
        if len(node.args) > 2 + offset
        else node.kwargs.get("start", 0)
    )
    end = _get_node_val(
        node.args[3 + offset]
        if len(node.args) > 3 + offset
        else node.kwargs.get("end", None)
    )
    step = _get_node_val(
        node.args[4 + offset]
        if len(node.args) > 4 + offset
        else node.kwargs.get("step", 1)
    )
    return dim, start, end, step

  def _replace_node_with_set_logical_size(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      new_op_node: torch.fx.Node,
      dim: int,
      size_tensor_node: torch.fx.Node,
  ) -> None:
    """Wraps new_op_node with set_dimension_logical_size, replaces all uses, and erases node."""
    with graph_module.graph.inserting_before(node):
      new_op_node.meta = node.meta.copy()
      set_dim_size_node = graph_module.graph.call_function(
          torch.ops.tpu.set_dimension_logical_size,
          args=(new_op_node, dim, size_tensor_node),
      )
      set_dim_size_node.meta = node.meta.copy()

    node.replace_all_uses_with(set_dim_size_node)
    graph_module.graph.erase_node(node)

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the slice-like ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_slice_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes slice op node and applies set_dimension_logical_size if end or step is dynamic."""
    inp = node.args[0]
    dim, start, end, step = self._extract_slice_args(node, offset=0)

    if sym_utils.is_symint(start) or (start is not None and start != 0):
      return

    inp_val = _get_node_val(inp)

    # If end is omitted (None or sys.maxsize), default to full dimension size.
    if end is None or (isinstance(end, int) and end >= sys.maxsize):
      if hasattr(inp_val, "shape"):
        end = inp_val.shape[dim]

    if step is None:
      step = 1

    if not (sym_utils.is_symint(end) or sym_utils.is_symint(step)):
      return

    # Normalize negative dim and end indices to absolute indices.
    if hasattr(inp_val, "shape"):
      if isinstance(dim, int) and dim < 0:
        dim = dim + len(inp_val.shape)
      if isinstance(end, int) and end < 0:
        end = inp_val.shape[dim] + end

    # Convert end, step, and constant 1 values into 0D int32 tensors.
    end_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, end, node, dtype=torch.int32
    )
    step_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, step, node, dtype=torch.int32
    )
    one_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, 1, node, dtype=torch.int32
    )

    # Determine static integer upper bounds for the static slice op.
    if hasattr(inp_val, "shape") and sym_utils.is_symint(inp_val.shape[dim]):
      end_upper = symbol_bounds.get_upper_bound(inp_val.shape[dim])
    else:
      end_upper = symbol_bounds.get_upper_bound(end)
    step_upper = symbol_bounds.get_upper_bound(step)

    # Determine sliced dimension logical size via tensor arithmetic:
    # (end + step - 1) // step
    with graph_module.graph.inserting_before(node):
      p_end = torch.fx.Proxy(end_tensor)
      p_step = torch.fx.Proxy(step_tensor)
      p_one = torch.fx.Proxy(one_tensor)
      p_size = ((p_end + p_step - p_one) // p_step).to(dtype=torch.int32)
      size_tensor_node = p_size.node

      new_slice_node = graph_module.graph.call_function(
          torch.ops.aten.slice.Tensor,
          args=(inp, dim, start, end_upper, step_upper),
      )

    self._replace_node_with_set_logical_size(
        graph_module, node, new_slice_node, dim, size_tensor_node
    )

  def _process_slice_backward_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes slice_backward op node and applies set_dimension_logical_size if parameters are dynamic."""
    grad_output, input_sizes = node.args[0], node.args[1]
    dim, start, end, step = self._extract_slice_args(node, offset=1)

    if sym_utils.is_symint(start):
      return

    has_dynamic = (
        sym_utils.is_symint(end)
        or sym_utils.is_symint(step)
        or any(sym_utils.is_symint(s) for s in input_sizes)
    )
    if not has_dynamic:
      return

    input_sizes_upper = [symbol_bounds.get_upper_bound(s) for s in input_sizes]
    end_upper = symbol_bounds.get_upper_bound(end)
    step_upper = symbol_bounds.get_upper_bound(step)

    input_size_sym = input_sizes[dim]
    size_tensor_node = self._sym_shape_manager.ensure_tensor(
        graph_module, input_size_sym, node, dtype=torch.int32
    )

    with graph_module.graph.inserting_before(node):
      new_slice_bwd_node = graph_module.graph.call_function(
          torch.ops.aten.slice_backward.default,
          args=(
              grad_output,
              input_sizes_upper,
              dim,
              start,
              end_upper,
              step_upper,
          ),
      )

    self._replace_node_with_set_logical_size(
        graph_module, node, new_slice_bwd_node, dim, size_tensor_node
    )


class ReplaceDynamicOutputBroadcastOpsPreGradPass:
  """Pre-grad pass to replace returned dynamic broadcast/expand ops with expand_copy.

  Why this pass is needed:
  When a model returns a broadcast or expand operation directly as a graph
  output (e.g. `return x.expand(dynamic_size, ...)` or `return
  torch.broadcast_to(...)`) and the target shape depends on dynamic SymInt
  expressions, returning a view op causes PyTorch's AOTAutograd alias
  resolution to fail. AOTAutograd attempts to generate an alias reconstruction
  function using `as_strided`, but the original input tensor storage size is
  smaller than the expanded output shape causing a runtime storage size
  validation error.

  By replacing returned dynamic broadcast/expand nodes with
  `aten.expand_copy.default` before AOTAutograd runs (Pre-Grad), the returned
  tensor becomes an explicit contiguous copy rather than an aliased view,
  allowing AOTAutograd to handle the output safely.
  """

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the pre-grad pass to replace returned dynamic broadcast ops."""
    logging.debug(
        "[PreGrad] ReplaceDynamicOutputBroadcastOpsPreGradPass running on"
        " graph module: %s",
        graph_module,
    )
    output_nodes = list(graph_module.graph.find_nodes(op="output"))
    if not output_nodes:
      logging.debug("[PreGrad] No output nodes found")
      return

    returned_nodes: set[torch.fx.Node] = set()
    for out_node in output_nodes:
      flat_args, _ = pytree.tree_flatten(out_node.args)
      for arg in flat_args:
        if isinstance(arg, torch.fx.Node):
          returned_nodes.add(arg)

    logging.debug("[PreGrad] returned_nodes: %s", returned_nodes)
    changed = False
    for node in list(graph_module.graph.nodes):
      is_expand = (
          node.op == "call_function"
          and node.target
          in (
              torch.ops.aten.expand.default,
              torch.ops.aten.expand,
              "expand",
              torch.ops.aten.broadcast_to.default,
              torch.ops.aten.broadcast_to,
              torch.broadcast_to,
              "broadcast_to",
          )
      ) or (
          node.op == "call_method" and node.target in ("expand", "broadcast_to")
      )
      if not is_expand:
        continue

      is_returned = node in returned_nodes
      # Extract input tensor node (`inp`) and target shape dimensions (`shape`).
      # Method calls (e.g. `x.expand(d0, d1)` or `x.expand([d0, d1])`) pass
      # `self` as `node.args[0]`, followed by shape as varargs or sequence.
      if node.op == "call_method":
        inp = node.args[0]
        if len(node.args) == 2 and isinstance(node.args[1], (list, tuple)):
          shape = list(node.args[1])
        else:
          shape = list(node.args[1:])
      # Function calls (e.g. `torch.broadcast_to(x, size)` or
      # `aten.expand(x, size)`) pass input tensor `x` as `node.args[0]` and
      # shape via `node.args[1]` or `node.kwargs["size"]`.
      else:
        inp = node.args[0]
        shape = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("size", [])
        )
        if not isinstance(shape, (list, tuple)):
          shape = [shape]

      has_symint = any(
          isinstance(arg, torch.fx.Node)
          or isinstance(arg, torch.SymInt)
          or sym_utils.is_symint_node(arg)
          for arg in shape
      )

      if is_returned and has_symint:
        with graph_module.graph.inserting_after(node):
          expand_copy_node = graph_module.graph.call_function(
              torch.ops.aten.expand_copy.default,
              args=(inp, shape),
              kwargs=node.kwargs,
          )
          expand_copy_node.meta = node.meta.copy()

        node.replace_all_uses_with(expand_copy_node)
        changed = True
        logging.debug(
            "[PreGrad] Successfully replaced %s with expand_copy_node %s",
            node,
            expand_copy_node,
        )

    if changed:
      graph_module.graph.eliminate_dead_code()
      graph_module.recompile()
