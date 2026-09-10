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

"""FX pass to reassociate normalization weight expressions into distributive FMA form.

Rewrites patterns such as:
  `norm_x * (1.0 + weight)` or `(1.0 + weight) * norm_x`
  or `(weight + 1.0) * norm_x` or with dtype conversions `_to_copy` on `weight`
into:
  `norm_x + (norm_x * weight)`.

This pass only matches 1D tensors (rank <= 1) for `weight`. There is no such
rank restriction on `norm_x`, and handling for higher-dimensional weights has
not yet been explored.

This tries to eliminate rms norm related computations/conversions that are
unanchored by an activation. The outcome is better memory scheduling behavior
and the opportunity to avoid weight conversions since fma will account for acc
precision internally.
"""

from typing import Any

import torch
import torch.fx
from torch_tpu._internal.compile import tpu_torch_compile


def _is_reassociate_norm_weights_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  return tpu_torch_compile.get_reassociate_norm_weights_env_value()


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _is_one_scalar(val: Any) -> bool:
  """Checks whether an argument represents a scalar 1 or 1.0."""
  if isinstance(val, (int, float)) and val == 1:
    return True
  if isinstance(val, torch.fx.Node):
    if val.op == "call_function":
      target = getattr(val.target, "overloadpacket", val.target)
      if target == torch.ops.aten.scalar_tensor:
        if (
            len(val.args) >= 1
            and isinstance(val.args[0], (int, float))
            and val.args[0] == 1
        ):
          return True
      elif target == torch.ops.aten.full:
        fill_val = (
            val.args[1]
            if len(val.args) >= 2
            else val.kwargs.get("fill_value", None)
        )
        if isinstance(fill_val, (int, float)) and fill_val == 1:
          shape = (
              val.args[0] if len(val.args) >= 1 else val.kwargs.get("size", ())
          )
          if shape in ((), [], (1,), [1]):
            return True
      elif target == torch.ops.aten.ones:
        shape = (
            val.args[0] if len(val.args) >= 1 else val.kwargs.get("size", ())
        )
        if shape in ((), [], (1,), [1]):
          return True
    for meta_key in ("val", "example_value"):
      meta_val = val.meta.get(meta_key, None)
      if (
          meta_val is not None
          and isinstance(meta_val, torch.Tensor)
          and meta_val.numel() == 1
      ):
        try:
          if meta_val.item() == 1:
            return True
        except Exception:
          pass
  return False


def _get_tensor_rank(node: torch.fx.Node) -> int | None:
  """Extracts the tensor rank (ndim) from node metadata if available."""
  if "val" in node.meta and hasattr(node.meta["val"], "ndim"):
    return node.meta["val"].ndim
  if "example_value" in node.meta and hasattr(
      node.meta["example_value"], "ndim"
  ):
    return node.meta["example_value"].ndim
  if "tensor_meta" in node.meta and hasattr(node.meta["tensor_meta"], "shape"):
    return len(node.meta["tensor_meta"].shape)
  if (
      node.op == "call_function"
      and getattr(node.target, "overloadpacket", node.target)
      == torch.ops.aten._to_copy
      and len(node.args) >= 1
      and isinstance(node.args[0], torch.fx.Node)
  ):
    return _get_tensor_rank(node.args[0])
  return None


def _get_tensor_dtype(node: torch.fx.Node) -> torch.dtype | None:
  """Extracts the tensor dtype from node metadata or kwargs if available."""
  if "val" in node.meta and hasattr(node.meta["val"], "dtype"):
    return node.meta["val"].dtype
  if "example_value" in node.meta and hasattr(
      node.meta["example_value"], "dtype"
  ):
    return node.meta["example_value"].dtype
  if "tensor_meta" in node.meta and hasattr(node.meta["tensor_meta"], "dtype"):
    return node.meta["tensor_meta"].dtype
  if (
      node.op == "call_function"
      and getattr(node.target, "overloadpacket", node.target)
      == torch.ops.aten._to_copy
  ):
    if "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
      return node.kwargs["dtype"]
    if len(node.args) >= 2 and isinstance(node.args[1], torch.dtype):
      return node.args[1]
    if len(node.args) >= 1 and isinstance(node.args[0], torch.fx.Node):
      return _get_tensor_dtype(node.args[0])
  return None


def _has_dependency(node: torch.fx.Node, target_node: torch.fx.Node) -> bool:
  """Checks if target_node is an ancestor of node."""
  visited = set()
  stack = [node]
  while stack:
    curr = stack.pop()
    if curr == target_node:
      return True
    if curr in visited:
      continue
    visited.add(curr)
    for inp in curr.all_input_nodes:
      stack.append(inp)
  return False


def _match_add_one_weight(
    node: Any,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node | None] | None:
  """Matches (1.0 + weight) or (weight + 1.0), with optional outer _to_copy.

  Args:
    node: An FX node to match.

  Returns:
    A tuple of (add_node, weight_node, outer_to_copy_node) if matched, or None.
  """
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return None

  outer_to_copy_nodes = []
  curr = node
  while (
      isinstance(curr, torch.fx.Node)
      and curr.op == "call_function"
      and getattr(curr.target, "overloadpacket", curr.target)
      == torch.ops.aten._to_copy
      and len(curr.args) >= 1
      and isinstance(curr.args[0], torch.fx.Node)
  ):
    outer_to_copy_nodes.append(curr)
    curr = curr.args[0]
  add_candidate = curr
  outer_to_copy_node = outer_to_copy_nodes[0] if outer_to_copy_nodes else None

  if (
      not isinstance(add_candidate, torch.fx.Node)
      or add_candidate.op != "call_function"
  ):
    return None

  add_target = getattr(
      add_candidate.target, "overloadpacket", add_candidate.target
  )
  if add_target != torch.ops.aten.add:
    return None

  if len(add_candidate.args) < 2:
    return None

  # Reject non-unit alpha scaling in add(..., alpha=...)
  if add_candidate.kwargs.get("alpha", 1) != 1:
    return None
  if len(add_candidate.args) >= 3 and add_candidate.args[2] != 1:
    return None

  arg0 = add_candidate.args[0]
  arg1 = add_candidate.args[1]

  weight_node = None
  if _is_one_scalar(arg0) and isinstance(arg1, torch.fx.Node):
    weight_node = arg1
  elif _is_one_scalar(arg1) and isinstance(arg0, torch.fx.Node):
    weight_node = arg0

  if weight_node is None:
    return None

  # Weight should be a 1D tensor (or rank 0/1, or unknown).
  weight_rank = _get_tensor_rank(weight_node)
  if weight_rank is not None and weight_rank > 1:
    return None

  return add_candidate, weight_node, outer_to_copy_node


def _unwrap_to_copy(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.dtype | None]:
  """Unwraps chained _to_copy nodes to find the original node and dtype."""
  curr = node
  while (
      curr.op == "call_function"
      and getattr(curr.target, "overloadpacket", curr.target)
      == torch.ops.aten._to_copy
      and len(curr.args) >= 1
      and isinstance(curr.args[0], torch.fx.Node)
  ):
    curr = curr.args[0]
  return curr, _get_tensor_dtype(curr)


def _set_node_meta(
    new_node: torch.fx.Node,
    source_node: torch.fx.Node,
    dtype: torch.dtype | None = None,
) -> None:
  """Copies metadata from source_node and updates tensor dtype if specified."""
  if hasattr(source_node, "meta"):
    new_node.meta = source_node.meta.copy()
    if dtype is not None:
      for key in ("val", "example_value"):
        if key in new_node.meta and isinstance(
            new_node.meta[key], torch.Tensor
        ):
          new_node.meta[key] = new_node.meta[key].to(dtype)
      if "tensor_meta" in new_node.meta and hasattr(
          new_node.meta["tensor_meta"], "_replace"
      ):
        new_node.meta["tensor_meta"] = new_node.meta["tensor_meta"]._replace(
            dtype=dtype
        )


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Rewrites `norm_x * (1.0 + weight)` to `norm_x + (norm_x * weight)`.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  if not _is_reassociate_norm_weights_enabled():
    return

  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  mul_nodes = [
      n
      for n in graph.nodes
      if n.op == "call_function"
      and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
  ]

  for mul_node in mul_nodes:
    if mul_node.graph is None or len(mul_node.args) < 2:
      continue

    arg0, arg1 = mul_node.args[0], mul_node.args[1]
    if not isinstance(arg0, torch.fx.Node) and not isinstance(
        arg1, torch.fx.Node
    ):
      continue

    # Try matching arg1 as (1.0 + weight) and arg0 as norm_x
    match_result = _match_add_one_weight(arg1)
    if match_result is not None and isinstance(arg0, torch.fx.Node):
      _, weight_node, _ = match_result
      norm_x = arg0
    else:
      # Try matching arg0 as (1.0 + weight) and arg1 as norm_x
      match_result = _match_add_one_weight(arg0)
      if match_result is not None and isinstance(arg1, torch.fx.Node):
        _, weight_node, _ = match_result
        norm_x = arg1
      else:
        continue

    # Verify weight does not depend on norm_x
    if _has_dependency(weight_node, norm_x):
      continue

    # Determine desired dtype for the multiplication.
    # In aten IR, mul_node takes norm_x and weight in the same dtype.
    norm_x_dtype = _get_tensor_dtype(norm_x)
    weight_dtype = _get_tensor_dtype(weight_node)
    orig_weight, orig_weight_dtype = _unwrap_to_copy(weight_node)

    # Check if all users of mul_node are _to_copy nodes casting to orig_weight_dtype.
    mul_users = list(mul_node.users)
    downstream_copy_users = []
    if (
        orig_weight_dtype is not None
        and mul_users
        and all(
            u.op == "call_function"
            and getattr(u.target, "overloadpacket", u.target)
            == torch.ops.aten._to_copy
            and _get_tensor_dtype(u) == orig_weight_dtype
            for u in mul_users
        )
    ):
      downstream_copy_users = mul_users

    # Select target dtype for the distributive FMA:
    # If downstream users cast back to orig_weight_dtype, or if norm_x matches orig_weight_dtype,
    # execute FMA directly in orig_weight_dtype.
    if downstream_copy_users or (
        orig_weight_dtype is not None and norm_x_dtype == orig_weight_dtype
    ):
      target_dtype = orig_weight_dtype
    else:
      target_dtype = norm_x_dtype

    with graph.inserting_before(mul_node):
      # Prepare effective norm_x in target_dtype
      effective_norm_x = norm_x
      if target_dtype is not None and norm_x_dtype != target_dtype:
        # Check if norm_x is a _to_copy whose input already matches target_dtype
        unwrapped_norm_x, unwrapped_norm_dtype = _unwrap_to_copy(norm_x)
        if unwrapped_norm_dtype == target_dtype:
          effective_norm_x = unwrapped_norm_x
        else:
          effective_norm_x = graph.call_function(
              torch.ops.aten._to_copy.default,
              args=(norm_x,),
              kwargs={"dtype": target_dtype},
          )
          _set_node_meta(effective_norm_x, norm_x, target_dtype)

      # Prepare effective weight in target_dtype
      effective_weight = weight_node
      if target_dtype is not None:
        if orig_weight_dtype == target_dtype:
          effective_weight = orig_weight
        elif weight_dtype != target_dtype:
          effective_weight = graph.call_function(
              torch.ops.aten._to_copy.default,
              args=(weight_node,),
              kwargs={"dtype": target_dtype},
          )
          _set_node_meta(effective_weight, weight_node, target_dtype)

      # 1. new_mul = norm_x * weight
      new_mul = graph.call_function(
          torch.ops.aten.mul.Tensor,
          args=(effective_norm_x, effective_weight),
      )
      _set_node_meta(new_mul, mul_node, target_dtype)

      # 2. new_add = norm_x + (norm_x * weight)
      new_add = graph.call_function(
          torch.ops.aten.add.Tensor,
          args=(effective_norm_x, new_mul),
      )
      _set_node_meta(new_add, mul_node, target_dtype)

    # Replace uses and clean up nodes
    if downstream_copy_users:
      for u in downstream_copy_users:
        u.replace_all_uses_with(new_add)
        graph.erase_node(u)
      graph.erase_node(mul_node)
    else:
      mul_node.replace_all_uses_with(new_add)
      graph.erase_node(mul_node)

  graph.eliminate_dead_code()
  graph.lint()
  if gm is not None:
    for _, submod in gm.named_children():
      if isinstance(submod, torch.fx.GraphModule):
        apply(submod)
    gm.recompile()
