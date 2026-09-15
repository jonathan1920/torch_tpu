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

"""Pass to convert constant SymInt placeholders and expressions to concrete integers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.utils import InputType
from torch_tpu._internal.compile.dynamic import sym_utils


def _replace_node_with_value(node: torch.fx.Node, value: Any) -> None:
  """Replaces uses of an FX node with a scalar constant value."""
  for user in list(node.users.keys()):
    user.args = torch.fx.map_arg(user.args, lambda n: value if n is node else n)
    user.kwargs = torch.fx.map_arg(
        user.kwargs, lambda n: value if n is node else n
    )


class ConvertConstantSymIntsPass:
  """Converts constant SymInt placeholders and expressions to concrete integers.

  Dynamo can produce SymInt placeholders or intermediate expressions that
  represent constant integers (having 0 free symbols, e.g. Sym(62)).
  This pass converts such SymInts to concrete Python integers in both the
  FX graph and example inputs, preventing downstream passes from treating them
  as dynamic symbols.
  """

  def __init__(
      self,
      placeholders: Sequence[torch.fx.Node],
      example_inputs: Sequence[InputType],
  ):
    self._placeholders = placeholders
    self._example_inputs = list(example_inputs)

  @property
  def updated_example_inputs(self) -> Sequence[InputType]:
    return self._example_inputs

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    # 1. Update placeholder nodes and example_inputs
    for idx, node in enumerate(self._placeholders):
      val = node.meta.get("val")

      if sym_utils.is_constant_symint(val):
        const_int = sym_utils.get_constant_symint_value(val)
        if const_int is not None:
          self._example_inputs[idx] = const_int
          node.meta["val"] = const_int
          node.type = int
          _replace_node_with_value(node, const_int)
      elif isinstance(val, torch.Tensor):
        if any(sym_utils.is_constant_symint(d) for d in val.shape):
          new_shape = tuple(
              sym_utils.get_constant_symint_value(d)
              if sym_utils.is_constant_symint(d)
              else d
              for d in val.shape
          )
          node.meta["val"] = val.new_empty(new_shape)
          example_input = self._example_inputs[idx]
          if isinstance(example_input, torch.Tensor):
            self._example_inputs[idx] = example_input.new_empty(new_shape)

    # 2. Update intermediate nodes whose evaluated value is a constant SymInt
    for node in list(graph_module.graph.nodes):
      if node.op in ("placeholder", "output"):
        continue
      val = node.meta.get("val")
      if sym_utils.is_constant_symint(val):
        const_int = sym_utils.get_constant_symint_value(val)
        if const_int is not None:
          node.meta["val"] = const_int
          _replace_node_with_value(node, const_int)

    graph_module.graph.eliminate_dead_code()
