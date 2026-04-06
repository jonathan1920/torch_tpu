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

"""Rewrites stateful random operations into their stateless equivalents."""

import operator
import torch
from torch import fx
from torch._inductor import pattern_matcher

CallFunctionVarArgs = pattern_matcher.CallFunctionVarArgs
Match = pattern_matcher.Match
PatternMatcherPass = pattern_matcher.PatternMatcherPass
register_graph_pattern = pattern_matcher.register_graph_pattern


def rewrite_stateless_rng_ops(
    gm_or_graph: fx.GraphModule | fx.Graph,
) -> fx.GraphModule | fx.Graph:
  """Rewrites stateful random operations into their stateless equivalents.

  Graph Modifications & Behavior:
  - Inputs: Adds an additional input to the graph as the last placeholder,
  representing
    the external `rng_state`.
  - Nodes: Matches `aten.native_dropout.default` and replaces it with
    `torch_tpu.stateless_dropout.default`. Threads the `rng_state` sequentially
    through
    multiple dropout calls in topological order.
  - Outputs: Adds an additional output to the graph representing the final,
    updated `rng_state`.

  Args:
    gm_or_graph: The GraphModule or Graph to rewrite.

  Returns:
    The rewritten GraphModule or Graph.
  """
  # -----------------------------------------------------------------
  # STEP 0: Handle both GraphModule and Graph inputs
  # -----------------------------------------------------------------
  if isinstance(gm_or_graph, fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  # -----------------------------------------------------------------
  # STEP 1: Add external rng_state as a new graph input
  # -----------------------------------------------------------------
  first_non_placeholder = None
  for node in graph.nodes:
    if node.op != 'placeholder':
      first_non_placeholder = node
      break

  if first_non_placeholder is None:
    raise ValueError('Invalid graph: no non-placeholder nodes found.')

  # Insert the new input right after the existing inputs
  with graph.inserting_before(first_non_placeholder):
    rng_state_input = graph.placeholder('rng_state')

  # Track the running RNG state as we traverse and match nodes
  current_rng_state = rng_state_input

  # -----------------------------------------------------------------
  # STEP 2: Define and apply the pattern matcher
  # -----------------------------------------------------------------
  graph_pass = PatternMatcherPass()

  @register_graph_pattern(
      CallFunctionVarArgs(torch.ops.aten.native_dropout.default),
      pass_dict=graph_pass,
  )
  def rng_op_replacement(match: Match, *args, **kwargs):
    nonlocal current_rng_state
    old_dropout = match.nodes[0]

    with graph.inserting_before(old_dropout):
      # Create the custom TPU dropout node using the running RNG state
      new_dropout = graph.call_function(
          torch.ops.torch_tpu.stateless_dropout.default,
          args=(current_rng_state, *args),
          kwargs=kwargs,
      )

      # Unpack the tuple return: (next_rng_state, out, mask)
      next_rng_state = graph.call_function(
          operator.getitem, args=(new_dropout, 0)
      )
      new_out = graph.call_function(operator.getitem, args=(new_dropout, 1))
      new_mask = graph.call_function(operator.getitem, args=(new_dropout, 2))

      # native_dropout returns (out, mask). Remap downstream getitem users.
      for user in list(old_dropout.users.keys()):
        if user.op == 'call_function' and user.target == operator.getitem:
          idx = user.args[1]
          if idx == 0:
            user.replace_all_uses_with(new_out)
          elif idx == 1:
            user.replace_all_uses_with(new_mask)

      # Update the running RNG state for the next matched operation
      current_rng_state = next_rng_state

  # Apply the pass directly to the graph/module
  graph_pass.apply(gm_or_graph)

  # -----------------------------------------------------------------
  # STEP 3: Add the final updated rng_state to the graph outputs
  # -----------------------------------------------------------------
  output_node = next(n for n in graph.nodes if n.op == 'output')
  orig_returns = output_node.args[0]

  if isinstance(orig_returns, (tuple, list)):
    new_returns = (*orig_returns, current_rng_state)
  else:
    # Failsafe if the graph previously returned a single unwrapped tensor
    new_returns = (orig_returns, current_rng_state)

  output_node.args = (new_returns,)

  # -----------------------------------------------------------------
  # STEP 4: Clean up
  # -----------------------------------------------------------------
  graph.eliminate_dead_code()

  return gm_or_graph
