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

"""Simple trainer library for compiling and running PyTorch training steps on TPU."""

import functools
import typing
from typing import Any, Callable

import torch
import torch.func as func
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree
from torch_tpu._internal.compile.compiler import StaticCompiler
from examples.benchmarks import optimizers

# Re-export optimizer classes for simple_trainer
Optimizer = optimizers.Optimizer
AdamW = optimizers.AdamW
SGD = optimizers.SGD
ReferenceAdamw = optimizers.ReferenceAdamw
FusedAdamw = optimizers.FusedAdamw
ReferenceSgd = optimizers.ReferenceSgd
FusedSgd = optimizers.FusedSgd


def _compute_loss(
    model: torch.nn.Module,
    key_map: dict[str, str],
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    inputs: Any,
    targets: Any,
) -> torch.Tensor:
  """Computes loss for functional execution."""
  p_orig = {key_map[k]: v for k, v in params.items()}
  b_orig = {key_map[k]: v for k, v in buffers.items()}

  if isinstance(inputs, dict):
    args, kwargs = (), inputs
  elif isinstance(inputs, (tuple, list)):
    args, kwargs = tuple(inputs), {}
  else:
    args, kwargs = (inputs,), {}

  out = func.functional_call(model, (p_orig, b_orig), args, kwargs)

  if targets is not None:
    return torch.nn.functional.mse_loss(out, targets)

  # Specific heuristics
  if hasattr(out, "loss") and out.loss is not None:
    return out.loss
  if isinstance(out, dict) and "loss" in out:
    return out["loss"]
  if hasattr(out, "logits") and out.logits is not None:
    return torch.mean(out.logits)
  if hasattr(out, "sample") and out.sample is not None:
    return torch.mean(out.sample)

  # Generic fallback using pytree
  flat_out, _ = _pytree.tree_flatten(out)
  for item in flat_out:
    if torch.is_tensor(item):
      return torch.mean(item)

  raise TypeError(f"Cannot extract loss from output of type {type(out)}")


class SingleTraceTrainer:
  """Facilitates compiled train steps with a single offload containing the optimizer step.

  This trainer captures the entire training loop (forward pass, loss
  computation,
  gradient calculation, and optimizer step) into a single functional graph using
  `make_fx`. This graph is then compiled and executed as a single unit
  (offload),
  reducing interpreter overhead and enabling better fusion opportunities by the
  compiler.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      optimizer: Any,
  ):
    """Initializes the SingleTraceTrainer."""
    self.model = model
    self.optimizer = optimizer

    raw_params = dict(model.named_parameters())
    raw_buffers = dict(model.named_buffers())

    self._key_map = {}
    initial_params = {}
    for name, param in raw_params.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      initial_params[safe_name] = param.detach()

    self.buffers = {}
    for name, buf in raw_buffers.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      self.buffers[safe_name] = buf.detach()

    self.param_group = self.optimizer.init_param_group(initial_params)

  @property
  def params(self) -> dict[str, torch.Tensor]:
    """Returns parameters with their original names."""
    return {self._key_map[k]: v for k, v in self.param_group.params.items()}

  def make_compiled_train_step(
      self,
      example_inputs: Any,
      example_targets: Any = None,
  ) -> Callable[..., torch.Tensor]:
    """Returns a callable for the training step with the model compiled."""

    flat_inputs, in_spec = _pytree.tree_flatten(
        (self.param_group, self.buffers, example_inputs, example_targets)
    )

    # Pure tensor-level functional step captured by make_fx and compiled.
    def flattened_stateless_train_step(*flat_args):
      structured_args = _pytree.tree_unflatten(flat_args, in_spec)
      p_group, bufs, inps, tgts = structured_args

      bound_loss = functools.partial(
          _compute_loss,
          self.model,
          self._key_map,
          buffers=bufs,
          inputs=inps,
          targets=tgts,
      )

      grads, loss = func.grad_and_value(bound_loss)(p_group.params)
      new_p_group = self.optimizer(p_group, grads)

      flat_outputs, _ = _pytree.tree_flatten((loss, new_p_group))
      return tuple(flat_outputs)

    unified_graph = make_fx(
        flattened_stateless_train_step, tracing_mode="fake"
    )(*flat_inputs)

    flat_param_group, _ = _pytree.tree_flatten(self.param_group)
    donated_inputs = [
        i for i, x in enumerate(flat_param_group) if isinstance(x, torch.Tensor)
    ]

    static_compiler = StaticCompiler()

    compiled_step = static_compiler(
        typing.cast(torch.fx.GraphModule, unified_graph),
        flat_inputs,
        donated_inputs=donated_inputs,
    )

    # Build a template to capture output structure (`TreeSpec`) for
    # restoring backend flat results.
    dummy_loss = torch.tensor(0.0)
    _, out_spec = _pytree.tree_flatten((dummy_loss, self.param_group))

    # Stateful wrapper returned to user; handles flattening, running compiled
    # graph, and state updates.
    def train_step(inputs: Any, targets: Any = None) -> torch.Tensor:
      flat_inputs, _ = _pytree.tree_flatten(
          (self.param_group, self.buffers, inputs, targets)
      )

      result = compiled_step(*flat_inputs)

      assert out_spec is not None
      loss, new_param_group = _pytree.tree_unflatten(result, out_spec)

      self.param_group = new_param_group

      return loss

    return train_step
