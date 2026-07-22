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

"""Step functions specifies what one step does for a benchmark.

Every step fn is defined the same way: a factory taking zero or more kwargs
and returning the actual step callable.

The returned callable always has the signature

    step(model, input_args, input_kwargs, optimizer) -> Any

and the expected model invocation is model(*input_args, **input_kwargs).

The step functions implemented here should not synchronize on the outputs. The
caller has the freedom to synchronize and/or block on outputs in any way they
desire.
"""

import enum
from typing import Callable, Dict

from absl import logging
import torch


class StepFn(enum.Enum):
  """The closed set of step functions."""

  FORWARD = "forward"
  TRAINING = "training"


# StepFn -> factory(**kwargs) -> step(model, input_args, input_kwargs, optimizer)
STEP_FNS: Dict[StepFn, Callable[..., Callable]] = {}


def step_fn(name: StepFn):
  """Register a step-fn factory under `name`."""

  def deco(factory):
    if name in STEP_FNS:
      raise ValueError(f"duplicate step fn: {name}")
    STEP_FNS[name] = factory
    return factory

  return deco


def resolve_step_fn(name: StepFn, **kwargs) -> Callable:
  """Build the step callable with the provided kwargs."""
  return STEP_FNS[name](**kwargs)


@step_fn(StepFn.FORWARD)
def forward():
  """Forward only, no grad."""

  def step(model, input_args, input_kwargs, optimizer=None):
    del optimizer  # unused
    with torch.inference_mode():
      return model(*input_args, **input_kwargs)

  return step


def real_loss(model, input_args, input_kwargs) -> torch.Tensor:
  """Default compute_loss for training step functions.

  Run the model forward pass, produce a scalar loss to call backward() on.

  Note: This doesn't do the naive fallback (`out.loss if hasattr(...) else
  out.sum()`). The fallback measures a different backward graph when a model
  has a fused loss path that wasn't triggered (e.g. it only returns `.loss` when
  passed `labels=`), so it would miss the loss/softmax/cross_entropy backward
  and measure the wrong thing.

  Raises TypeError if it cannot find loss value in the model output.
  """
  out = model(*input_args, **input_kwargs)

  if hasattr(out, "loss") and out.loss is not None:
    logging.info("real_loss: extracted from out.loss")
    loss = out.loss
  elif isinstance(out, dict) and "loss" in out:
    logging.info("real_loss: extracted from out['loss']")
    loss = out["loss"]
  elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
    logging.info("real_loss: extracted from out[0]")
    loss = out[0]
  elif torch.is_tensor(out):
    logging.info("real_loss: out is directly a tensor")
    loss = out
  else:
    raise TypeError(
        f"real_loss found no loss in {type(out).__name__}. Pass"
        " step_fn_kwargs={'compute_loss': ...}, or use grad_probe for pure"
        " perf."
    )

  if not torch.is_tensor(loss):
    raise TypeError(f"real_loss: loss is {type(loss).__name__}, not a Tensor")

  if loss.ndim == 0:
    return loss
  else:
    logging.info(f"real_loss: Taking mean of loss with shape {loss.shape}")
    return loss.mean()


@step_fn(StepFn.TRAINING)
def training(accum_steps: int = 1, compute_loss=real_loss):
  """Forward+backward over accum_steps micro-batches, then one optimizer step.

  Args:
    accum_steps: The number of micro-batches to accumulate gradients over before
      taking an optimizer step.
    compute_loss: A callable to compute the loss from the model output. Defaults
      to `real_loss`. Override the compute_loss function if you want to measure
      a different backward graph than the one measured by the default.
  """
  if accum_steps < 1:
    raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
  compute = compute_loss or real_loss

  def step(model, input_args, input_kwargs, optimizer):
    if optimizer is None:
      raise ValueError("training step requires an optimizer")
    optimizer.zero_grad(set_to_none=True)
    loss = None
    for _ in range(accum_steps):
      # Scale by accum_steps: backward sums gradients across micro-batches, so
      # without this the gradient is silently scaled by accum_steps.
      loss = compute(model, input_args, input_kwargs) / accum_steps
      loss.backward()
    optimizer.step()
    return loss

  return step
