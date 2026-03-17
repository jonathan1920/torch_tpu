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

"""This module contains functions that are benchmarked in the e2e benchmark suite.

All benchmark functions in this module must adhere to the following signature:

def benchmark_function(
    model: torch.nn.Module,
    inputs: Any,
    optimizer: torch.optim.Optimizer | None = None,
) -> Any:
  ...

The output should be a tensor if it's required to synchronize the device before
measuring the performance metrics. This signature is enforced by the benchmark
suite to ensure consistency and
interchangeability of benchmark functions. The arguments are:
  - model: The model to be benchmarked.
  - inputs: The input tensors for the model.
  - optimizer: An optional optimizer. This is unused in inference benchmarks,
      but required for training benchmarks.
"""

import functools
from typing import Any, Callable, Mapping
import torch
from torch_tpu._internal.utils import device_utils


def huggingface_llm_forward_pass(
    model: torch.nn.Module,
    inputs: Any,
    optimizer: torch.optim.Optimizer | None = None,  # pylint: disable=unused-argument
) -> torch.Tensor:
  """Performs a forward pass for a Hugging Face LLM.

  Args:
    model: The Hugging Face LLM model.
    inputs: The input tensors for the model.
    optimizer: An optional optimizer (unused in this function). This is to match
      the expected function signature for benchmarks.

  Returns:
    The output tensor from the model, typically the logits.
  """
  # TODO(bbahl): Change to inference mode after fixing error in
  # torch.inference_mode(). Currently it raises: RuntimeError: Cannot set
  # version_counter for inference tensor in torch.embedding.
  with torch.no_grad():
    result = model(**inputs)
    if hasattr(result, "logits"):
      result = result.logits
    return result


def _huggingface_llm_train_1_step(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
    grad_accumulation_steps: int,
    optimizer_step_fn: Callable[[torch.optim.Optimizer], None],
) -> float:
  """Performs one training step for a Hugging Face LLM.

  This function assumes that the output of the model is an object that has a
  `loss` attribute (e.g. CausalLMOutput), which is typical for Huggingface
  transformer models.

  Args:
    model: The Hugging Face LLM model.
    inputs: The input tensors for the model.
    optimizer: The optimizer for training.
    grad_accumulation_steps: The number of steps to accumulate gradients for.
    optimizer_step_fn: The function to call to step the optimizer.

  Returns:
    The loss for the training step.
  """
  if optimizer is None:
    raise ValueError("Optimizer must be provided for training.")
  accumulated_losses = []
  optimizer.zero_grad()
  for _ in range(grad_accumulation_steps):
    output = model(**inputs)
    output.loss.backward()
    accumulated_losses.append(output.loss.detach())
  optimizer_step_fn(optimizer)
  step_loss = torch.sum(torch.stack(accumulated_losses)).item()
  return step_loss


def get_huggingface_llm_training_function(
    device: torch.device, torch_compile: bool, grad_accumulation_steps: int
) -> Callable[[torch.nn.Module, Any, torch.optim.Optimizer], Any]:
  """Returns the benchmark function for training a Hugging Face LLM.

  Args:
    device: The device to run the benchmark on.
    torch_compile: Whether to compile the optimizer step function using
      `torch.compile`.
    grad_accumulation_steps: The number of gradient accumulation steps.

  Returns:
    A callable function suitable for benchmarking training, which takes
    `model`, `inputs`, and `optimizer` as arguments.
  """

  def get_optimizer_step_fn():
    def step_fn(optimizer):
      optimizer.step()

    if torch_compile:
      return device_utils.torch_compile(step_fn, device.type)
    return step_fn

  optimizer_step_fn = get_optimizer_step_fn()

  return functools.partial(
      _huggingface_llm_train_1_step,
      grad_accumulation_steps=grad_accumulation_steps,
      optimizer_step_fn=optimizer_step_fn,
  )


def meta_llama_forward_pass(
    model: torch.nn.Module,
    inputs: Any,
    optimizer: torch.optim.Optimizer | None = None,  # pylint: disable=unused-argument
) -> torch.Tensor:
  """Performs a forward pass for a Meta Llama model.

  Args:
    model: The Meta Llama model.
    inputs: A tuple containing (tokens, start_pos), where tokens is a tensor of
      input token indices and start_pos is an integer indicating the starting
      position.
    optimizer: An optional optimizer (unused in this function). This is to match
      the expected function signature for benchmarks.

  Returns:
    The output tensor from the model.
  """
  tokens, start_pos = inputs
  with torch.no_grad():
    result = model(tokens, start_pos)
  return result


def ml_layer_forward_pass(
    model: torch.nn.Module,
    inputs: Any,
    optimizer: torch.optim.Optimizer | None = None,  # pylint: disable=unused-argument
) -> torch.Tensor:
  """Performs a forward pass for an ML layer.

  Args:
    model: The ML layer model.
    inputs: The input tensors for the model.
    optimizer: Unused.

  Returns:
    The output tensor from the model.
  """
  del optimizer  # Unused
  with torch.inference_mode():
    if isinstance(inputs, tuple):
      return model(*inputs)
    return model(inputs)


def _ml_layer_train_step(
    model: torch.nn.Module,
    inputs: Any,
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[torch.nn.Module, Any], Any],
    device: torch.device,
) -> None:
  """Performs a training step (forward + backward) for an ML layer.

  Args:
    model: The ML layer model.
    inputs: The input tensors for the model.
    optimizer: Unused.
    step_fn: The function to call to perform the forward and backward pass.
    device: The device to run the benchmark on.
  """
  del optimizer  # Unused

  loss = step_fn(model, inputs)

  device_utils.synchronize(device.type, loss)
  for p in model.parameters():
    if p.grad is not None:
      device_utils.synchronize(device.type, p.grad)


def get_ml_layer_train_step_function(
    device: torch.device, torch_compile: bool
) -> Callable[[torch.nn.Module, Any, torch.optim.Optimizer], Any]:
  """Returns the benchmark function for training an ML layer.

  Args:
    device: The device to run the benchmark on.
    torch_compile: Whether to compile the training step function using
      `torch.compile`.
  """

  def get_model_train_fn():
    def step_fn(model, inputs):

      if isinstance(inputs, tuple):
        y_pred = model(*inputs)
      else:
        y_pred = model(inputs)
      if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
      loss = torch.mean(y_pred)

      loss.backward()
      return loss.detach()

    if torch_compile:
      return device_utils.torch_compile(step_fn, device.type)
    return step_fn

  model_train_fn = get_model_train_fn()
  return functools.partial(
      _ml_layer_train_step, step_fn=model_train_fn, device=device
  )
