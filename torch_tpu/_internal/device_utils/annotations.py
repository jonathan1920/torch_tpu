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

"""API for providing XLA specific annotations on torch tensors."""

import functools
import torch
from torch import library
from torch_tpu._internal.compile import tpu_torch_compile

custom_op = library.custom_op
register_autograd = library.register_autograd
register_fake = library.register_fake


@custom_op("torch_tpu::layout_hint", mutates_args=())
def _layout_hint_op(x: torch.Tensor, tpu_layout_hint: str) -> torch.Tensor:
  """Eager mode implementation.

  We return a clone to ensure it's treated as a distinct node in the
  graph, but logically it's an identity.

  Args:
    x: The input tensor to annotate.
    tpu_layout_hint: A string identifier for the layout hint.
  """
  res = x.clone()
  tpu_torch_compile.set_layout_hint(res, tpu_layout_hint)
  return res


@register_fake("torch_tpu::layout_hint")
def _layout_hint_fake(
    x: torch.Tensor, tpu_layout_hint: str  # pylint: disable=unused-argument
) -> torch.Tensor:
  """Fake implementation used for tracing.

  Args:
    x: The input tensor that would be annotated.
    tpu_layout_hint: A string identifier for the layout hint.
  """
  return torch.empty_like(x)


def _layout_hint_backward(ctx, grad_output):  # pylint: disable=unused-argument
  # Input 0 was 'x' -> return grad_output
  # Input 1 was 'tpu_layout_hint' -> return None
  return grad_output, None


register_autograd(
    "torch_tpu::layout_hint",
    _layout_hint_backward,
    setup_context=lambda ctx, inputs, output: None,
)


def layout_hint(x: torch.Tensor, tpu_layout_hint: str) -> torch.Tensor:
  """Adds a layout hint to a tensor for the TPU compiler.

  Args:
    x: The input tensor.
    tpu_layout_hint: A string identifier for the layout hint.

  Returns:
    The input tensor with the layout hint attached.

  Example:
      x = layout_hint(x, "{1,0:T(16,128)(2,1)}")
  """

  return torch.ops.torch_tpu.layout_hint.default(x, tpu_layout_hint)


def set_layout(tpu_layout_hint: str):
  """Sets the layout hint for all outputs of a function it decorates."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      outputs = func(*args, **kwargs)

      return _recursive_apply_hint(outputs, tpu_layout_hint)

    return wrapper

  return decorator


def _recursive_apply_hint(out, tpu_layout_hint):
  if isinstance(out, torch.Tensor):
    return layout_hint(out, tpu_layout_hint)
  if isinstance(out, (list, tuple)):
    return type(out)(_recursive_apply_hint(x, tpu_layout_hint) for x in out)
  if isinstance(out, dict):
    return {
        k: _recursive_apply_hint(v, tpu_layout_hint) for k, v in out.items()
    }
  return out
