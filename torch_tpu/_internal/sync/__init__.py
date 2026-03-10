# Copyright 2025 Google LLC
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
"""Utility functions for materializing and synchronizing TPU tensors.

In eager mode, tensors in torch_tpu are sometimes deferred, meaning that they
have a pending operation that needs to be compiled and executed before their
data is available.

Accessing the data of a deferred tensor (through .cpu(), .item(), or similar)
always forces the following steps:
  1. Graph traversal: the deferred operations graph is analyzed and prepared
  2. Compilation: the graph is compiled into executable code
  3. Materialization: the execution is scheduled, and a PjRtBuffer is created
     to hold the future result
  4. Synchronization: waiting for the running executable to complete
  5. Transfer: copying data from device to host

However, in some cases it can be useful to only materialize and/or synchronize.
  * Manual materialization splits the deferred op graph, resulting in multiple
    smaller compilation steps. This can reduce compilation time, but may result
    in a less efficient executable.
  * Manual materialization can be used to insert breakpoints in loops, reusing
    a cached compilation across each iteration.
  * Manual materialization can be used to profile compilation time, while
    avoiding the overhead of data transfer and execution.
  * Manual synchronization can be used to profile the combined compilation and
    execution time; or, if the compilation is cached, the execution time alone.

These are not intended as the primary interface for users. These are currently
only used internally for performance evaluation by the torch_tpu developers.
"""

from collections.abc import Callable, Sequence
import functools
from typing import Any, Optional, Union
import torch
from torch_tpu._internal.sync import sync


computation_graphviz = sync.computation_graphviz
computation_mlir = sync.computation_mlir
dump_computation_graphviz = sync.dump_computation_graphviz
synchronize = sync.synchronize
is_materialized = sync.is_materialized
is_ready = sync.is_ready
is_bufferless_zero_size = sync.is_bufferless_zero_size

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "computation_graphviz",
    "computation_mlir",
    "dump_computation_graphviz",
    "is_bufferless_zero_size",
    "is_materialized",
    "is_ready",
    "synchronize",
    # go/keep-sorted end
]

# Note: this is using the term "outputs" and "inputs" in the same way as
# torch.autograd.backward and torch.autograd.grad. For a standard model training
# this means the loss is the output, and the parameters are the inputs.
# This is the **inverse** of the naming convention used in
# torch/torch/csrc/autograd/engine.cpp which uses "inputs" to mean the roots
# of the backpropagation graph (the loss) and "outputs" to mean the leaves
# (the parameters).
Outputs = Union[tuple[torch.Tensor], tuple[torch.autograd.graph.GradientEdge]]


class PatchedImperativeEngine(torch._C._ImperativeEngine):  # pylint: disable=protected-access
  """Patched ImperativeEngine trigger TPU synchronization after backward."""

  def run_backward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
    outputs: Outputs = args[0]
    _retain_outputs(outputs)
    del outputs

    result = super().run_backward(*args, **kwargs)
    del args, kwargs
    _sync_retained_outputs()
    return result


def _can_sync_tensor(t: torch.Tensor) -> bool:
  # Fake tensors don't have any data that can be synced
  return not torch._subclasses.fake_tensor.is_fake(t)  # pylint: disable=protected-access


# Install the patched engine.
torch.autograd.Variable._execution_engine = PatchedImperativeEngine()  # pylint: disable=protected-access

retained_outputs: list[torch.Tensor] = []


def _retain_outputs(outputs: Outputs):
  """Stores outputs in a global so they can be referenced post-backward."""
  global retained_outputs  # pylint: disable=global-variable-not-assigned
  for o in outputs:
    if isinstance(o, torch.Tensor):
      retained_outputs.append(o)


def _sync_retained_outputs():
  """Synchronizes the loss and all leaf tensors after the backward pass.

  In TorchTPU's eager mode, operations are often deferred. This can lead to
  memory pressure if large activation or activation-gradient graphs are kept in
  memory because the final parameter gradients haven't been materialized yet.

  By calling synchronize() after the backward pass completes,
  we force XLA to compile and execute all pending operations. This allows
  PyTorch to release intermediate tensors that are no longer needed, reducing
  the memory footprint of training loops.
  """
  global retained_outputs
  sync.synchronize(
      [t for t in retained_outputs if _can_sync_tensor(t)], wait=False
  )
  retained_outputs = []
