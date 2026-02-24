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

"""TPU backend for `torch.compile()` integration.

TODO(wzz): These notes are for internal development
The `torch.compile()` function has the following relevant arguments:
  fullgraph (bool):
    We let `torch.compile()` handle graph breaks. We would likely not care.
  dynamic (bool or None): Only False is supported. Dynamic shapes are not
    supported. If a torch.SymInt is encountered in the graph, an error
    will be raised, prompting the user to set dynamic=False.
  [P1]mode (str):
    We are currently only testing the 'default' mode.
    Investigate the other modes with P1
  [p2]options (dict):
    The list is huge.
    The effects of many of these options are still under investigation. P1
  disable (bool):
    Disables the compilation. No ops for us.
"""

import logging
from typing import Any, Callable, List, Sequence, TypeAlias
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.utils import _pytree
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import utils


_ExpectedTypes: TypeAlias = torch.Tensor | torch.nn.Module | torch.SymInt

logger = logging.getLogger(__name__)


def to_device(
    x: Any,
    backend: str | torch.device,
) -> Any:
  """Moves complex objects containing tensors to the specified backend device.

  This function uses pytree to traverse the input object, and find torch.Tensor
  objects and move them to the target backend.

  Args:
    x: The object to move. Can be a torch.Tensor, torch.nn.Module, list, or
      tuple containing tensors.
    backend: The target device to move the tensors to (e.g., 'cpu', 'cuda',
      'tpu').

  Returns:
    The object with all contained tensors moved to the backend device.
  """

  def _to(t: _ExpectedTypes) -> _ExpectedTypes:
    match t:
      case torch.Tensor():
        return t.to(backend)
      case torch.nn.Module():
        return t.to(backend)
      case _:
        return t

  return _pytree.tree_map_only(_ExpectedTypes, _to, x)


def _raise_on_symint(
    x: Any,
) -> Any:
  """We will defer the support of symint for now.

  Here it scans the input object for SymInt. And raises error if found.
  """

  def _raise(t: torch.SymInt):
    raise NotImplementedError(
        "TPU backend: does not support dynamic shape. Please set"
        " torch.compile(..., dynamic=False, ...) and try again."
    )

  return _pytree.tree_map_only(torch.SymInt, _raise, x)


UNSET_GRAPH_HELPER_STR = (
    "FX/MLIR graph is not recorded by default. Turn on TPU backend debug mode"
    " to record it."
)


class _DisableFakeTensorMode:
  """Context manager to temporarily exit FakeTensorMode.

  AOT autograd will trace the model with FakeTensorMode enabled. This converts
  all tensors to FakeTensors which do not have valid storage to avoid
  computation.

  TorchTPU tracing requires all tensors to have valid storage, so is not
  compatible with FakeTensorMode. This applies to placeholder tensors as well,
  fake-tensor will attempt to convert our placeholder tensors into FakeTensors
  before continuing eager tracing.
  """

  def __init__(self):
    self.fake_mode = None

  def __enter__(self):
    # Requires internal API, no public way to temporarily exit fake mode.
    self.fake_mode = torch._guards.active_fake_mode()
    if self.fake_mode and not self.fake_mode.enter_stack:
      # In fake mode, but it is not actively enabled, do nothing.
      self.fake_mode = None
    if self.fake_mode:
      logger.debug("[DisableFakeTensorMode] In fake mode, exiting to compile")
      self.fake_mode.__exit__(None, None, None)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.fake_mode:
      logger.debug("[DisableFakeTensorMode] Reentering fake mode")
      self.fake_mode.__enter__()


class _TorchTpuCompiledExecutable:
  """A callable for a TorchTPU compiled executable.

  This class is returned to dynamo and stored in the dynamo cache. Any time an
  FX graph being traced by dynamo satisfies the guards setup to produce this
  executable, a call to the __call__ method is made.
  """

  def __init__(
      self,
      executable: tpu_torch_compile.PjRtLoadedExecutable,
      map_output_fn: (
          Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] | None
      ),
  ):
    """Initializes the compiled executable wrapper.

    Args:
      executable: The result returned by `tpu_torch_compile.compile_mlir`.
      executable_output_shapes: The output shapes of the executable.
      executable_output_dtypes: The output dtypes of the executable.
      map_output_fn: An optional callable that transforms the output from the
        TPU executable back to the output structure expected by PyTorch. The
        existing use case is to re-insert `None` values into the output list if
        they were filtered out during MLIR conversion.
    """
    self._executable = executable
    self._map_output_fn = map_output_fn
    self._graph_module_debug_str: str | None = None
    self._mlir_graph: str | None = None

  @property
  def graph_module_debug_str(self) -> str | None:
    if self._graph_module_debug_str is None:
      logger.info(UNSET_GRAPH_HELPER_STR)
    return self._graph_module_debug_str

  @graph_module_debug_str.setter
  def graph_module_debug_str(self, value) -> None:
    self._graph_module_debug_str = value

  @property
  def mlir_graph(self) -> str | None:
    if self._mlir_graph is None:
      logger.info(UNSET_GRAPH_HELPER_STR)
    return self._mlir_graph

  @mlir_graph.setter
  def mlir_graph(self, value: str) -> None:
    self._mlir_graph = value

  def __call__(self, *args):
    outputs = tpu_torch_compile.execute(self._executable, args)
    if self._map_output_fn is not None:
      outputs = self._map_output_fn(outputs)

    return outputs


class TpuBackend:
  """TPU backend for torch.compile() integratation."""

  def __init__(
      self,
      debug: bool = False,
  ):
    """Initializes the TPU backend.

    Args:
      debug (bool): If True, enable debug logging and save a dump of the fx
        graph.
    """
    self._debug = debug
    # Stores information about each compiled executable.
    # Organized by order of compilation (index 0 is the first compilation, etc.)
    self._compiled_executables: list[_TorchTpuCompiledExecutable] = []
    if self._debug:
      logger.setLevel(logging.DEBUG)

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
  ) -> Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    # The TorchTPU team decided to defer symint support for now. Here it scans
    # the input object for SymInt. And raises error if found.
    #
    # We should consider ways around this, perhaps we lazily compile in the
    # function invocation when we have arg shapes. Or perhaps we can add a
    # mechanism for backends to notify dynamo that dynamic shapes aren't
    # supported so the user isn't expected to use `dynamic=False` on all compile
    # calls.
    #
    # Per @wan, we might want to remove SymInt from the graph and static
    # recompile to avoid user changes -- this is potentially unsafe since dynamo
    # will store the returned executable in the cache and might not re-dispatch
    # to our backend compiler if future SymInts fit the allowed range. We could
    # consider compiling lazily on each function invocation, but this will have
    # performance impact.
    #
    # TODO(wzz): Figure out how to remove symint from the graph and recompile
    # without dynamism.
    _raise_on_symint(example_inputs)

    logger.info("[TpuBackend] Compiling FX Graph")

    return aot_autograd(
        fw_compiler=self._compile_graph_module,
        # This is to avoid inplace generating graph modules that contains inplace update.
        keep_inference_input_mutations=False,
    )(graph_module, example_inputs)

  def _compile_graph_module(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
  ) -> Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    """Compiles the graph_module with the given inputs for TPU.

    torch.compile() will generate a graph_module and call this function to
    finish
    the compilation.

    Args:
      graph_module: The FX graph module to compile.
      example_inputs: Example inputs to the FX graph for tracing (not the actual
        inputs).

    Returns:
      A function that executes the compiled graph on the TPU.
    """
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
          "[TpuBackend.compile_graph_module] Graph:\n%s",
          graph_module.print_readable(print_output=False),
      )
      logger.debug(
          "[TpuBackend.compile_graph_module] Sample Inputs (len = %d): \n%s",
          len(example_inputs),
          utils.InputMetadata(example_inputs),
      )

    # Exit fake mode to trace, see _DisableFakeTensorMode docstring for details.
    with _DisableFakeTensorMode():
      # Convert example_inputs to placeholders. This is done to:
      # (1) Prevent the unintentional compilation of deferred operations.
      #     placeholders will error.
      # (2) Act as TPU-compatible FakeTensors, so that tracing does not depend
      #     on tensor data.
      placeholder_args = [
          tpu_torch_compile.placeholder_like(arg)
          if isinstance(arg, torch.Tensor)
          else arg
          for arg in example_inputs
      ]

      # Convert the ATen greph to StableHLO
      # Use serialized format by default since it is much more compact.
      print_config = torch_tpu_export.MlirPrintConfig.MLIR_SERIALIZED
      if self._debug:
        print_config = torch_tpu_export.MlirPrintConfig.MLIR_DEBUG_INFO

      mlir_graph, _, map_output_fn = torch_tpu_export.fx_to_mlir(
          graph_module, placeholder_args, print_config=print_config
      )

    cached_executable = tpu_torch_compile.compile_mlir(mlir_graph)

    executable = _TorchTpuCompiledExecutable(
        executable=cached_executable,
        map_output_fn=map_output_fn,
    )

    self._compiled_executables.append(executable)

    if self._debug:
      # Do not use print_readable() as it include original line of code which is
      # too verbose.
      executable.graph_module_debug_str = str(graph_module.code)
      executable.mlir_graph = mlir_graph

    # If there are non-tensor inputs (e.g. concrete ints from dynamic shapes),
    # wrap the executable to filter them out at call time since
    # tpu_torch_compile.execute() only accepts tensors.
    has_non_tensor_inputs = any(
        not isinstance(arg, torch.Tensor) for arg in example_inputs
    )
    if has_non_tensor_inputs:
      return _TensorFilterExecutable(executable)

    return executable


class _TensorFilterExecutable:
  """Wraps a compiled executable to filter out non-tensor args at call time.

  Used when aot_autograd passes concrete integers alongside tensors.
  Implemented as a class (not a closure) to support pickling.
  """

  def __init__(self, executable: _TorchTpuCompiledExecutable):
    self._executable = executable

  def __call__(self, *args):
    tensor_args = tuple(a for a in args if isinstance(a, torch.Tensor))
    return self._executable(*tensor_args)

  def __reduce__(self):
    return (type(self), (self._executable,))
