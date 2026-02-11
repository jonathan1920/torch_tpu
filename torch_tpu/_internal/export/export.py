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

"""TorchTPU support for torch.export to MLIR."""

import contextlib
import copy
import enum
from functools import partial
import logging
from typing import Any, Callable, List, Sequence, Tuple

import torch
import torch.export
from torch.fx import node
from torch.utils import _pytree
from torch_tpu._internal import device_utils
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile

logger = logging.getLogger(__name__)


def _extract_states_from_exported_program(exported_model):
  """Return a list of parameters that were lifted during export."""

  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = list(
      exported_model.graph_signature.parameters
  ) + list(exported_model.graph_signature.buffers)
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, "constants", None)) is not None:
    state_dict.update(constants)
  param_buffer_values = list(state_dict[key] for key in param_and_buffer_keys)

  if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
    for name in exported_model.graph_signature.lifted_tensor_constants:
      param_buffer_values.append(exported_model.tensor_constants[name])

  return param_and_buffer_keys, param_buffer_values


def _extract_sample_arguments(exported: torch.export.ExportedProgram):
  """Return a list of sample arguments for the given exported program."""

  def _to_aval(arg_meta):
    """Convert from torch type to TPU meta buffer for tracing."""
    val = arg_meta["val"]
    is_scalar = not isinstance(val, torch.Tensor)
    if is_scalar:
      return tpu_torch_compile.placeholder([], type(arg_meta["val"]), False)

    tensor_meta = arg_meta["tensor_meta"]
    # tensor_meta is a torch.TensorMetadata, which is a NamedTuple and not
    # and actual tensor.
    # See torch/fx/passes/shape_prop.py
    return tpu_torch_compile.placeholder(
        list(tensor_meta.shape),
        tensor_meta.dtype,
        tensor_meta.requires_grad,
    )

  def _get_inputs(exported):
    """Return placeholders with input metadata."""
    placeholders = [p for p in exported.graph.nodes if p.op == "placeholder"]
    input_placeholders = [
        p
        for p, s in zip(placeholders, exported.graph_signature.input_specs)
        if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
    ]
    return input_placeholders

  args = _get_inputs(exported)
  return [_to_aval(arg.meta) for arg in args]


def _to_list(obj: Any) -> List[Any]:
  """Ensures a return value is a list."""
  if isinstance(obj, list):
    return obj
  if isinstance(obj, tuple):
    return list(obj)
  return [obj]


class MlirPrintConfig(enum.Enum):
  """Configuration for printing MLIR."""

  MLIR_PRETTY = "MlirPretty"
  MLIR_DEBUG_INFO = "MlirDebugInfo"
  MLIR_SERIALIZED = "MlirSerialized"
  MLIR_SERIALIZED_VERSIONED = "MlirVersionedSerialized"


class EagerLikeFxInterpreter(torch.fx.Interpreter):
  """Interpreter for TorchTPU.

  Interprets each node in the FX graph using the eager plugin with some
  additional setup so that torch.compile and torch.export produce similar MLIR
  to torch eager.
  """

  def __init__(
      self,
      module: torch.fx.GraphModule,
  ):
    super().__init__(module)
    self.state_dict = module.state_dict()

  def _set_func_code_to_node_source_locations(self, func, node):
    """Update the function's code locations to the node's source locations.

    Overwrite the function's code locations to the node's source locations
    so that this node original source info appears on the python stack,
    making compiled/exported source locations resemble eager's source locations.
    """
    if not node.stack_trace:
      return

    # Note: This uses internal APIs and may break in the future.
    try:
      pt_trace = torch.fx.graph._parse_stack_trace(node.stack_trace)
      if pt_trace is None:
        # We will catch if something changes and this fails with unittests
        return

      # Overwrite function with node's source location info.
      # This is fairly unsafe, but it will set the python __code__ object that
      # is used in the captured stack trace to the node's source location, so it
      # will appear more like we were eagerly evaluating the function.
      func.__code__ = func.__code__.replace(
          co_filename=pt_trace.file,
          co_firstlineno=int(pt_trace.lineno),
          co_name=pt_trace.name,
      )
    except Exception as e:
      logging.warning("Failed to set func code to node source locations: %s", e)
    return

  def get_attr(
      self,
      target: node.Target,
      args: tuple[node.Argument, ...],
      kwargs: dict[str, Any],
  ) -> Any:
    logger.debug(
        "[EagerLikeFxInterpreter::get_attr] target: %s, args: %s, kwargs: %s",
        target,
        args,
        kwargs,
    )
    if isinstance(target, str) and target.startswith("_tensor_constant"):
      # Embed the _tensor_constant in the graph as a scalar constant
      scalar: torch.Tensor = self.state_dict[target]
      # TODO: Add empty.fill_ support for 0-d tensor constants
      if scalar.shape.numel() != 1:
        return super().get_attr(target, args, kwargs)
      # Scalar constants are always on CPU but need to be on the active XLA device.
      device = device_utils.available_xla_device()
      if device is None:
        raise RuntimeError(
            "the TPU backend must be initialized before compiling ops; please"
            " call torch.tpu.api.tpu_device() first"
        )
      return torch.empty(scalar.shape, device=device, dtype=scalar.dtype).fill_(
          scalar.item()
      )

    # Fallback to the default implementation.
    return super().get_attr(target, args, kwargs)

  def run_node(self, node) -> Any:
    """(Override) Run a single node in the FX graph, adding source locations."""

    def dispatch_fx_node(node):
      return super(EagerLikeFxInterpreter, self).run_node(node)

    self._set_func_code_to_node_source_locations(dispatch_fx_node, node)
    return dispatch_fx_node(node)


def exported_to_mlir(
    exported: torch.export.ExportedProgram,
    print_config: MlirPrintConfig = MlirPrintConfig.MLIR_DEBUG_INFO,
) -> bytes:
  """Converts a `torch.export.ExportedProgram` into its MLIR representation.

  This function serves as a high-level wrapper around `fx_to_mlir` for exported
  models. It extracts the FX graph, sample arguments, and lifted states
  (parameters, buffers, and constants) from the `ExportedProgram`, moves them
  to an available XLA device, and then generates the corresponding MLIR.

  An XLA device must be initialized before calling this function.

  Args:
    exported: The `torch.export.ExportedProgram` to convert.
    print_config: An enum specifying the format of the output MLIR, such as
      pretty-printed or with debug info.

  Returns:
    The MLIR representation of the graph as a bytes string.
  """

  args = [
      tpu_torch_compile.placeholder_like(arg)
      for arg in _extract_sample_arguments(exported)
  ]
  _, state = _extract_states_from_exported_program(exported)
  args = state + args

  device = device_utils.available_xla_device()
  if device is None:
    raise ValueError(
        "export requires an XLA device to be initialized before use"
    )

  to_xla = lambda x: x.to(device)
  args = _pytree.tree_map_only(torch.Tensor, to_xla, args)
  module = exported.graph_module.to(device)

  result, _, _ = fx_to_mlir(module, args=args, print_config=print_config)
  return result


def _remove_none_from_outputs(
    outputs: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor] | None]:
  """Removes None values from a list of outputs and generates a map to restore them.

  This is necessary because MLIR graphs does not accept `None` in outputs. This
  function filters `None` values from the output list and produces a map that
  canbe used by `_map_output` to reconstruct the original list, including `None`
  values, after MLIR processing is complete.

  Args:
    outputs: A list of tensors, potentially containing `None`.

  Returns:
    A tuple `(non_none_outputs, output_none_map)`:
    - `non_none_outputs`: A list containing only the non-`None` tensors from
      the input `outputs`.
    - `output_none_map`: A list used for reconstruction. Each element is
      either `None` (if the original output at this position was `None`) or an
      integer index into `non_none_outputs` (if the original output was a
      tensor). If `outputs` is empty, this value is `None`.
  """
  output_none_map = []
  non_none_outputs = []
  node_index_counter = 0
  for item in outputs:
    if item is None:
      output_none_map.append(None)
    else:
      output_none_map.append(node_index_counter)
      non_none_outputs.append(item)
      node_index_counter += 1
  return non_none_outputs, output_none_map if output_none_map else None


def _map_output(
    outputs: Sequence[torch.Tensor],
    output_none_map: Sequence[torch.Tensor] | None,
) -> Sequence[torch.Tensor]:
  """Reconstructs the original output sequence.

  This function uses the `output_none_map` generated by
  `_remove_none_from_outputs` to re-insert `None` values into a sequence of
  non-`None` tensors, restoring the original output structure.

  Args:
    outputs: A sequence containing only non-`None` tensors.
    output_none_map: A sequence where items are either an index into `outputs`
      or `None`. If an item is `None`, `None` is inserted at that position in
      the result; otherwise, the item is an index used to retrieve the tensor
      from `outputs`. If `output_none_map` itself is `None`, `outputs` is
      returned unchanged.

  Returns:
    The reconstructed sequence of tensors, potentially containing `None`.
  """
  if output_none_map is not None:
    outputs = tuple(
        outputs[map_item] if map_item is not None else None
        for map_item in output_none_map
    )
    assert len(outputs) > 0, "Output map must have at least one output"
  return outputs


@contextlib.contextmanager
def enable_tracebacks():
  """A context manager that enables MLIR location tracebacks."""
  prev_tracebacks_enabled = tpu_torch_compile.get_mlir_tracebacks_enabled()
  tpu_torch_compile.set_mlir_tracebacks_enabled(True)
  try:
    yield
  finally:
    tpu_torch_compile.set_mlir_tracebacks_enabled(prev_tracebacks_enabled)


def fx_to_mlir(
    module: torch.fx.GraphModule,
    args: List[torch.Tensor],
    print_config: MlirPrintConfig = MlirPrintConfig.MLIR_PRETTY,
) -> Tuple[
    bytes,
    List[torch.Tensor],
    Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] | None,
]:
  """Converts an FX graph module to MLIR using TorchTPU's defer mode.

  This function traces the given FX graph module by running it with an
  EagerLikeFxInterpreter in full defer mode (`execution_mode.DeferMode.ALL`).
  The
  resulting deferred graph is then converted to MLIR bytes.

  Args:
    module: The `torch.fx.GraphModule` to be converted to MLIR.
    args: A list of input tensors to trace the module.
    print_config: The desired MLIR output format.
    allow_output_modification: If True, filters `None` from outputs and returns
      a function to reconstruct them. If False, `None` outputs are not
      permitted.

  Returns:
    A tuple `(mlir_bytes, output_map_fn)` where:
    - `mlir_bytes`: The MLIR representation of the graph as bytes.
    - `output_tensors`: The output tensors from the MLIR trace, with `None`
      values removed.
    - `output_map_fn`: A callable `fn(outputs) -> outputs` that reconstructs
      the original output sequence from `output_tensors` by re-inserting `None`
      values.
  """
  # Run the module through the EagerLikeFxInterpreter with MLIR location
  # tracebacks enabled so that the MLIR we generate has file location info.
  with execution_mode.defer_mode(
      execution_mode.DeferMode.ALL
  ), enable_tracebacks():
    tpu_outputs = EagerLikeFxInterpreter(module).run(*args)

  tpu_outputs, output_none_map = _remove_none_from_outputs(tpu_outputs)
  output_none_map_fn = partial(_map_output, output_none_map=output_none_map)
  mlir_bytes = tpu_torch_compile.build_mlir(
      result_tensors=_to_list(tpu_outputs),
      argument_tensors=_to_list(args),
      print_config=print_config.value,
  )

  return (
      mlir_bytes,
      tpu_outputs,
      output_none_map_fn,
  )
