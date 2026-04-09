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
import dataclasses
import enum
import functools
from typing import Any, Callable, List, Sequence, Tuple
from absl import logging
import torch
import torch.export
from torch.fx import node
import torch.utils._pytree as pytree
from torch_tpu._internal import device_utils
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile

__all__ = [
    "MlirPrintConfig",
    "ExportedMlir",
    "exported_to_mlir",
    "fx_to_mlir",
]


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


class MlirPrintConfig(enum.Enum):
  """Configuration for printing MLIR."""

  MLIR_PRETTY = "MlirPretty"
  MLIR_DEBUG_INFO = "MlirDebugInfo"
  MLIR_SERIALIZED = "MlirSerialized"
  MLIR_SERIALIZED_VERSIONED = "MlirVersionedSerialized"


@dataclasses.dataclass(frozen=True)
class ExportedMlir:
  """Represents the MLIR representation of an FX graph module.

  Attributes:
    mlir_bytes: The MLIR module in bytecode or text format.
    mlir_result_tensors: A flattened list of tensors that the MLIR graph will
      actually produce as outputs.
    reconstruct_fx_outputs_fn: A function that takes the flattened MLIR outputs
      and reconstructs the original FX graph's output structure (e.g., restoring
      `None` values or nested tuples).
  """

  mlir_bytes: bytes
  mlir_result_tensors: List[torch.Tensor]
  reconstruct_fx_outputs_fn: Callable[[Sequence[torch.Tensor]], Any]


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
    logging.debug(
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
) -> ExportedMlir:
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
    An `ExportedMlir` object containing the MLIR representation of the graph and
    FX output reconstruction information.
  """

  sample_args = _extract_sample_arguments(exported)
  _, state = _extract_states_from_exported_program(exported)

  args = state + sample_args
  args = [tpu_torch_compile.placeholder_like(arg) for arg in args]

  device = device_utils.available_xla_device()
  if device is None:
    raise ValueError(
        "export requires an XLA device to be initialized before use"
    )

  to_xla = lambda x: x.to(device)
  args = pytree.tree_map_only(torch.Tensor, to_xla, args)
  module = exported.graph_module.to(device)

  return fx_to_mlir(module, args=args, print_config=print_config)


def _reconstruct_fx_outputs(
    deduped_outputs: Sequence[torch.Tensor],
    output_indices: List[int | None],
    spec: pytree.TreeSpec,
) -> Any:
  """Restores the original FX output structure from flattened MLIR outputs."""
  reconstructed_flat: List[torch.Tensor | None] = []
  output_used = [False] * len(deduped_outputs)
  for idx in output_indices:
    if idx is None:
      reconstructed_flat.append(None)
    elif output_used[idx]:
      reconstructed_flat.append(deduped_outputs[idx].clone())
    else:
      reconstructed_flat.append(deduped_outputs[idx])
      output_used[idx] = True

  return pytree.tree_unflatten(reconstructed_flat, spec)


def _process_fx_outputs(
    outputs: Any,
) -> Tuple[
    List[torch.Tensor],
    Callable[[Sequence[torch.Tensor]], Any],
]:
  """Removes Nones and dedups outputs.

  This is necessary because MLIR graphs do not accept `None` in outputs, and
  we don't want to produce duplicate outputs for tensors that are the same. This
  function filters `None` values and reduces to unique tensors. It returns a
  tuple containing a list of result tensors and a function to reconstruct
  the original FX output structure.

  Args:
    outputs: The outputs of the graph.

  Returns:
    A tuple containing a list of result tensors and a function to reconstruct
    the original FX output structure.

  Raises:
    TypeError: If an element in the flattened outputs is not a `torch.Tensor`
      or `None`.
  """
  flat_outputs, spec = pytree.tree_flatten(outputs)
  deduped_outputs: List[torch.Tensor] = []
  output_indices: List[int | None] = []
  # We deduplicate outputs based on a composite key: (data_ptr, dtype, shape).
  # - data_ptr(): Identifies the starting memory address.
  # - dtype: Differentiates views of the same memory interpreted as different
  #   types.
  # - shape: Differentiates views of the same memory with different shapes
  #   (e.g., overlapping slices).
  #
  # This prevents false deduplication where different views of the same storage
  # share the same base address but represent different data structures (e.g.,
  # overlapping views from `unbind` or complex tensor operations).
  #
  # Note: If two slices of the same buffer have the same dtype and shape but
  # different offsets, their data_ptr() will be different (since data_ptr()
  # includes the storage offset), so they will not be falsely deduplicated.
  index_by_dedupe_key: dict[tuple[int, torch.dtype, torch.Size], int] = {}

  for item in flat_outputs:
    if item is None:
      output_indices.append(None)
    elif isinstance(item, torch.Tensor):
      ptr = item.data_ptr()
      key = (ptr, item.dtype, item.shape)
      if key not in index_by_dedupe_key:
        index_by_dedupe_key[key] = len(deduped_outputs)
        deduped_outputs.append(item)
      output_indices.append(index_by_dedupe_key[key])
    else:
      raise TypeError(
          f"Expect FX graph output to be a Tensor or None, got {item}"
      )

  return deduped_outputs, functools.partial(
      _reconstruct_fx_outputs,
      output_indices=output_indices,
      spec=spec,
  )


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
    args: List[torch.Tensor | Any],
    print_config: MlirPrintConfig = MlirPrintConfig.MLIR_PRETTY,
    donate_args: Sequence[int] | None = None,
) -> ExportedMlir:
  """Converts an FX graph module to MLIR using TorchTPU's defer mode.

  This function traces the given FX graph module by running it with an
  EagerLikeFxInterpreter in full defer mode
  (`execution_mode.EagerMode.DEFER_ALL`).
  The
  resulting deferred graph is then converted to MLIR bytes.

  Args:
    module: The `torch.fx.GraphModule` to be converted to MLIR.
    args: A list of input arguments to trace the module. These will be run
      through an FX graph interpreter to identify the graph's output tensors.
    print_config: The desired MLIR output format.
    donate_args: The list of argument indices that are allowed to be donated.

  Returns:
    An `ExportedMlir` object containing the MLIR representation of the graph and
    FX output reconstruction information.
  """
  # Filter out non-tensor arguments.
  argument_tensors = []
  tensor_idx_map = {}
  for i, arg in enumerate(args):
    if isinstance(arg, torch.Tensor):
      tensor_idx_map[i] = len(argument_tensors)
      argument_tensors.append(arg)

  # Remap donate_args to be indices into argument_tensors.
  if donate_args:
    donate_args = [
        tensor_idx_map[i] for i in donate_args if i in tensor_idx_map
    ]
  else:
    donate_args = []
  del tensor_idx_map

  # Sync the RNG state before FX tracing to force materialization of deferred
  # ops.
  #
  # Context: TorchTPU uses "deferred execution". Operations aren't
  # computed instantly. Instead, they are added to a deferred graph.
  # RNG state is treated as just another tensor in this graph.
  #
  # The "Graph Leaking" Problem:
  # If a module was run in eager mode previously (e.g., a warmup step), the
  # current global RNG state might be an unexecuted node sitting at the end of
  # that old graph. If FX tracing accesses this state (e.g., via dropout
  # kernels) without syncing, it accidentally connects your clean export graph
  # to the old eager execution graph.
  #
  # Illustration of the leak:
  # [Old Eager Graph] ---> (Pending RNG State) <--- [New FX Tracing Graph]
  #                               ^
  #                  FX grabs this, merging both graphs!
  #                  (Leads to bloat and validation errors)
  #
  # The Solution:
  # Syncing forces the TPU to compute the old graph, turning the RNG state
  # back into a concrete, standalone tensor. This gives the FX tracer a
  # clean slate.
  #
  # Safe state after sync:
  # [Old Graph Executed] | (Concrete RNG State) ---> [Clean FX Tracing Graph]
  if argument_tensors:
    sync.synchronize(
        torch.get_device_module(argument_tensors[0].device).get_rng_state(),
        wait=True,
    )

  # Run the module through the EagerLikeFxInterpreter with MLIR location
  # tracebacks enabled so that the MLIR we generate has file location info.
  with execution_mode.eager_mode(
      execution_mode.EagerMode.INTERNAL_DEFER_ALL
  ), enable_tracebacks():
    # We clone the args so that inplace updates do not overwrite the placeholder
    # args, these copies will be removed in the compiled code so there is no
    # performance impact.
    # Remove once b/491716758 is implemented.
    cloned_args = (
        x.clone() if isinstance(x, torch.Tensor) else x for x in args
    )
    fx_outputs = EagerLikeFxInterpreter(module).run(*cloned_args)

  result_tensors, reconstruct_fx_outputs_fn = _process_fx_outputs(fx_outputs)

  mlir_bytes = tpu_torch_compile.build_mlir(
      result_tensors=result_tensors,
      argument_tensors=argument_tensors,
      print_config=print_config.value,
      donate_args=donate_args,
  )

  return ExportedMlir(
      mlir_bytes=mlir_bytes,
      mlir_result_tensors=result_tensors,
      reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
  )
