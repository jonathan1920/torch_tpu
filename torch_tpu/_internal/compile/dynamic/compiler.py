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

"""Compiler for handling dynamic shape in torch.compile()."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import numbers
from typing import Any, NamedTuple
from absl import logging
import torch
from torch._inductor.utils import InputType
from torch._logging import LazyString
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.dynamic.graph_transformations import apply_dynamism_transformations
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager
from torch_tpu._internal.compile.dynamic.symbol_bounds import get_symint_bounds


def _get_example_inputs(
    example_inputs: Sequence[Any], sym_shape_manager: SymShapeManager
) -> list[torch.Tensor | int]:
  """Specializes dynamic inputs to their static upper bounds.

  Args:
    example_inputs: The example inputs to the FX graph.
    sym_shape_manager: Symbolic shape manager to get the dynamic shape
      information.

  Returns:
    A list of updated example inputs with SymInts in inputs replaced by
    their upper bounds. The updated example inputs also include
    any new runtime size placeholders introduced for dynamic tensors and
    generative ops.

  Example:
    example_inputs = [SymInt(s0), tensor(s0, 2)]  # s0 <= 10

    Returns:
      updated_example_inputs = [
          10,  # s0 -> upper bound
          tensor(10, 2),  # tensor with upper bounds
          tensor(), # runtime size placeholder for s0
      ]
  """
  updated_example_inputs = []
  for index, arg in enumerate(example_inputs):
    if isinstance(arg, torch.Tensor):
      num_dynamic_dims = sym_shape_manager.get_num_dynamic_dims(index)
      if num_dynamic_dims == 0:
        updated_example_inputs.append(arg)
        continue
      bounded_shape = sym_shape_manager.get_bounded_shape(index)
      updated_example_inputs.append(
          torch.zeros(
              bounded_shape,
              dtype=arg.dtype,
              requires_grad=arg.requires_grad,
          )
      )
      # The dynamic shape info stores upper bounds for the dynamic dimensions.
      # We add a placeholder for each dynamic dimension to be used by the
      # set_dimension_size operation.
      dynamic_bounds = sym_shape_manager.input_tensors_metadata[
          index
      ].dynamic_bounds
      for _, upper_bound in dynamic_bounds:
        updated_example_inputs.append(
            torch.tensor(upper_bound, dtype=torch.int32)
        )
    elif isinstance(arg, torch.SymInt):
      _, upper = get_symint_bounds(arg)
      updated_example_inputs.append(upper)
      # For generative ops, we may create a new placeholder for the symint
      # that is passed to set_dimension_size as an argument.
      if str(arg) in sym_shape_manager.symint_to_placeholder:
        updated_example_inputs.append(torch.tensor(upper, dtype=torch.int32))
    else:
      updated_example_inputs.append(arg)

  return updated_example_inputs


class _TensorInfo(NamedTuple):
  """Tensor information.

  Attributes:
    shape: The static shape of the tensor.
    dtype: The data type of the tensor.
  """

  shape: list[int]
  dtype: torch.dtype


class _ShapeBoundInfo(NamedTuple):
  """Shape bound information for dynamic dimensions of a tensor.

  Attributes:
    dynamic_dims: A list of indices of the dynamic dimensions.
    upper_bounds: A list of upper bounds for each corresponding dynamic
      dimension.
  """

  dynamic_dims: list[int]
  upper_bounds: list[int]


def _compile_and_execute_slice_subgraph(
    tensor_outputs: Sequence[torch.Tensor],
    output_shapes: Sequence[list[int]],
) -> list[torch.Tensor]:
  """Compiles and executes the slice subgraph."""
  target_shapes = []
  padded_shapes = []
  input_scalar_types = []

  for tensor, target_shape in zip(tensor_outputs, output_shapes):
    target_shapes.append(target_shape)
    padded_shapes.append(list(tensor.shape))
    input_scalar_types.append(tensor.dtype)

  logging.debug(
      "[DynamicTpuBackend] Compile Slice Subgraph, target_shapes: %s,"
      " padded_shapes: %s",
      LazyString(lambda: str(target_shapes)),
      LazyString(lambda: str(padded_shapes)),
  )

  mlir_module = tpu_torch_compile.get_slice_module_mlir(
      target_shapes, padded_shapes, input_scalar_types
  )
  logging.debug(
      "[DynamicTpuBackend] MLIR slice module: %s",
      LazyString(lambda: tpu_torch_compile.serialize_mlir_text(mlir_module)),
  )
  return tpu_torch_compile.execute(
      tpu_torch_compile.compile_mlir(mlir_module), tensor_outputs
  )


def _compute_output_shapes(
    sym_shape_manager: SymShapeManager,
    args: Sequence[Any],
) -> list[list[int]] | None:
  """Computes the runtime output shapes from symbolic shapes and arguments."""
  if not sym_shape_manager.outputs_sym_shape:
    return None

  output_shapes = [
      output_sym_shape.get_output_runtime_shape(args)
      for output_sym_shape in sym_shape_manager.outputs_sym_shape
  ]
  logging.debug(
      "[_DynamicTpuCompiledExecutable] Output shapes: %s", output_shapes
  )
  return output_shapes


class _DynamicTpuCompiledExecutable(compiler.CompiledArtifact):
  """A callable wrapper for dynamic MLIR programs with TPU executable."""

  def __init__(
      self,
      model_executable: Any,
      sym_shape_manager: SymShapeManager,
  ):
    self.model_executable = model_executable
    self.sym_shape_manager = sym_shape_manager
    self._precomputed_bounds_list = self._precompute_bounds_list()
    self._scalar_tensor_cache = {}
    self._backend_device = None
    self._dynamic_scalar_indices = self._precompute_dynamic_scalar_indices()

  def _precompute_dynamic_scalar_indices(self) -> set[int]:
    indices = set()
    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if isinstance(arg, torch.SymInt):
        if str(arg) in self.sym_shape_manager.symint_to_placeholder:
          indices.add(idx)
    return indices

  def _get_backend_device(self) -> torch.device:
    if self._backend_device is None:
      self._backend_device = torch.accelerator.current_accelerator()
    return self._backend_device

  def _get_cached_scalar_tensor(
      self, val: int, dtype: torch.dtype, device: torch.device
  ) -> torch.Tensor:
    key = (val, dtype, device)
    tensor = self._scalar_tensor_cache.get(key)
    if tensor is None:
      tensor = torch.tensor(val, dtype=dtype, device=device)
      self._scalar_tensor_cache[key] = tensor
    return tensor

  def _get_pad_subgraph_inputs(
      self,
      args: Sequence[Any],
  ) -> tuple[list[torch.Tensor], list[_TensorInfo]]:
    """Returns inputs for the pad subgraph."""
    tensor_args = []
    tensor_info = []

    backend_device = self._get_backend_device()

    for idx, arg in enumerate(args):
      if arg is None:
        continue

      if isinstance(arg, numbers.Integral):
        if idx in self._dynamic_scalar_indices:
          tensor_val = self._get_cached_scalar_tensor(
              arg, dtype=torch.int32, device=backend_device
          )
          tensor_args.append(tensor_val)
          tensor_info.append(_TensorInfo(shape=[], dtype=torch.int32))
        continue

      # Handle any other missing type above
      assert isinstance(arg, torch.Tensor)

      if arg.device.type == "cpu":
        raise ValueError(
            "CPU tensors are not supported in dynamic shapes compilation,"
            " please move inputs to TPU."
        )

      tensor_args.append(arg)
      static_shape = list(arg.shape)
      tensor_info.append(_TensorInfo(shape=static_shape, dtype=arg.dtype))

    return tensor_args, tensor_info

  def _compile_and_execute_pad_subgraph(
      self,
      args: Sequence[Any],
  ) -> list[torch.Tensor]:
    """Compiles and executes the pad subgraph."""
    tensor_args, tensor_info = self._get_pad_subgraph_inputs(args)

    logging.debug(
        "[DynamicTpuBackend] Compile Pad Subgraph, tensor_info: %s,"
        " bounds_list: %s",
        tensor_info,
        self._precomputed_bounds_list,
    )

    # Get the MLIR module for the pad subgraph.
    mlir_module = tpu_torch_compile.get_pad_module_mlir(
        tensor_info, self._precomputed_bounds_list
    )
    logging.debug(
        "[DynamicTpuBackend] MLIR pad module: %s",
        LazyString(lambda: tpu_torch_compile.serialize_mlir_text(mlir_module)),
    )
    # Compile the pad subgraph to a PJRT executable.
    executable = tpu_torch_compile.compile_mlir(mlir_module)

    # Execute the pad subgraph with the input tensors and get the
    # padded tensors.
    return tpu_torch_compile.execute(executable, tensor_args)

  def _precompute_bounds_list(self) -> list[_ShapeBoundInfo]:
    bounds_list = []
    for idx, arg in enumerate(self.sym_shape_manager.example_inputs):
      if arg is None:
        continue

      if isinstance(arg, torch.SymInt):
        if str(arg) in self.sym_shape_manager.symint_to_placeholder:
          bounds_list.append(_ShapeBoundInfo(dynamic_dims=[], upper_bounds=[]))
        continue

      assert isinstance(arg, torch.Tensor)
      tensor_metadata = self.sym_shape_manager.input_tensors_metadata[idx]
      if not tensor_metadata.dynamic_dims:
        bounds_list.append(_ShapeBoundInfo(dynamic_dims=[], upper_bounds=[]))
        continue
      bounds_list.append(
          _ShapeBoundInfo(
              dynamic_dims=tensor_metadata.dynamic_dims,
              upper_bounds=[
                  upper for _, upper in tensor_metadata.dynamic_bounds
              ],
          )
      )
    return bounds_list

  def __call__(self, *args: Any) -> Any:
    logging.debug("[DynamicTpuBackend] Execute Model")

    if not args:
      # If no args are passed, we assume there is no dynamic shape and we can
      # directly execute the model executable.
      return self.model_executable(list(args))

    # Compile and run pad executable to get statically padded tensors
    # and runtime size tensors
    pad_outputs = self._compile_and_execute_pad_subgraph(args)

    # Run model executable with pads, sizes, and explicit output shapes
    outputs = self.model_executable(list(pad_outputs))

    output_shapes = _compute_output_shapes(self.sym_shape_manager, args)

    # If the output has dynamic shape, we need to slice it back
    if output_shapes is not None:
      flat_outputs, spec = _pytree.tree_flatten(outputs)
      tensor_outputs = []
      tensor_indices = []
      target_shapes = []

      for idx, (output_tensor, target_shape) in enumerate(
          zip(flat_outputs, output_shapes)
      ):
        if isinstance(output_tensor, torch.Tensor):
          tensor_outputs.append(output_tensor)
          tensor_indices.append(idx)
          target_shapes.append(target_shape)

      if tensor_outputs:
        sliced_tensors = _compile_and_execute_slice_subgraph(
            tensor_outputs, target_shapes
        )
        for idx, sliced_tensor in zip(tensor_indices, sliced_tensors):
          flat_outputs[idx] = sliced_tensor

      return _pytree.tree_unflatten(flat_outputs, spec)
    return outputs

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    # TODO(b/903508278): Add support for pickling to DynamicCompiler.
    raise NotImplementedError(
        "Serialization support for DynamicCompiler not yet implemented"
    )


class DynamicCompiler(compiler.Compiler):
  """Compiler for handling dynamic shapes in torch.compile()."""

  def __init__(
      self,
      compilation_context: compiler.CompilationContext | None = None,
      debug: bool = False,
  ):
    """Initializes the DynamicCompiler instance.

    Args:
      compilation_context: A `compiler.CompilationContext` instance used for
        maintaining compilation state.
      debug: A `bool` that, when `True`, enables debug logging and artifact
        generation.
    """
    if compilation_context is None:
      compilation_context = compiler.CompilationContext()
    super().__init__(compilation_context, debug=debug)
    self.static_compiler = compiler.StaticCompiler(
        compilation_context, debug=debug
    )

  def execute_pre_grad_passes(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    self.static_compiler.execute_pre_grad_passes(graph_module)

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[InputType],
      is_fwd: bool = True,
  ) -> _DynamicTpuCompiledExecutable:
    """Called by AOT Autograd to compile the graph.

    Args:
      graph_module: The FX graph module to be compiled.
      example_inputs: A list of example inputs for the graph module.
      is_fwd: Indicates whether the forward or backward pass is being compiled.

    Returns:
      A callable `_DynamicTpuCompiledExecutable` that wraps the compiled model
      and handles dynamic shapes at runtime.
    """
    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Compile FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    # Create a SymInt shape manager.
    sym_shape_manager = SymShapeManager(graph_module, example_inputs)

    # Add bound checks on symints. This forces dynamo to trigger a recompilation
    # if the bound constraint isn't satisfied.
    sym_shape_manager.add_bound_checks()

    # Transform the graph module to handle dynamic shapes.
    apply_dynamism_transformations(graph_module, sym_shape_manager)

    # TODO: Prevent truncation of log lines.
    logging.debug(
        "[DynamicTpuBackend] Transformed FX Graph: %s",
        LazyString(graph_module.print_readable),
    )

    # Create example inputs for the model executable.
    model_example_inputs = _get_example_inputs(
        example_inputs, sym_shape_manager
    )

    logging.debug(
        "[DynamicTpuBackend] Example inputs: %s",
        model_example_inputs,
    )

    # Create a static model executable with padded shape tensors as inputs
    static_model_executable = self.static_compiler(
        graph_module,
        model_example_inputs,
        is_fwd=is_fwd,
    )

    logging.debug(
        "[DynamicTpuBackend] Static Model Executable MLIR: %s",
        LazyString(lambda: static_model_executable.mlir_text),
    )

    return _DynamicTpuCompiledExecutable(
        model_executable=static_model_executable,
        sym_shape_manager=sym_shape_manager,
    )
