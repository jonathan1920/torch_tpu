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

"""APIs for Pallas integration.

This module provides APIs to call Pallas kernels from PyTorch. Note that use
of these APIs requires that your environment also has JAX/Pallas installed.
"""

from collections.abc import Mapping
import functools
import logging
from typing import Any, Callable, Sequence
import frozendict
import torch
from torch_tpu._internal.pallas import tpu_torch_pallas

logger = logging.getLogger(__name__)

try:
  # TODO: TorchXLA hit issues when workin with both JAX and PT in the same
  # process, hitting deadlocks when each framework tried to access the PJRT
  # plugin. Unclear if this applies for trace-only, but we may need to better
  # protect against this, see @requires_jax in PT/XLA repo.
  import jax
  import jax.export
  from jax.experimental import pallas as pl
except ImportError as ex:
  raise ImportError(
      "JAX/Pallas is not available. Please install JAX to use this API."
  ) from ex


# Based on torch/_C/__init__.pyi
# Skip alias types since they are covered by other entries in the table. JAX
# also has similar alias types, but it relies on JAX config.py, which may not
# align to PT's configuration (jax.numpy.float_ => float32 | float64)
TORCH_TO_JAX_DTYPE_MAP = frozendict.frozendict({
    torch.float32: jax.numpy.float32.dtype,
    # torch.float: jax.numpy.float_.dtype,  # Avoid PT alias types
    torch.float64: jax.numpy.float64.dtype,
    torch.double: jax.numpy.double.dtype,
    torch.float16: jax.numpy.float16.dtype,
    torch.bfloat16: jax.numpy.bfloat16.dtype,
    torch.float8_e4m3fn: jax.numpy.float8_e4m3fn.dtype,
    torch.float8_e4m3fnuz: jax.numpy.float8_e4m3fnuz.dtype,
    torch.float8_e5m2: jax.numpy.float8_e5m2.dtype,
    torch.float8_e5m2fnuz: jax.numpy.float8_e5m2fnuz.dtype,
    torch.float8_e8m0fnu: jax.numpy.float8_e8m0fnu.dtype,
    # torch.float4_e2m1fn_x2: jax.numpy.float4_e2m1fn.dtype,  # x2 is packed
    # torch.half: jax.numpy.float16.dtype,  # PT alias type
    torch.uint8: jax.numpy.uint8.dtype,
    torch.uint16: jax.numpy.uint16.dtype,
    torch.uint32: jax.numpy.uint32.dtype,
    torch.uint64: jax.numpy.uint64.dtype,
    torch.int8: jax.numpy.int8.dtype,
    torch.int16: jax.numpy.int16.dtype,
    # torch.short: jax.numpy.int16.dtype,  # PT alias type
    torch.int32: jax.numpy.int32.dtype,
    # torch.int: jax.numpy.int_.dtype,  # PT alias type
    torch.int64: jax.numpy.int64.dtype,
    # torch.long: jax.numpy.int64.dtype,  # PT alias type
    # torch.complex32: jax.numpy.complex32.dtype, # complex32 DNE in JAX
    torch.complex64: jax.numpy.complex64.dtype,
    # torch.chalf: jax.numpy.chalf.dtype, # PT alias type
    # torch.cfloat: jax.numpy.csingle.dtype,  # PT alias type
    torch.complex128: jax.numpy.complex128.dtype,
    torch.cdouble: jax.numpy.cdouble.dtype,
    # torch.quint8: jax.numpy.quint8.dtype,
    # torch.qint8: jax.numpy.qint8.dtype,
    # torch.qint32: jax.numpy.qint32.dtype,
    torch.bool: jax.numpy.bool_.dtype,
    # torch.quint4x2: jax.numpy.quint4x2.dtype,
    # torch.quint2x4: jax.numpy.quint2x4.dtype,
    # torch.bits1x8: jax.numpy.bits1x8.dtype,
    # torch.bits2x4: jax.numpy.bits2x4.dtype,
    # torch.bits4x2: jax.numpy.bits4x2.dtype,
    # torch.bits8: jax.numpy.bits8.dtype,
    # torch.bits16: jax.numpy.bits16.dtype,
})
JAX_TO_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_TO_JAX_DTYPE_MAP.items()}


def jax_placeholder(tensor: torch.Tensor) -> jax.ShapeDtypeStruct:
  """Converts a torch tensor to a jax placeholder for tracing."""
  if not isinstance(tensor, torch.Tensor):
    # Preserve POD constants / None types
    return tensor

  jax_dtype = TORCH_TO_JAX_DTYPE_MAP.get(tensor.dtype, None)
  logger.debug("[jax_placeholder] dtype: %s -> %s", tensor.dtype, jax_dtype)
  if jax_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {tensor.dtype}"
    )
  return jax.ShapeDtypeStruct(tensor.shape, jax_dtype)


def jax_placeholders(tensors: Sequence[torch.Tensor] | torch.Tensor):
  """Converts a sequence of torch tensors to a sequence of jax placeholders."""
  if isinstance(tensors, torch.Tensor):
    return jax_placeholder(tensors)
  return [jax_placeholder(tensor) for tensor in tensors]


def torch_placeholder(
    tensor: jax.core.ShapedArray | None,
) -> torch.Tensor | None:
  """Converts a jax output to a torch tensor for return type.

  JAX JIT functions only support tensor and None types, all other POD types will
  error.
  """
  if not isinstance(tensor, jax.core.ShapedArray):
    # Preserve POD constants / None types
    return tensor

  torch_dtype = JAX_TO_TORCH_DTYPE_MAP.get(tensor.dtype, None)
  logger.debug("[torch_placeholder] dtype: %s -> %s", tensor.dtype, torch_dtype)
  if torch_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {tensor.dtype}"
    )
  return torch.empty(tensor.shape, dtype=torch_dtype)


def _get_kernel_invocation_key(
    trace_key: str,
    input_args: Sequence[torch.Tensor | Any],
    input_kwargs: dict[str, Any],
    static_argnums: tuple[int, ...] = (),
) -> str:
  """Construct a unique ID for the kernel invocation.

  Builds a string key that includes both the pallas compile args and
  the input shapes; if either of these changes, we need to recompile.
  For static args, includes value hash instead of shape.

  Args:
    trace_key: A string key used to identify the exported kernel.
    input_args: The input arguments to the kernel.
    input_kwargs: The input keyword arguments to the kernel.
    static_argnums: Tuple of argument positions that are compile-time constants.

  Returns:
    A string key that includes both the pallas compile args and
    the input shapes; if either of these changes, we need to recompile.
  """
  static_argnums_set = frozenset(static_argnums)

  def get_hash(i, item):
    if i in static_argnums_set:
      return f"static_{get_hash(-1, item)}"
    if isinstance(item, torch.Tensor):
      return f"{item.shape}x{item.dtype}"
    return str(item)

  input_kwargs_str = ";".join(
      f"{get_hash(-1, key)}:{get_hash(-1, value)}"
      for key, value in input_kwargs.items()
  )

  inputs_shapes = [get_hash(i, arg) for i, arg in enumerate(input_args)]
  inputs_shapes_str = ",".join(inputs_shapes)
  return ";".join([trace_key, inputs_shapes_str, input_kwargs_str])


_KernelResultT = Sequence[torch.Tensor] | torch.Tensor | None


class JaxCallable:
  """A torch callable function that wraps a JAX function.

  Supports static_argnums to allow passing non-tensor arguments directly.
  """

  def __init__(
      self,
      name: str,
      jit_fn: Any,
      trace_key: str,
      static_argnums: tuple[int, ...] = (),
      input_output_aliases: Mapping[int, int] | None = None,
  ):
    """Initializes a JaxCallable, a cached callable for custom kernels.

    The callable is cached based on the (trace_key, input_shapes, input_kwargs),
    if a hash is not found the function is re-traced and re-registered. The JIT
    function passed in is a `jax.jit` wrapped pallas_call or function. This
    level of abstraction allows output types to be determined via tracing,
    instead of requiring they be passed in explicitly.

    Note that for JAX/python functions the trace_key should be the function's ID
    as well as a hash of any jit / pallas kwargs used to create the callable
    jit_fn.

    Args:
      name: The name of the JAX function, often including the function ID.
      jit_fn: The JAX function or wrapped pallas function output from `jax.jit`.
      trace_key: A string key of args / kwargs used to trace the exported
        function.
      static_argnums: Tuple of argument positions that are compile-time
        constants.
      input_output_aliases: A mapping of input indices to output indices that
        alias each input. See docs for jax.experimental.pallas.pallas_call.
    """
    self.name = name
    self.trace_key = trace_key
    self.output_shapes = {}  # cache output shapes per kernel specialization
    self.static_argnums = static_argnums
    self.exported = jax.export.export(jit_fn, platforms=["tpu"])
    self.input_output_aliases = (
        dict(input_output_aliases) if input_output_aliases is not None else {}
    )

    logger.debug("Creating JAX callable: %s", self)

  def __repr__(self):
    return (
        f"JaxCallable(name={self.name}, trace_key={self.trace_key},"
        f" static_argnums={self.static_argnums})"
    )

  def _validate_args(self, *args) -> None:
    """Validates that the arguments are supported by the kernel."""
    for i, arg in enumerate(args):
      if i in self.static_argnums:
        if isinstance(arg, torch.Tensor):
          raise ValueError(
              f"Argument at index {i} is marked as static but is a"
              " torch.Tensor. Tensors cannot be static arguments."
          )
        continue
      if arg is not None and not isinstance(arg, torch.Tensor):
        raise TypeError(
            "custom kernels only support torch.Tensor and None inputs, but"
            f" input {i} was of type {type(arg)}. Use static_argnums or"
            " functools.partial for non-tensor args."
        )

  def __call__(self, *args, **kwargs) -> _KernelResultT:
    self._validate_args(*args)

    kernel_key = _get_kernel_invocation_key(
        self.trace_key, args, kwargs, self.static_argnums
    )
    output_shapes, out_tree = self.output_shapes.get(kernel_key, (None, None))
    kernel_exists = tpu_torch_pallas.lookup_custom_kernel(self.name, kernel_key)
    if not output_shapes or not kernel_exists:
      jax_args = jax_placeholders(args)
      lowered = self.exported(*jax_args, **kwargs)
      tpu_torch_pallas.register_custom_kernel(
          self.name,
          kernel_key,
          serialized_mlir_module=lowered.mlir_module_serialized,
      )
      # Use out_tree to format the outputs shapes - i.e. pack in tuple, nested
      # tuple, etc.
      output_shapes = [torch_placeholder(aval) for aval in lowered.out_avals]
      out_tree = lowered.out_tree
      self.output_shapes[kernel_key] = (output_shapes, out_tree)

    tensor_args = [
        arg
        for i, arg in enumerate(args)
        if arg is not None and i not in self.static_argnums
    ]

    results = tpu_torch_pallas.call_custom_kernel(
        self.name,
        kernel_key,
        inputs=tensor_args,
        output_shapes=output_shapes,
        input_output_aliases=self.input_output_aliases,
    )
    return out_tree.unflatten(results)


def custom_kernel(
    output_shapes: Callable[..., _KernelResultT] | _KernelResultT,
    pallas_kernel: Callable[..., Any] | None = None,
    **pl_kwargs,
) -> (
    Callable[..., _KernelResultT]
    | Callable[[Callable[..., Any]], Callable[..., _KernelResultT]]
):
  """A decorator that imports a Pallas kernel into for use with torch_tpu.

  ```py
  @pallas.custom_kernel(lambda x, y: torch.empty_like(x))
  def add_vectors(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

  x = torch.ones(10)
  y = torch.ones(10)
  z = add_vectors(x, y)
  ```

  Args:
    output_shapes: A tensor or tensor list, or a function that takes the input
      arguments and returns the output shapes of the Pallas kernel. Note, only
      the tensor's shape and dtype information is used, the data is not used.
    pallas_kernel: The pallas kernel to compile. If None then return a decorator
      that expects a pallas_kernel argument.
    **pl_kwargs: Additional keyword arguments to configure the inner
      pallas_call, like `grid`, `in_specs`, `out_specs`, `input_output_aliases`,
      etc.

  Returns:
    A decorator that takes a pallas kernel function and returns a callable that
    can be used to call the kernel on torch.Tensor inputs.
  """

  def get_output_shapes(*args, **kwargs):
    if callable(output_shapes):
      return output_shapes(*args, **kwargs)
    return output_shapes

  def decorator(
      pallas_kernel: Callable[..., None],
  ) -> Callable[..., _KernelResultT]:
    # Prefer explicit name if specified in kwargs (pallas_call allows this)
    # If that isn't specified, we use the function name if available, with
    # "kernel" as a fallback name.
    name = pl_kwargs.get("name", getattr(pallas_kernel, "__name__", "kernel"))
    name_key = f"{name}_{id(pallas_kernel)}"  # distinguish funcs with same name
    trace_key = _get_kernel_invocation_key(name_key, [], pl_kwargs)

    # Make a JAX callable for the wrapped pallas kernel.
    # Unfortunately for Pallas kernels we need to defer until call time to
    # build the JAX Callable since the output is a function of the input that
    # isn't known until call time.
    @functools.wraps(pallas_kernel)
    def wrapped_call(*args, **kwargs) -> _KernelResultT:
      out_shapes = get_output_shapes(*args, **kwargs)
      jax_out_shapes = jax_placeholders(out_shapes)
      pl_fn_jit = jax.jit(
          pl.pallas_call(pallas_kernel, out_shape=jax_out_shapes, **pl_kwargs),
      )
      jax_callable = JaxCallable(
          name=name,
          jit_fn=pl_fn_jit,
          trace_key=trace_key,
          input_output_aliases=pl_kwargs.get("input_output_aliases", None),
      )
      return jax_callable(*args, **kwargs)

    return wrapped_call

  if pallas_kernel is None:
    return decorator
  return decorator(pallas_kernel)


def custom_jax_kernel(
    jax_fn: Callable[..., Any] | None = None,
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    input_output_aliases: Mapping[int, int] | None = None,
    **jit_kwargs,
) -> Callable[..., JaxCallable] | Callable[[Callable[..., Any]], JaxCallable]:
  """A decorator that imports a JAX kernel into for use with torch_tpu.

  Often pallas kernel libraries are written with thin JAX usability layers, i.e.
  in tokamax. `custom_jax_kernel` allows for interop with these JAX kernel
  wrappers.

  ```py
  # Option 1: Using functools.partial (original approach)
  tokamax_ragged_dot = functools.partial(
      tokamax.ragged_dot,
      implementation="mosaic",
      preferred_element_type=jnp.bfloat16
  )
  ragged_dot = pallas.custom_jax_kernel(tokamax_ragged_dot)

  # Option 2: Using static_argnums
  ragged_dot = pallas.custom_jax_kernel(
      tokamax.ragged_dot,
      static_argnums=(2, 3),  # Mark positions 2 and 3 as static
  )
  # Call with static args directly:
  ragged_dot(lhs, rhs, "mosaic", jnp.bfloat16)
  ```

  Args:
    jax_fn: The JAX function to compile. If None then return a decorator that
      expects a jax_fn argument.
    name: Optional debug friendly kernel name, attempt to use the function's
      name if not provided.
    static_argnums: Tuple of argument positions that are compile-time constants.
      These args can be non-tensors and are used for caching by value.
    input_output_aliases: A mapping of input indices to output indices that
      alias each input. Optional, default is no aliasing. This must match the
      settings of `donate_argnums` and/or `donate_argnames` in jit_kwargs.
    **jit_kwargs: Additional keyword arguments to configure the inner `jax.jit`,
      like `donate_argnums`, etc.

  Returns:
    A decorator that takes a JAX function and returns a callable that
    can be used to call the kernel with torch.Tensor inputs.
  """

  def decorator(jax_fn: Callable[..., Any]) -> JaxCallable:
    nonlocal name
    if name is None:
      name = getattr(jax_fn, "__name__", "kernel")
    name_key = f"{name}_{id(jax_fn)}"  # distinguish funcs with same name

    trace_key = _get_kernel_invocation_key(
        name_key,
        [],
        {**jit_kwargs, "static_argnums": static_argnums},
    )

    jit_fn = jax.jit(jax_fn, static_argnums=static_argnums, **jit_kwargs)
    return JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        static_argnums=static_argnums,
        input_output_aliases=input_output_aliases,
    )

  if jax_fn is None:
    return decorator
  return decorator(jax_fn)
