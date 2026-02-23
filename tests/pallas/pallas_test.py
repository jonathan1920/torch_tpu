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

"""Small graph test for TPU backend."""

import functools
import logging
from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import jax.export
import torch
from torch_tpu import api
from torch_tpu._internal import compile
from torch_tpu._internal import pallas
from torch_tpu._internal.utils import utils


@pallas.custom_kernel(lambda x, y: torch.empty_like(x))
def add_vectors(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


def add_subtract_vectors_kernel(x_ref, y_ref, oadd_ref, osub_ref):
  x, y = x_ref[...], y_ref[...]
  oadd_ref[...] = x + y
  osub_ref[...] = x - y


@pallas.custom_kernel(lambda x, y: (torch.empty_like(x), torch.empty_like(y)))
def add_subtract_vectors(x_ref, y_ref, oadd_ref, osub_ref):
  return add_subtract_vectors_kernel(x_ref, y_ref, oadd_ref, osub_ref)


@pallas.custom_kernel(lambda x, y: torch.empty_like(x))
def subtract_vectors(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x - y


def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


@functools.partial(jax.jit, static_argnums=(0,))
def jax_add_or_subtract_vectors_wrapper(mode, x, y):
  """Emulate kernel lib like tokamax with thin JAX wrapper around pallas.

  TorchTPU custom kernel calls require that all input and output arguments are
  tensor (or None) types, so a function like this must be further wrapped before
  use, i.e. `functools.partial(jax_add_or_subtract_vectors_wrapper, "add")`.

  This constraint can be changed but should be revisited once use cases and
  usability requirements are more concrete.

  Args:
    mode: The mode to use, either "add" or "sub".
    x: The first input tensor.
    y: The second input tensor.
  """

  @functools.partial(jax.jit, static_argnums=(0,))
  def add_subtract_vectors(mode, x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    if mode == "add":
      o_ref[...] = x + y
    else:
      o_ref[...] = x - y

  out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  wrapped = functools.partial(add_subtract_vectors, mode)
  return pl.pallas_call(wrapped, out_shape=out_shape)(x, y)


class TestPallasKernels(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()

  def test_kernel_single_output_functional_style(self):
    add_vectors_fn = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_vectors_fn(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_single_output_functional_style_shape_arg(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    add_vectors_fn = pallas.custom_kernel(
        output_shapes=x,
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
    )

    expected = torch.add(x, y).to("cpu")
    actual = add_vectors_fn(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_single_output_decorator(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_vectors(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_multiple_outputs(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected_add = torch.add(x, y).to("cpu")
    expected_sub = torch.sub(x, y).to("cpu")
    actual_add, actual_sub = add_subtract_vectors(x, y)
    utils.assert_close(actual_add.to("cpu"), expected_add)
    utils.assert_close(actual_sub.to("cpu"), expected_sub)

  def test_kernel_multiple_outputs_functional_style(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    add_subtract_vectors_fn = pallas.custom_kernel(
        output_shapes=[x, y],
        pallas_kernel=add_subtract_vectors_kernel,
        name="add_subtract_vectors",
    )

    expected_add = torch.add(x, y).to("cpu")
    expected_sub = torch.sub(x, y).to("cpu")
    actual_add, actual_sub = add_subtract_vectors_fn(x, y)
    utils.assert_close(actual_add.to("cpu"), expected_add)
    utils.assert_close(actual_sub.to("cpu"), expected_sub)

  def test_kernel_called_twice_same_shapes(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, device=self.device)
    expected = torch.add(torch.add(x, y), z).to("cpu")
    actual = add_vectors(add_vectors(x, y), z).to("cpu")
    utils.assert_close(actual, expected)

  def test_kernel_called_twice_different_shapes(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8], dtype=torch.float32, device=self.device)
    w = torch.tensor([0.9, 1.0], dtype=torch.float32, device=self.device)
    expected = torch.cat([torch.add(x, y), torch.add(z, w)]).to("cpu")
    actual = torch.cat([add_vectors(x, y), add_vectors(z, w)]).to("cpu")
    utils.assert_close(actual, expected)

  def test_two_different_kernels(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    z = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, device=self.device)
    expected = torch.sub(torch.add(x, y), z).to("cpu")
    actual = subtract_vectors(add_vectors(x, y), z).to("cpu")
    utils.assert_close(actual, expected)

  def test_two_kernels_same_name_different_kwargs(self):
    propagate = lambda x, y: torch.empty_like(x)
    add_vectors_no_metadata = pallas.custom_kernel(
        propagate,
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
    )
    add_vectors_with_metadata = pallas.custom_kernel(
        propagate,
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        debug=True,
        metadata={"foo": "bar"},
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual_no_metadata = add_vectors_no_metadata(x, y).to("cpu")
    actual_with_metadata = add_vectors_with_metadata(x, y).to("cpu")
    utils.assert_close(actual_no_metadata, expected)
    utils.assert_close(actual_with_metadata, expected)

  def test_two_kernels_same_name_different_func(self):
    """Test two different kernels registered with the same name.

    The explicit name of a kernel is used for debuggability. The function ID
    should be used to distinguish different python kernels.
    """

    def sub_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x - y

    propagate = lambda x, y: torch.empty_like(x)
    add_vectors_fn = pallas.custom_kernel(
        propagate,
        pallas_kernel=add_vectors_kernel,
        name="math_kernel",
    )
    sub_vectors_fn = pallas.custom_kernel(
        propagate,
        pallas_kernel=sub_vectors_kernel,
        name="math_kernel",
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected_add = torch.add(x, y).to("cpu")
    expected_sub = torch.sub(x, y).to("cpu")
    actual_add = add_vectors_fn(x, y).to("cpu")
    actual_sub = sub_vectors_fn(x, y).to("cpu")
    utils.assert_close(actual_add, expected_add)
    utils.assert_close(actual_sub, expected_sub)

  def test_kernel_input_output_aliasing(self):
    aliasing_add_vectors = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        input_output_aliases={0: 0},
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = aliasing_add_vectors(x, y).to("cpu")
    utils.assert_close(actual, expected)
    # x should be aliased by the output
    utils.assert_close(x.to("cpu"), actual)
    # y should not be aliased
    expected_y = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"
    )
    utils.assert_close(y.to("cpu"), expected_y)

  def test_kernel_donation_invalidates_deferred_op(self):
    aliasing_add_vectors = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        input_output_aliases={0: 0},
    )
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    # Create a deferred op that depends on the pre-donation value of x.
    pre_x_sum = x.sum()

    # Run the aliasing operation, consuming the pre-donation value of x.
    z = aliasing_add_vectors(x, y)
    z.cpu()  # force execution

    # The pre-donation value of x can no longer be used.
    with self.assertRaisesRegex(
        RuntimeError, "INVALID_ARGUMENT: Buffer has been deleted or donated"
    ):
      pre_x_sum.cpu()

  def test_pallas_kernel_compiled_mode(self):

    @torch.library.custom_op(
        "pallas::add_vectors_backward",
        mutates_args=(),
        schema="(Tensor grad) -> (Tensor, Tensor)",
        device_types=["tpu"],
    )
    @pallas.custom_kernel(lambda g: (torch.empty_like(g), torch.empty_like(g)))
    def add_vectors_backward(g_ref, o_x_ref, o_y_ref):
      g = g_ref[...]
      o_x_ref[...] = g
      o_y_ref[...] = g

    add_vectors_backward.register_fake(
        lambda g: (torch.empty_like(g), torch.empty_like(g))
    )

    add_vectors_op = torch.library.custom_op(
        "pallas::add_vectors",
        add_vectors,
        mutates_args=(),
        schema="(Tensor x, Tensor y) -> Tensor",
        device_types=["tpu"],
    )
    add_vectors_op.register_fake(lambda x, _: torch.empty_like(x))

    add_vectors_op.register_autograd(
        lambda ctx, grad: add_vectors_backward(grad)
    )

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def add_vectors_sum(x, y):
      return add_vectors_op(x, y).sum()

    x = torch.tensor(
        [0.1, 0.2, 0.3], dtype=torch.float32, device=self.device
    ).requires_grad_(True)
    y = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float32, device=self.device
    ).requires_grad_(True)

    # Do the forward pass.
    expected_forward = torch.add(x, y).sum()
    actual_forward = add_vectors_sum(x, y)
    utils.assert_close(
        actual_forward.detach().to("cpu"), expected_forward.detach().to("cpu")
    )

    # Do the backward pass for the non-compiled run.
    expected_forward.backward()
    x_grad_expected = x.grad.to("cpu")
    y_grad_expected = y.grad.to("cpu")
    x.grad = None
    y.grad = None

    # Do the backward pass for the compiled run.
    actual_forward.backward()
    x_grad_actual = x.grad.to("cpu")
    y_grad_actual = y.grad.to("cpu")
    utils.assert_close(x_grad_actual, x_grad_expected)
    utils.assert_close(y_grad_actual, y_grad_expected)

  def test_kernel_input_output_aliasing_compiled_mode(self):
    aliasing_add_vectors = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        input_output_aliases={0: 0},
    )

    torch_aliasing_add_vectors = torch.library.custom_op(
        "pallas::aliasing_add_vectors",
        lambda x, y: aliasing_add_vectors(x, y).clone(),
        mutates_args=("x",),
        schema="(Tensor(a!) x, Tensor y) -> Tensor",
        device_types=["tpu"],
    )
    torch_aliasing_add_vectors.register_fake(lambda x, _: torch.empty_like(x))

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def aliased_add_vectors_sum(x, y):
      return torch_aliasing_add_vectors(x, y).sum()

    expected_updated_x = torch.add(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cpu"),
        torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device="cpu"),
    )
    expected_sum = expected_updated_x.sum()

    actual_sum = aliased_add_vectors_sum(x, y)
    utils.assert_close(actual_sum.to("cpu"), expected_sum)
    utils.assert_close(x.to("cpu"), expected_updated_x)

  def test_kernel_compiled_mode_donation_invalidates_deferred_op(self):
    aliasing_add_vectors = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        input_output_aliases={0: 0},
    )

    torch_aliasing_add_vectors = torch.library.custom_op(
        "pallas::aliasing_add_vectors",
        lambda x, y: aliasing_add_vectors(x, y).clone(),
        mutates_args=("x",),
        schema="(Tensor(a!) x, Tensor y) -> Tensor",
        device_types=["tpu"],
    )
    torch_aliasing_add_vectors.register_fake(lambda x, _: torch.empty_like(x))

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    # Create a deferred op that depends on the pre-donation value of x.
    pre_x_sum = x.sum()

    # Run an aliasing compiled operation.
    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def aliased_add_vectors_sum(x, y):
      return torch_aliasing_add_vectors(x, y).sum()

    _ = aliased_add_vectors_sum(x, y)

    # The pre-donation value of x can no longer be used.
    with self.assertRaisesRegex(
        RuntimeError, "INVALID_ARGUMENT: Buffer has been deleted or donated"
    ):
      pre_x_sum.cpu()

  def test_kernel_compile_both_donated_and_non_donated_ops(
      self,
  ):
    aliasing_add_vectors = pallas.custom_kernel(
        lambda x, y: torch.empty_like(x),
        pallas_kernel=add_vectors_kernel,
        name="add_vectors",
        input_output_aliases={0: 0},
    )
    torch_aliasing_add_vectors = torch.library.custom_op(
        "pallas::aliasing_add_vectors",
        lambda x, y: aliasing_add_vectors(x, y).clone(),
        mutates_args=("x",),
        schema="(Tensor(a!) x, Tensor y) -> Tensor",
        device_types=["tpu"],
    )
    torch_aliasing_add_vectors.register_fake(lambda x, _: torch.empty_like(x))

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)

    @torch.compile(fullgraph=True, dynamic=False, backend=compile.TpuBackend())
    def x_used_and_donated(x, y):
      # This op uses x but does not donate it.
      x_sum = x.sum()
      # This op donates x.
      z = torch_aliasing_add_vectors(x, y)

      return x_sum, z

    # This operation executes successfully, because both uses of x are provided
    # to XLA; it inserts defensive copies as needed.
    x_sum, z = x_used_and_donated(x, y)
    utils.assert_close(x_sum.cpu(), torch.tensor(0.6, dtype=torch.float32))
    utils.assert_close(
        z.cpu(), torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32)
    )


class TestJaxWrappedPallasKernels(absltest.TestCase):
  """Test JAX wrapped pallas kernels.

  It is common in kernel libraries like Tokamax to wrap pallas kernels with a
  thin JAX layer that does some trace-time algorithm selection. These will
  eventually generate StableHLO but can take different conditional branches
  depending on the compile-time constants available in JAX.

  This test suite emulates these kernel libraries and ensures these kernel
  libraries that wrap pallas kernels can be used with torch_tpu.
  """

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()
    logging.basicConfig(level=logging.DEBUG, force=True)

  def test_jax_kernel_wrapper_simple(self):
    """Test a kernel wrapper that is a simple passthrough to a JAX kernel."""

    @pallas.custom_jax_kernel
    def add_fn(x, y):
      return jax.numpy.add(x, y)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_fn(x, y).to("cpu")
    utils.assert_close(actual, expected)

  def test_jax_kernel_wrapper_with_donation(self):
    """Test a kernel wrapper with a donated argument."""

    @pallas.custom_jax_kernel(input_output_aliases={0: 0}, donate_argnums=(0,))
    def add_fn(x, y):
      return jax.numpy.add(x, y)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    expected = torch.add(x, y).to("cpu")
    actual = add_fn(x, y).to("cpu")
    utils.assert_close(actual, expected)
    # x should be aliased by the output.
    utils.assert_close(x.cpu(), expected)

  def test_jax_kernel_with_trace_time_conditional(self):
    """Test a kernel wrapper that has a trace time conditional.

    This is a common requirement for kernel libraries like Tokamax, where
    different kernels are selected at call time based on compile time constants.

    Today TorchTPU requires all kernels to resolve their trace time args with
    a wrapper function, i.e. `functools.partial`. This can be revisited once
    use cases and usability requirements are more concrete.
    """
    add_fn = functools.partial(jax_add_or_subtract_vectors_wrapper, "add")
    sub_fn = functools.partial(jax_add_or_subtract_vectors_wrapper, "sub")

    add_kernel = pallas.custom_jax_kernel(add_fn)
    sub_kernel = pallas.custom_jax_kernel(sub_fn)

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    add_expected = torch.add(x, y).to("cpu")
    sub_expected = torch.sub(x, y).to("cpu")
    actual_add = add_kernel(x, y).to("cpu")
    actual_sub = sub_kernel(x, y).to("cpu")
    utils.assert_close(actual_add, add_expected)
    utils.assert_close(actual_sub, sub_expected)

  def test_jax_kernel_with_static_argnums(self):
    """Test custom_jax_kernel with static_argnums."""
    kernel = pallas.custom_jax_kernel(
        jax_add_or_subtract_vectors_wrapper, static_argnums=(0,)
    )

    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    add_expected = torch.add(x, y).to("cpu")
    sub_expected = torch.sub(x, y).to("cpu")
    actual_add = kernel("add", x, y).to("cpu")
    actual_sub = kernel("sub", x, y).to("cpu")
    utils.assert_close(actual_add, add_expected)
    utils.assert_close(actual_sub, sub_expected)

  def test_kernel_wrapper_with_none_return(self):
    """Simulate kernel libraries like tokamax that wrap pallas kernels."""

    @pallas.custom_jax_kernel
    def wrapper():
      return None

    none = wrapper()
    self.assertIsNone(none)

  def test_kernel_wrapper_with_none_input(self):
    """Simulate kernel libraries like tokamax that wrap pallas kernels."""

    @pallas.custom_jax_kernel
    def wrapper(x):
      return x

    none = wrapper(None)
    self.assertIsNone(none)


if __name__ == "__main__":
  absltest.main()
