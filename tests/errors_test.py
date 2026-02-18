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

"""Tests error handling."""

import re
from typing import Any
import unittest
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import env
from torch_tpu._internal import testing as tt_testing
from tests import error_testing as et

_TEST_MODE = et.TEST_MODE


def _is_internal() -> bool:
  """Returns true if the test is running in the internal environment.

  Do not remove the `import env` above even if _is_internal() is not used
  directly in this file. It will be useful for future changes.
  """

  return env.IS_INTERNAL_TORCH_TPU


def setUpModule():
  """Called by absltest after flags are parsed and before tests are run."""
  et.set_up_module()


def _get_aminmax_outputs(
    op: Any, device: str, dtype: torch.dtype
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Creates the output tensors for `op` of `dtype`, on `device`.

  Args:
    op: a torch function. Should be one of: amax, amin, or aminmax.
    device: the device of the output tensors
    dtype: the dtype of the output tensors

  Returns:
    The out object to be used as a kwarg when calling `op`.
  """

  assert op in (torch.amax, torch.amin, torch.aminmax)

  def scalar_tensor() -> torch.Tensor:
    return torch.tensor(0, device=device, dtype=dtype)

  return (
      scalar_tensor()
      if op != torch.aminmax
      else (scalar_tensor(), scalar_tensor())
  )


def _get_convolution_default_args(is_backward: bool = False) -> dict[str, Any]:
  """Returns the default arguments for the convolution ops.

  Default arguments for both ops:
    - `torch.convolution`
    - `torch.ops.aten.convolution_backward`

  IMPORTANT: keep the arguments ordered. Even though we can pass this dictionary
  as keyword arguments to `torch.convolution`, we can't do the same for
  `torch.ops.aten.convolution_backward`, since the latter only accepts
  positional arguments.

  This is used so that we don't have to keep specifying all default arguments
  when calling both ops.

  Args:
    is_backward: flags whether the returned parameter-value dictionary should
      correspond to the forward or backward op.

  Returns:
    A dictionary that associates parameters with default values for either the
    forward or the backward op.
  """

  default = {
      "bias": None,
      "bias_sizes": None,
      "stride": (1,),
      "padding": (0,),
      "dilation": (1,),
      "transposed": False,
      "output_padding": (0,),
      "groups": 1,
      "output_mask": (True, True, True),
  }

  # Remove forward/backward-only keyword arguments.
  if is_backward:
    default.pop("bias")
  else:
    default.pop("bias_sizes")
    default.pop("output_mask")

  return default


def _run_convolution(*args, **kwargs):
  """Runs torch.convolution with the given arguments.

  Convenient function for running `torch.convolution`, without having to specify
  all arguments.

  Reason: even though ops like `torch.conv2d` (and other more popular ops) has
  default arguments, `torch.convolution` doesn't.

  Args:
    *args: Positional arguments for `torch.convolution`.
    **kwargs: Keyword arguments to override

  Returns:
    The output of `torch.convolution`.
  """

  merged = _get_convolution_default_args()
  # Update the copied default args dictionary with the kwargs passed as
  # argument.
  merged.update(kwargs)
  # Concatenate the values to the given positional arguments.
  args = args + tuple(merged.values())

  return torch.convolution(*args)


def _run_convolution_backward(*args, **kwargs):
  """Runs `torch.ops.aten.convolution_backward` with the given arguments.

  Convenient function for running `torch.ops.aten.convolution_backward`, without
  having to specify all arguments.

  Reason: this function doesn't have default arguments, and only accepts
  positional arguments.

  Args:
    *args: Positional arguments for `torch.convolution`.
    **kwargs: Keyword arguments to override

  Returns:
    The output of `torch.ops.aten.convolution_backward`.
  """

  merged = _get_convolution_default_args(is_backward=True)
  # Update the copied default args dictionary with the kwargs passed as
  # argument.
  merged.update(kwargs)
  # Concatenate the values to the given positional arguments.
  args = args + tuple(merged.values())

  return torch.ops.aten.convolution_backward(*args)


def _parameterize_convolution_fwd_bwd(
    forward: dict[str, Any] | None = None,
    backward: dict[str, Any] | None = None,
):
  """Parameterizes convolution tests, running the forward and backward ops.

  Convenient test decorator, for parameterizing a convolution test into 2
  variants: forward (`torch.convolution`) and backward tests
  (`torch.ops.aten.convolution_backward`). It's equivalent to:

  ```py
  @parameterized.named_parameters(
      { "testcase_name": "forward", "convolution": ..., **forward },
      { "testcase_name": "forward", "convolution": ..., **backward },
  )
  ```

  In summary, this test decorator is a wrapper for calling
  `parameterized.named_parameters()` with 2 dictionaries (forward and backward).
  In each of them, it will automatically set the value for the following keys:

  - `testcase_name`: with values _"forward"_ and _"backward"_
  - `convolution` (the function that actually runs the forward or backward
  convolution): with values `_run_convolution` and `_backward_wrapped_with_grad`
  (see its definition below)

  Although parameterizing tests is not ideal, we do this because there are many
  checks (7) that are run on both forward and backward convolution ops.
  Therefore, this small extra complexity is justified by the following reasons:

    - Setup is identical
    - Error checks covered are the same
    - Error messages are identical (modulo name of the function)
    - Avoid duplicating the tests

  This decorator assumes that:

    - Both `forward` and `backward` parameters have the same key. It will raise
      an `AssertionError`, otherwise.
    - the grad shape will be (2, 1, 8, 8). It can be easily adapted by promoting
      this variable to be a parameter

  Args:
    forward: a dictionary that will be passed down to the test case keyword
      arguments of the forward test.
    backward: a dictionary that will be passed down to the test case keyword
      arguments of the backward test.

  Returns:
    A `parameterized.named_parameter()` decorator with forward and backward
    parameters set.
  """

  # Leave this variable here for visibility.
  grad_shape = (2, 1, 8, 8)

  # From this point onwards, both `forward` and `backward` should:
  #
  #   - Not be `None`
  #   - Have the same set of keys
  forward = forward or {}
  backward = backward or {}

  assert forward.keys() == backward.keys(), (
      "convolution parameterization for `forward` and `backward` dictionaries"
      f" must have the same keys, got {set(forward.keys())} and"
      f" {set(backward.keys())}"
  )

  # Wraps `_run_convolution_backward()` function.
  #
  # Before actually calling the convolution backward function, creates a grad
  # tensor. This allows the forward and the backward function to be called with
  # the same set of positional arguments.
  def backward_wrapped_with_grad(*args, **kwargs):
    grad = torch.zeros(grad_shape, device=et.device())
    return _run_convolution_backward(grad, *args, **kwargs)

  return parameterized.named_parameters(
      {
          "testcase_name": "forward",
          "convolution": _run_convolution,
          **forward,
      },
      {
          "testcase_name": "backward",
          "convolution": backward_wrapped_with_grad,
          **backward,
      },
  )


def _make_lu_unpack_outputs(
    p: tuple[int, ...], l: tuple[int, ...], u: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
  """Creates a 3-tuple of tensors for `lu_unpack()` op."""
  return (
      torch.empty(p, device=et.device()),
      torch.empty(l, device=et.device()),
      torch.empty(u, device=et.device()),
  )


class TpuOnlyErrorTest(et.TpuOnlyErrorTestBase, parameterized.TestCase):
  """Tests error messages on TPU."""

  def test_ones_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        # PyTorch implements ones() in terms of empty() automatically, so we
        # don't have a good way to know that the op in question is "ones"
        # instead of "empty".
        "empty(): TorchTPU does not yet support dtype int1",
        message_reviewed_by="wan",
    ):
      # This succeeds on CPU but is unimplemented on TPU.
      torch.ones(2, 3, device=et.device(), dtype=torch.int1)

  def test_prod_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "prod(): my error",
    ):
      torch.prod(t1).to("cpu")

  def test_prod_out_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "prod(): my error",
    ):
      torch.prod(t1, dim=1, out=torch.zeros_like(t1)).to("cpu")

  def test_nonzero_with_op_dispatch_failure(self):
    """Tests that nonzero() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "nonzero(): my error",
    ):
      out = torch.nonzero(t1)
      out.to("cpu")

  def test_nonzero_out_with_op_dispatch_failure(self):
    """Tests that nonzero() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "nonzero(): my error",
    ):
      out = torch.zeros(1, 3, device="tpu", dtype=torch.long)
      torch.nonzero(t1, out=out)
      out.to("cpu")

  def test_nonzero_size_with_op_dispatch_failure(self):
    """Tests that nonzero_size() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero_size", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "nonzero(): my error",
    ):
      out = torch.nonzero(t1)
      out.to("cpu")

  def test_nonzero_out_size_with_op_dispatch_failure(self):
    """Tests that nonzero_size() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero_size", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "nonzero(): my error",
    ):
      out = torch.zeros(1, 3, device="tpu", dtype=torch.long)
      torch.nonzero(t1, out=out)
      out.to("cpu")

  def test_topk_with_op_dispatch_failure(self):
    """Tests that topk() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("topk", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "topk(): my error",
    ):
      values, indices = torch.topk(t1, 2)
      values.to("cpu")
      indices.to("cpu")

  def test_index_put_with_op_dispatch_failure(self):
    """Tests that index_put() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("index_put_", "my error")
    t = torch.tensor([0, 1], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "index_put_(): my error",
    ):
      torch.index_put_(
          t,
          (torch.tensor([0], device="tpu"),),
          torch.tensor([0], device="tpu"),
      )
      t.to("cpu")

  def test_index_put_bool_mask_with_op_dispatch_failure(self):
    """Tests that index_put() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("index_put_", "my error")
    t = torch.tensor([0, 1], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        "index_put_(): my error",
    ):
      torch.index_put_(
          t,
          (torch.tensor([True, False], device="tpu"),),
          torch.tensor(1, device="tpu"),
      )
      t.to("cpu")

  def test_index_put_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_put() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    with et.assert_raises_message(
        RuntimeError,
        "index_put_(): inplace writes to overlapping views are undefined"
        " behavior and are not supported.\nBecause multiple logical tensor"
        " indices point to the same buffer elements, writes from multiple"
        " indices may overwrite each other.\nPlease use clone() or contiguous()"
        " to copy the tensor before writing",
    ):
      torch.index_put_(
          t,
          (
              torch.tensor([0], device=et.device(), dtype=torch.long),
              torch.tensor([0], device=et.device(), dtype=torch.long),
          ),
          torch.tensor(0.0, device=et.device()),
      )

  def test_index_copy_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_copy() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    self_tensor = torch.zeros(3, 3, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)
    source = torch.ones(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        "index_copy(): inplace writes to overlapping views are undefined"
        " behavior and are not supported.\nBecause multiple logical tensor"
        " indices point to the same buffer elements, writes from multiple"
        " indices may overwrite each other.\nPlease use clone() or contiguous()"
        " to copy the tensor before writing",
    ):
      torch.index_copy(self_tensor, 0, index, source, out=t)

  def test_mm_with_oom_result(self):
    """Tests that mm with a large result that OOMs fails with expected error."""

    # Each input tensor is 4 MB.
    t1 = torch.ones(2**20, 1, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(1, 2**20, device=et.device(), dtype=torch.float32)
    # The result tensor would need 4 TB, which is impossible to allocate.
    t3 = torch.mm(t1, t2)
    with et.assert_raises_message(
        RuntimeError,
        "copy_(): in _copy_from, the tensor shape float32[1048576, 1048576] is"
        " too large to fit in memory",
    ):
      t3.to("cpu")

  def test_add_tpu_and_cpu(self):
    tpu_tensor = torch.tensor([1, 2, 3], device="tpu")
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        "Expected all tensors to be on the same device, but found at least"
        " two devices, tpu:0 and cpu!",
    ):
      tpu_tensor + cpu_tensor  # pylint: disable=pointless-statement

  def test_add_cpu_and_tpu(self):
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    tpu_tensor = torch.tensor([1, 2, 3], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        "Expected all tensors to be on the same device, but found at least"
        " two devices, tpu:0 and cpu!",
    ):
      cpu_tensor + tpu_tensor  # pylint: disable=pointless-statement

  def test_add_out_cpu(self):
    tpu_tensor1 = torch.tensor([1, 2, 3], device="tpu")
    tpu_tensor2 = torch.tensor([4, 5, 6], device="tpu")
    cpu_tensor = torch.tensor([7, 8, 9], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        "add(): the out tensor is expected to be on tpu, got cpu",
    ):
      torch.add(tpu_tensor1, tpu_tensor2, out=cpu_tensor)

  def test_dtype_complex32_unsupported(self):
    with et.assert_raises_message(
        RuntimeError,
        "empty(): TorchTPU does not yet support dtype complex32",
        message_reviewed_by="wan",
    ):
      torch.empty(2, dtype=torch.complex32, device="tpu")

  def test_dtype_float4_e2m1fn_x2_unsupported(self):
    # The float4_e2m1fn_x2 dtype represents 2x f4e2m1fn values packed into 8bits
    # which is different from XLA's supported single-value f4e2m1fn dtype.
    with et.assert_raises_message(
        RuntimeError,
        "empty(): TorchTPU does not yet support dtype float4_e2m1fn_x2",
        message_reviewed_by="wan",
    ):
      torch.empty(2, dtype=torch.float4_e2m1fn_x2, device="tpu")

  def test_empty_strided(self):
    with et.assert_raises_message(
        RuntimeError,
        "empty_strided(): TorchTPU does not yet support dtype complex32",
        message_reviewed_by="wan",
    ):
      torch.empty_strided((2,), (1,), dtype=torch.complex32, device="tpu")

  def test_empty_unsupported_layout(self):
    with et.assert_raises_message(
        NotImplementedError,
        "empty(): only layout=torch.strided is supported by TorchTPU for now,"
        " got torch.jagged",
    ):
      torch.empty(2, layout=torch.jagged, device="tpu")

  def test_histc_bounds_unsupported_dtype(self):
    """Tests that torch.histc() fails when the bounds have an unsupported dtype."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "histc(): expected min and max to be float or int type, got Bool and"
        " Bool",
    ):
      torch.histc(t, min=False, max=True)

  def test_is_nonzero_with_more_than_one_value(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        "Boolean value of Tensor with more than one value is ambiguous",
    ):
      torch.is_nonzero(torch.tensor([1, 3, 5], device="tpu"))

  def test_is_nonzero_with_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        "Boolean value of Tensor with no values is ambiguous",
    ):
      torch.is_nonzero(torch.tensor([], device="tpu"))

  def test_is_nonzero_with_nested_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        "Boolean value of Tensor with no values is ambiguous",
    ):
      torch.is_nonzero(torch.tensor([[]], device="tpu"))

  def test_masked_select_out_on_different_device(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="tpu", dtype=torch.float32)
    mask = torch.ones(5, device="tpu", dtype=torch.bool)
    out = torch.ones(5, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "masked_select(): the out tensor is expected to be on tpu, got cpu",
    ):
      torch.masked_select(t, mask, out=out)

  def test_masked_select_out_on_different_device2(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="cpu", dtype=torch.float32)
    mask = torch.ones(5, device="cpu", dtype=torch.bool)
    out = torch.ones(5, device="tpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        "masked_select(): tensor is expected to be on tpu, got cpu",
    ):
      torch.masked_select(t, mask, out=out)

  def test_set_invalid_metadata(self):
    t = torch.zeros(1, device="tpu", dtype=torch.float32)
    source = torch.arange(8, device="tpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "set_(): tensor would require at least 64 bytes, but only 32 are"
        " available",
    ):
      t.set_(source.untyped_storage(), storage_offset=0, size=[16], stride=[1])

  def test_resize_materialization_error(self):
    """Materializing a tensor that has been resized down throws an error."""
    # This is not a supported operation on CPU either, but if and what message
    # gets thrown depends on how the elements are accessed.
    tensor = torch.arange(8, dtype=torch.float32, device="tpu")
    nbytes = tensor.untyped_storage().nbytes()
    tensor.untyped_storage().resize_(nbytes // 2)
    with et.assert_raises_message(
        IndexError,
        "copy_(): cannot read 32 bytes (8 elements of type float32 with an"
        " offset of 0 elements) from a storage buffer with 16 bytes",
    ):
      tensor.to("cpu")

  # Confirmed this test does NOT fail on cpu.
  def test_addmm_on_complex_input(self):
    complex_val = torch.complex(torch.tensor(1.0), torch.tensor(1.0))
    complex_val = complex_val.tile((2, 2))
    complex_val = complex_val.to(et.device())

    input_ = complex_val.clone()
    mat1 = complex_val.clone()
    mat2 = complex_val.clone()
    with et.assert_raises_message(
        NotImplementedError,
        "addmm(): complex dtypes are not supported yet",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_cumsum_with_unsupported_boolean_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        "cumsum(): dtype bool is not supported yet",
    ):
      t = torch.ones(2, 2, device="tpu")
      res = torch.cumsum(t, dim=1, dtype=torch.bool)
      res.to("cpu")

  def test_index_add_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_add() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    self_tensor = torch.zeros(3, 3, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)
    source = torch.ones(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        "index_add(): inplace writes to overlapping views are undefined"
        " behavior and are not supported.\nBecause multiple logical tensor"
        " indices point to the same buffer elements, writes from multiple"
        " indices may overwrite each other.\nPlease use clone() or contiguous()"
        " to copy the tensor before writing",
    ):
      torch.index_add(self_tensor, 0, index, source, out=t)

  # TODO: b/480225714 remove this after the corresponding PyTorch#173995 bug
  # is fixed.
  @unittest.skip("PyTorch upstream bug #173995 crashes this test.")
  def test_unsafe_masked_index_error(self):
    with et.assert_raises_message(
        RuntimeError,
        "index(): at least one index tensor must be defined",
    ):
      torch.ops.aten._unsafe_masked_index(
          torch.tensor([1.0], device=et.device()),
          torch.tensor([True], device=et.device()),
          [],  # indices
          0.0,  # fill
      )

  def test_max_pool2d_with_indices_input_too_large(self):
    """Tests max_pool2d_with_indices fails if input has > 2^31-1 elements."""
    # h * w = 2^15 * 2^16 = 2^31, exceeding the ui32 index limit 2^31 - 1
    t = torch.empty(1, 1, 2**15, 2**16, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        re.compile(
            r".*tpu doesn't support max_pool2d_with_indices on inputs "
            r"with more than 2147483647 spatial elements "
            r"due to int32 indices limitation for now, got 2147483648.*",
            re.DOTALL,
        ),
    ):
      y, _ = torch.nn.functional.max_pool2d(
          t,
          kernel_size=1,
          stride=1,
          padding=0,
          dilation=1,
          return_indices=True,
      )
      y.cpu()

  # TODO: remove this test once we support complex alpha on TPU.
  def test_add_complex_alpha(self):
    """Tests add with an alpha of complex dtype."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.complex64)
    s = torch.ones(4, device=device, dtype=torch.complex64)
    with et.assert_raises_message(
        NotImplementedError,
        "add(): complex128 alpha value is not supported yet",
    ):
      torch.add(t, s, alpha=1j)

  def test_threshold_unsupported_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        "threshold(): threshold is not implemented for bool type",
    ):
      torch.threshold(torch.tensor([True, False], device=et.device()), 0.5, 0.0)

  def test_threshold_unsupported_complex_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        "threshold(): threshold is not implemented for complex types",
    ):
      torch.threshold(
          torch.tensor([1 + 1j, 2 + 2j], device=et.device()), 0.5, 0.0
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # There are no other available devices on CPU runs, other than CPU.
  @parameterized.named_parameters(
      {"testcase_name": "amin", "op_name": "amin", "op": torch.amin},
      {"testcase_name": "amax", "op_name": "amax", "op": torch.amax},
      {"testcase_name": "aminmax", "op_name": "aminmax", "op": torch.aminmax},
  )
  def test_aminmax_invalid_output_device(self, op_name: str, op: Any):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = _get_aminmax_outputs(op, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"{op_name}(): expected output tensor to be on tpu, got cpu",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # This test should be run only on TPU because there are no other available
  # devices on CPU runs, other than CPU.
  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op_name": "argmin", "op": torch.argmin},
      {"testcase_name": "argmax", "op_name": "argmax", "op": torch.argmax},
  )
  def test_argmin_argmax_invalid_output_device(self, op_name: str, op: Any):
    tensor = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, 1, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"{op_name}(): expected output tensor to be on tpu, got cpu",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch core implementations don't check `pivots` rank.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_unpack_pivots_invalid_rank(self):
    data = torch.ones(2, 4, 4, device=et.device())

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="lu_unpack(): lu_pivots must have at least 1 dimension, got 0",
    ):
      pivots = torch.tensor(1, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

    # TODO: b/483972819 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "lu_unpack(): p must have one more dimension than lu_pivots, got 1"
            " and 2"
        ),
    ):
      pivots = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
      out = _make_lu_unpack_outputs(p=(4,), l=(2, 4, 4), u=(2, 4, 4))
      torch.lu_unpack(data, pivots, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch core implementations don't check `pivots` dimensions.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_unpack_pivots_invalid_dimension(self):
    data = torch.ones(2, 4, 4, device=et.device())

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "lu_unpack(): pivots size must be less than or equal to the size of"
            " the matrix, got 5 and 4"
        ),
    ):
      pivots = torch.ones(2, 5, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "lu_unpack(): pivots and tensor must have the same batch"
            " dimensions, got [2] and [3]"
        ),
    ):
      pivots = torch.ones(3, 4, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

  # Why do we run this test only on TPU (and not on CPU)?
  # CPU kernel raises an `IndexError`, instead of a `RuntimeError`, because it
  # tries to get `pivots.size(-1)` of a 0-dim tensor.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_solve_pivots_rank_too_low(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.tensor(0, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, 4, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    # TODO: b/485628812 also test CPU when the TPU kernel is fixed, raising an
    # `IndexError`, instead of an `RuntimeError`.
    with et.assert_raises_message(
        RuntimeError,
        tpu="linalg_lu_solve(): pivots must have at least 1 dimension, got 0",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)


class TpuVsCpuErrorTest(et.ErrorTestBase, parameterized.TestCase):
  """Tests error messages on TPU vs on CPU."""

  def test_triu_insufficient_dims(self):
    """Tests that triu with insufficient dims fails with expected error."""
    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This message is generated by pytorch. We don't have a good way to
        # replace it.
        "triu: input tensor must have at least 2 dimensions",
    ):
      torch.triu(t, 1)

  def test_tril_insufficient_dims(self):
    """Tests that tril with insufficient dims fails with expected error."""
    t = torch.tensor(42, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This message is generated by pytorch. We don't have a good way to
        # replace it.
        "tril: input tensor must have at least 2 dimensions",
    ):
      torch.tril(t, 0)

  def test_linalg_solve_triangular_non_sq_failure(self):
    """Tests that linalg.solve_triangular() fails with less than 2 dimensions."""
    a = torch.ones(2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "linalg_solve_triangular(): expected the first argument to have at"
        " least 2 dimensions, got 1",
        cpu=(
            "linalg.solve_triangular: The input tensor A must have at least 2"
            " dimensions."
        ),
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_linalg_solve_triangular_dim_mismatch_failure_right(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "linalg_solve_triangular(): left == False means we are solving X * A ="
        " B; expected the two inputs to have matching last dimension, got 2"
        " and 3",
        cpu=(
            "linalg.solve_triangular: Incompatible shapes of A and B for the"
            " equation XA = B (2x2 and 2x3)"
        ),
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=False, unitriangular=False
      )

  def test_linalg_solve_triangular_dim_mismatch_failure_left(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    b = torch.ones(2, 3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "linalg_solve_triangular(): left == True means we are solving A * X ="
        " B; expected the two inputs to have matching second to last dimension,"
        " got 2 and 3",
        cpu=(
            "linalg.solve_triangular: Incompatible shapes of A and B for the"
            " equation AX = B (2x2 and 3x2)"
        ),
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_linalg_solve_unsupported_dtype_failure(self):
    """Tests that linalg.solve_triangular() fails when lh and rh have mismatching dimensions."""
    a = torch.ones(2, 2, 2, device=et.device(), dtype=torch.bfloat16)
    b = torch.ones(2, 2, 2, device=et.device(), dtype=torch.bfloat16)
    with et.assert_raises_message(
        RuntimeError,
        "linalg_solve_triangular(): triangular solve not supported for dtype"
        " bfloat16",
        cpu="\"triangular_solve_cpu\" not implemented for 'BFloat16'",
    ):
      torch.linalg.solve_triangular(
          a, b, upper=True, left=True, unitriangular=False
      )

  def test_eq__for_scalar_with_integral_dtype(self):
    """Tests that eq_ with an integral dtype (e.g. uint8) doesn't fail."""
    t1 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor(3, device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 0)

    t1 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor(2, device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 1)

  def test_eq__for_tensor_with_integral_dtype(self):
    """Tests that eq_ with an integral dtype (e.g. uint8) doesn't fail."""
    # device = "cpu"
    t1 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor([3], device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 0)

    t1 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t2 = torch.tensor([2], device=et.device(), dtype=torch.uint8)
    t1.eq_(t2)  # The Boolean results should be converted to uint8 (0 or 1).
    t1.to("cpu")
    self.assertEqual(t1.dtype, torch.uint8)
    self.assertEqual(t1.item(), 1)

  def test_masked_select_with_type_mismatch(self):
    """Masked select function fails when mask has an invalid type."""
    t = torch.ones(2, 3, 3, device=et.device(), dtype=torch.float32)
    mask = torch.rand(2, 3, 3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="masked_select: expected BoolTensor for mask",
        tpu="masked_select(): expected Boolean tensor for mask, got float32",
    ):
      t.masked_select(mask)

  def test_masked_select_with_shape_mismatch(self):
    """Masked select function fails when mask has a mismatching shape."""
    t = torch.ones(2, 3, 3, device=et.device(), dtype=torch.float32)
    mask = torch.rand(2, device=et.device(), dtype=torch.float32) > 0.5

    with et.assert_raises_message(
        RuntimeError,
        "The size of tensor a (2) must match the size of tensor b (3) at"
        " non-singleton dimension 2",
    ):
      t.masked_select(mask)

  def test_index_copy_rank_mismatch(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "index_copy_(): When source and destination are not scalars,"
            " their dimensionality must match. Source dimensionality (1),"
            " destination dimensionality (2)"
        ),
        tpu=(
            "index_copy(): self and source must have the same number of"
            " dimensions, got 2 and 1"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_copy_index_rank_not_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="index_copy_(): Index should have dimension 1 or 0 (got 2)",
        tpu="index_copy(): index must be 1D, got shape [1, 1]",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([[0]], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu=(
            "Dimension out of range (expected to be in range of [-2, 1], but"
            " got 2)"
        ),
        tpu=(
            # This error is generated by PyTorch and we cannot easily replace
            # it.
            "index_copy(): Dimension out of range (expected to be in range of"
            " [-2, 1], but got 2)"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_copy(
          t, 2, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_source_dim_ne_index_size(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "index_copy_(): Number of indices (1) should be equal to"
            " source.size(dim) (2)"
        ),
        tpu=(
            "index_copy(): source must have the same number of elements as the"
            " index along dimension 0, got 2 and 1"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, 2, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_self_source_size_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "index_copy_(): Source/destination tensor must have same slice"
            " shapes. Destination slice shape: 2 at dimension 0 and source"
            " slice shape: 3 at dimension 0."
        ),
        tpu=(
            "index_copy(): self and source must have the same size along"
            " dimension 1, got 2 and 3"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_copy_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu="index_copy(): dim must be 0 for scalar input, got 1",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_copy(
          t, 1, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_copy_scalar_source_not_scalar(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "index_copy_(): Source/destination tensor must have same slice"
            " shapes. Destination slice shape:  at dimension 0 and source"
            " slice shape: 1 at dimension 0."
        ),
        tpu=(
            "index_copy(): source shape must match self shape, excluding the"
            " specified dimension, got source shape [1, 1] and self shape []"
        ),
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([[1]], device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_copy_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "index_copy_(): When source is scalar, index should have one"
            " element (got 2)"
        ),
        tpu=(
            "index_copy(): index must be 1D of size 1 for scalar input, got"
            " shape [2]"
        ),
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_copy(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_fill_with_incorrect_shape_0_dim(self):
    self.do_test_fill_with_incorrect_shape([1])

  def test_fill_with_incorrect_shape_1_dim_with_multiple_values(self):
    self.do_test_fill_with_incorrect_shape([2])

  def test_fill_with_incorrect_shape_2_dim(self):
    self.do_test_fill_with_incorrect_shape([1, 1])

  def do_test_fill_with_incorrect_shape(self, shape):
    """Tests that fill with an empty tensor fails with expected error."""
    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.float32)
    value = torch.ones(shape, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "fill_ only supports 0-dimension value tensor but got tensor with"
            f" {len(shape)} dimensions."
        ),
        tpu=(
            "fill_(): only supports 0-dimension value tensor but got tensor"
            f" with {len(shape)} dimensions."
        ),
    ):
      torch.fill(t, value)

  def test_fmod_tensor_with_unsupported_dtype(self):
    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"fmod_cpu\" not implemented for 'ComplexFloat'",
        tpu="fmod(): complex dtypes are not supported",
    ):
      torch.fmod(t, other)

    t = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    other = torch.tensor([1, 2, 3], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"fmod_cpu\" not implemented for 'Bool'",
        tpu="fmod(): boolean dtypes are not supported",
    ):
      torch.fmod(t, other)

  def test_masked_select_out_with_different_scalar_types(self):
    """Masked select function fails when self and out have different scalar types."""
    t = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.ones(5, device=et.device(), dtype=torch.int32)
    mask = torch.ones(5, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        cpu="masked_select(): self and result must have the same scalar type",
        tpu=(
            "masked_select(): the out tensor dtype is expected to be"
            " float32, got int32"
        ),
    ):
      torch.masked_select(t, mask, out=out)

  def test_mm_with_matching_sizes(self):
    """Tests that mm with matching sizes doesn't fail."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    t3 = torch.mm(t1, t2)
    self.assertEqual(t3.shape, (2, 2))

  def test_mm_with_non_2d_arg1(self):
    """Tests that mm with non-2D argument 1 fails with expected error."""
    t1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(4, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "self must be a matrix",
    ):
      torch.mm(t1, t2)

  def test_mm_with_non_2d_arg2(self):
    """Tests that mm with non-2D argument 2 fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 3, 4, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "mat2 must be a matrix",
    ):
      torch.mm(t1, t2)

  def test_mm_with_mismatched_sizes(self):
    """Tests that mm with mismatched sizes fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(4, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)",
    ):
      torch.mm(t1, t2)

  def test_mm_with_mismatched_data_types(self):
    """Tests that mm with mismatched data types fails with expected error."""
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="expected m1 and m2 to have the same dtype, but got: float != int",
        tpu=(
            "mm(): expected the two arguments to have the same dtype, got"
            " float32 vs int32"
        ),
        message_reviewed_by="wan",
    ):
      torch.mm(t1, t2)

  def test_nll_loss_unsupported_input_dtype(self):
    t = torch.ones(3, 5, device=et.device(), dtype=torch.int32)
    target = torch.tensor([1, 0, 4], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"nll_loss_out_frame\" not implemented for 'Int'",
        tpu="nll_loss_forward(): unsupported input dtype: int32",
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss_unsupported_target_dtype(self):
    t = torch.ones(3, 5, device=et.device(), dtype=torch.float32)
    target = torch.tensor([1, 0, 4], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="expected scalar type Long but found Int",
        tpu=(
            "nll_loss_forward(): expected target to have dtype int64, got int32"
        ),
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss2d_shape_mismatch(self):
    t = torch.ones(1, 3, 2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 3, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        cpu="size mismatch (got input: [1, 3, 2, 2] , target: [1, 2, 3]",
        tpu=(
            "nll_loss2d_forward(): expect the shapes of the input [N, C, d1,"
            " ..., dk] and the target [N, d1, ..., dk] (k >= 1) to match, got"
            " input: [1, 3, 2, 2], target: [1, 2, 3]"
        ),
    ):
      torch.nn.functional.nll_loss(t, target)

  def test_nll_loss_invalid_reduction(self):
    t = torch.ones(1, 3, 2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(1, 2, 3, device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        "all is not a valid value for reduction",
    ):
      torch.nn.functional.nll_loss(t, target, reduction="all")

  def test_zero_dim_size_in_ones(self):
    """Tests that torch.ones(...) with zero dimension sizes doesn't fail."""

    torch.ones(0, device=et.device(), dtype=torch.float32)
    torch.ones(3, 0, 2, device=et.device(), dtype=torch.float32)

  def test_negative_dim_size_in_ones(self):
    """Tests that torch.ones(...) with negative dimension sizes fails with expected error."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="Trying to create tensor with negative dimension -1: [-1]",
        tpu=(
            "empty(): dimension sizes must be >= 0, got [-1], which contains -1"
        ),
    ):
      torch.ones(-1, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu="Trying to create tensor with negative dimension -2: [3, -2, -4]",
        tpu=(
            "empty(): dimension sizes must be >= 0, got [3, -2, -4], which"
            " contains -2 and -4"
        ),
    ):
      torch.ones(3, -2, -4, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Trying to create tensor with negative dimension -2: [3, -2, -4,"
            " 1, -5]"
        ),
        tpu=(
            "empty(): dimension sizes must be >= 0, got [3, -2, -4, 1, -5],"
            " which contains -2, -4, and -5"
        ),
    ):
      torch.ones(3, -2, -4, 1, -5, device=et.device(), dtype=torch.float32)

  def test_dim_size_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the dimension size overlows."""
    with et.assert_raises_message(
        TypeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        re.compile(r".*Overflow when unpacking long.*", re.DOTALL),
    ):
      # The dimension size 2**63 fits in uint64_t but not in int64_t.
      torch.ones(2**63, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        TypeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        re.compile(r".*Overflow when unpacking long.*", re.DOTALL),
    ):
      # The dimension size 2**64 fits in neither uint64_t nor int64_t.
      torch.ones(1, 2**64, device=et.device(), dtype=torch.float32)

  def test_sign_unsupported_complex(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        "Unlike NumPy, torch.sign is not intended to support complex numbers."
        " Please use torch.sgn instead.",
    ):
      torch.sign(torch.zeros(1, device=et.device(), dtype=torch.complex64))

  def test_size_product_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the size product is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Storage size calculation overflowed with sizes=[2147483648,"
            " 4294967296]"
        ),
        tpu=(
            "empty(): product of dimension sizes [2147483648, 4294967296]"
            " overflows as int64"
        ),
    ):
      # The product of the dimensions is 2 ** 63, which doesn't cause an
      # overflow in XLA. However, it doesn't fit in int64_t.
      torch.ones(2**31, 2**32, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Storage size calculation overflowed with sizes=[1073741824,"
            " 1073741824, 1073741824]"
        ),
        tpu=(
            "empty(): product of dimension"
            " sizes [1073741824, 1073741824, 1073741824] overflows as int64"
        ),
    ):
      # The product of the dimensions is 2 ** 90, which causes an overflow in
      # XLA.
      torch.ones(2**30, 2**30, 2**30, device=et.device(), dtype=torch.float32)

  def test_byte_size_overflow_in_ones(self):
    """Tests that torch.ones() fails with expected error when the byte size overlows."""
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Storage size calculation overflowed with sizes=[2147483648,"
            " 2147483648]"
        ),
        tpu=(
            "empty(): product of dimension sizes [2147483648, 2147483648] and"
            " size of f32 (4 bytes) overflows as int64"
        ),
    ):
      # The product of the dimensions is 2 ** 62, which doesn't cause an
      # overflow in XLA. However, the byte size is 2 ** 64, which overflows
      # int64_t.
      torch.ones(2**31, 2**31, device=et.device(), dtype=torch.float32)

  def test_oom_in_ones(self):
    """Tests that torch.ones() OOMs on CPU, but not on TPU."""
    if _TEST_MODE.value == "cpu":
      # CPU immediately tries to allocate 4 exabytes of memory and crashes.
      with et.assert_raises_message(
          RuntimeError,
          cpu=(
              re.compile(
                  r"\[enforce fail at .+\] err == 0\."
                  r" DefaultCPUAllocator: can't allocate memory: you tried to"
                  r" allocate 4000000000000000000 bytes. Error code 12 \(Cannot"
                  r" allocate memory\)"
              )
          ),
          tpu="error message is not used in this test",
      ):
        torch.ones(
            1_000_000_000,
            1_000_000_000,
            device=et.device(),
            dtype=torch.float32,
        )
    elif _TEST_MODE.value in ("tpu", "cov"):
      # TPU defers allocation until execution time, so it should not OOM.
      torch.ones(
          1_000_000_000, 1_000_000_000, device=et.device(), dtype=torch.float32
      )

  def test_histc_bounds_overflow(self):
    """Tests that torch.histc() fails when the bounds overflow."""
    max_int32 = torch.iinfo(torch.int32).max
    t = torch.tensor(
        [max_int32, max_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        cpu="value cannot be converted to type int without overflow",
        tpu=(
            "histc(): expected min and max to be within the range of their data"
            " types, but got min = 2147483646 and max = -2147483648. This"
            " happened because min and max were adjusted by one (due to min =="
            " max), which resulted in an overflow."
        ),
    ):
      torch.histc(t)

  def test_histc_bounds_underflow(self):
    """Tests that torch.histc() fails when the bounds underflow."""
    min_int32 = torch.iinfo(torch.int32).min
    t = torch.tensor(
        [min_int32, min_int32], device=et.device(), dtype=torch.int32
    )
    with et.assert_raises_message(
        RuntimeError,
        cpu="value cannot be converted to type int without overflow",
        tpu=(
            "histc(): expected min and max to be within the range of their data"
            " types, but got min = 2147483647 and max = -2147483647. This"
            " happened because min and max were adjusted by one (due to min =="
            " max), which resulted in an overflow."
        ),
    ):
      torch.histc(t)

  def test_histc_bounds_not_nan(self):
    """Tests that torch.histc() fails when the bounds are NaN."""
    t = torch.tensor([0, float("nan")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="torch.histc: range of [-nan, -nan] is not finite",
        tpu=(
            "histc(): expected min and max to be finite, got nan and nan."
            " Either make sure that the input data is finite, or provide valid"
            " finite bounds."
        ),
    ):
      torch.histc(t)

  def test_histc_bounds_inf(self):
    """Tests that torch.histc() fails when the bounds are infinity."""
    t = torch.tensor([0, float("inf")], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="torch.histc: range of [0, inf] is not finite",
        tpu=(
            "histc(): expected min and max to be finite, got 0 and inf. Either"
            " make sure that the input data is finite, or provide valid finite"
            " bounds."
        ),
    ):
      torch.histc(t)

  def test_histc_bounds_not_in_order(self):
    """Tests that torch.histc() fails when the bounds are not in order."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="torch.histc: max must be larger than min",
        tpu="histc(): expected min <= max, got 1 vs 0",
    ):
      torch.histc(t, min=1, max=0)

  def test_invalid_index_dtype_in_take(self):
    """Tests that torch.take() fails when the index has the wrong dtype."""
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="take(): Expected a long tensor for index, but got Int",
        tpu="take(): expected index dtype to be int64, got int32",
    ):
      torch.take(t, torch.tensor([0, 1], dtype=torch.int32, device=et.device()))

  def test_empty_tensor_in_take(self):
    """Tests that torch.take() fails when the input tensor is empty but index is not."""
    t = torch.tensor([], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        cpu="take(): tried to take from an empty tensor",
        tpu=(
            "take(): input tensor must be non-empty when the index tensor is"
            " non-empty"
        ),
    ):
      torch.take(t, torch.tensor([0], dtype=torch.int64, device=et.device()))

  def test_invalid_index_in_take(self):
    """Tests that torch.take() fails when the index is invalid."""
    t = torch.tensor([0, 1, 2], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        IndexError,
        cpu="out of range: tried to access index 3 on a tensor of 3 elements.",
        tpu="take(): expected indices to be in range [-3, 2], got 3",
    ):
      torch.take(t, torch.tensor([0, 3], dtype=torch.int64, device=et.device()))

    with et.assert_raises_message(
        IndexError,
        cpu="out of range: tried to access index -4 on a tensor of 3 elements.",
        tpu="take(): expected indices to be in range [-3, 2], got -4",
    ):
      torch.take(
          t, torch.tensor([0, -4], dtype=torch.int64, device=et.device())
      )

  def test_invalid_size_in_zeros(self):
    """Tests that torch.zeros() fails with expected error when the size is invalid."""
    # This error message is generated by PyTorch. In newer PyTorch v2.9.0, shape
    # validation for factory functions (like zeros) was moved to the frontend,
    # meaning it catches the error earlier and produces a simpler, more generic
    # error message different from the one produced in the previous PyTorch
    # v2.8.0.
    with et.assert_raises_message(
        RuntimeError,
        "zeros: Dimension size must be non-negative.",
    ):
      torch.zeros(-1, device=et.device(), dtype=torch.float32)

  def test_invalid_size_in_empty_memory_format(self):
    """Tests that torch.empty() fails with expected error when the size is invalid."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="Trying to create tensor with negative dimension -1: [-1]",
        tpu=(
            "empty(): dimension sizes must be >= 0, got [-1], which contains -1"
        ),
    ):
      torch.empty(-1, device=et.device(), dtype=torch.float32)

  def test_invalid_broadcast_in_binary_op_add(self):
    """Tests that torch.add() fails with expected error when the sizes are mismatched."""

    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "The size of tensor a (3) must match the size of tensor b (2) at"
            " non-singleton dimension 1"
        ),
        tpu=(
            # This error is generated by pytorch. We don't have a good way to
            # replace it.
            "The size of tensor a (3) must match the size of tensor b (2) at"
            " non-singleton dimension 1"
        ),
    ):
      torch.add(t1, t2)

  def test_round_decimals_param_integer_input(self):
    """torch.round() errors when input is an integer and decimals is specified."""

    t = torch.ones(1, device=et.device(), dtype=torch.int64)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"round_cpu\" not implemented for 'Long'",
        tpu="round(): dtype int64 is not supported when decimals is specified",
    ):
      t.round(decimals=-1)

    t = torch.ones(1, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"round_cpu\" not implemented for 'Int'",
        tpu="round(): dtype int32 is not supported when decimals is specified",
    ):
      t.round_(decimals=2)

    t = torch.ones(1, device=et.device(), dtype=torch.int16)
    out_t = torch.zeros(1, device=et.device(), dtype=torch.int16)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"round_vml_cpu\" not implemented for 'Short'",
        tpu="round(): dtype int16 is not supported when decimals is specified",
    ):
      torch.round(t, decimals=0, out=out_t)

  def test_round_invalid_input_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"round_vml_cpu\" not implemented for 'Bool'",
        tpu="round(): dtype bool is not supported",
    ):
      torch.round(t)

  def test_roll_errors(self):
    """roll() fails when input parameters are invalid."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="shifts and dimensions must align. shifts: 2, dims:1",
        tpu="roll(): shifts and dims must align, got shifts: 2, dims: 1",
    ):
      torch.roll(t, shifts=(2, 3), dims=(0,))

    with et.assert_raises_message(
        RuntimeError,
        cpu="shifts and dimensions must align. shifts: 2, dims:0",
        tpu="roll(): shifts and dims must align, got shifts: 2, dims: 0",
    ):
      torch.roll(t, shifts=(2, 3))

  def test_reduction_dim_out_of_bounds_var(self):
    self.do_test_reduction_dim_out_of_bounds(torch.var)

  def test_reduction_dim_out_of_bounds_mean(self):
    self.do_test_reduction_dim_out_of_bounds(torch.mean)

  def test_reduction_dim_out_of_bounds_sum(self):
    self.do_test_reduction_dim_out_of_bounds(torch.sum)

  def do_test_reduction_dim_out_of_bounds(self, reduction_fn):
    """Reduction function fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    for dim in [-3, 3]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by PyTorch.
          "Dimension out of range (expected to be in range of [-2, 1], but"
          f" got {dim})",
      ):
        reduction_fn(t, dim=dim)

  def test_reduction_dim_repeated_var(self):
    self.do_test_reduction_dim_repeated(torch.var)

  def test_reduction_dim_repeated_mean(self):
    self.do_test_reduction_dim_repeated(torch.mean)

  def test_reduction_dim_repeated_sum(self):
    self.do_test_reduction_dim_repeated(torch.sum)

  def do_test_reduction_dim_repeated(self, reduction_fn):
    """Reduction function fails when canonical dimension is repeated."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch.
        "dim 1 appears multiple times in the list of dims",
    ):
      reduction_fn(t, dim=[-1, 1])

  def test_reduction_dim_scalar_var(self):
    self.do_test_reduction_dim_scalar(torch.var)

  def test_reduction_dim_scalar_mean(self):
    self.do_test_reduction_dim_scalar(torch.mean)

  def test_reduction_dim_scalar_sum(self):
    self.do_test_reduction_dim_scalar(torch.sum)

  def do_test_reduction_dim_scalar(self, reduction_fn):
    """Reduction function fails when dimension is invalid for scalar."""
    t_rank0 = torch.tensor(1.0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        IndexError,
        # This error is generated by PyTorch.
        "Dimension out of range (expected to be in range of [-1, 0], "
        "but got 1)",
    ):
      reduction_fn(t_rank0, dim=1)

  def test_reduction_unsupported_int_dtype_mean(self):
    """Mean fails for integral dtypes."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by PyTorch.
        "mean(): could not infer output dtype. Input dtype must be either a"
        " floating point or complex dtype. Got: Int",
    ):
      torch.mean(t, dim=0)

  def test_reduction_unsupported_int_dtype_var(self):
    """Var fails for integral dtypes."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="std and var only support floating point and complex dtypes",
        tpu="var(): expected a floating point or complex dtype, got int32",
    ):
      torch.var(t, dim=0)

  def test_unfold_size_too_large(self):
    """Unfold fails when size is larger than dimension."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="maximum size for tensor at dimension 0 is 5 but size is 6",
        tpu=(
            "unfold(): expected size <= dimension size (shape[0]: 5), got"
            " size: 6"
        ),
    ):
      t.unfold(0, 6, 1)

  def test_unfold_dim_out_of_bounds(self):
    """Unfold fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device())
    with et.assert_raises_message(
        IndexError,
        cpu=(
            "Dimension out of range (expected to be in range of [-2, 1], but"
            " got 2)"
        ),
        tpu=(
            "unfold(): expected dimension to be in range of [-2, 1] for shape"
            " [2, 3], got 2"
        ),
    ):
      t.unfold(2, 1, 1)

  def test_unfold_zero_step(self):
    """Unfold fails when step is 0."""
    t = torch.ones(5, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        cpu="step is 0 but must be > 0",
        tpu="unfold(): expected step > 0, got 0",
    ):
      t.unfold(0, 2, 0)

  def test_var_negative_reduction_factor(self):
    """A warning is issued and nan returned when the variance degrees of freedom is <= 0."""
    # TODO: b/435570003 - Create a utility to compare warning messages.
    cpu_warn_msg = (
        "var(): degrees of freedom is <= 0. Correction should be strictly less"
        " than the reduction factor (input numel divided by output numel)."
    )

    t = torch.ones(1, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 1, correction = 1, and degrees of"
        " freedom = 0"
    )

    warn_msg = cpu_warn_msg if et.device().type == "cpu" else tpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, correction=1)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

    t = torch.ones(3, 2, device=et.device(), dtype=torch.float32)
    tpu_warn_msg = (
        "var(): degrees of freedom (i.e., reduction size - correction) should"
        " be positive, got reduction size = 3, correction = 4, and degrees of"
        " freedom = -1"
    )
    warn_msg = cpu_warn_msg if et.device().type == "cpu" else tpu_warn_msg

    with self.assertWarnsRegex(UserWarning, re.escape(warn_msg)):
      result = torch.var(t, dim=0, correction=4)
    self.assertTrue(torch.all(torch.isnan(result)), f"Got {result.to('cpu')}")

  def test_view_not_contiguity_like(self):
    t = torch.randn(2, 3).T
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        "view size is not compatible with input tensor's size and stride"
        " (at least one dimension spans across two contiguous subspaces)."
        " Use .reshape(...) instead.",
    ):
      t.view(6)

  def test_view_as_real_non_complex(self):
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        cpu="view_as_real is only supported for complex tensors",
        tpu=(
            "view_as_real(): expected complex dtypes (torch.complex64 and"
            " torch.complex128), got float32"
        ),
    ):
      torch.view_as_real(t)

  def test_view_as_complex_unsupported_dtypes(self):
    device = et.device()
    t1 = torch.ones(2, 3, 2, device=device, dtype=torch.float16)

    if _TEST_MODE.value == "cpu":
      # view_as_complex supports float16 on CPU.
      torch.view_as_complex(t1)
      return

    with et.assert_raises_message(
        RuntimeError,
        "view_as_complex(): this op currently only supports float32 and float64"
        " dtype as input, got float16",
    ):
      torch.view_as_complex(t1)

  def test_select_index_out_of_bounds(self):
    """Select function fails when index is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    dim = 1

    for index in [-40, 40]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by pytorch before our kernel is called;
          # we don't have control over this error message.
          f"select(): index {index} out of range for tensor of size [2, 3]"
          f" at dimension {dim}",
      ):
        t.select(dim, index)  # pylint: disable=unused-variable

  def test_select_dim_out_of_bounds(self):
    """Select function fails when dimension is out of bounds."""
    t = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    for dim in [-3, 3]:
      with et.assert_raises_message(
          IndexError,
          # This error is generated by pytorch before our kernel is called;
          # we don't have control over this error message.
          "Dimension out of range (expected to be in range of [-2, 1], but"
          f" got {dim})",
      ):
        t.select(dim, 1)  # pylint: disable=unused-variable

  def test_slice_on_scalar(self):
    t = torch.scalar_tensor(1.0, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        IndexError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        "slice() cannot be applied to a 0-dim tensor.",
    ):
      sliced_t = t[0:1:1]  # pylint: disable=unused-variable

  def test_slice_zero_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        "slice step cannot be zero",
    ):
      sliced_t = t[0:10:0]  # pylint: disable=unused-variable

  def test_slice_negative_step(self):
    t = torch.ones(10, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before redispatching to as_strided;
        # we don't have control over this error message.
        "step must be greater than zero",
    ):
      sliced_t = t[0:10:-1]  # pylint: disable=unused-variable

  def test_cat_empty_input(self):
    with et.assert_raises_message(
        ValueError,
        # This error is generated by pytorch before our kernel is called;
        # we don't have control over this error message.
        "torch.cat(): expected a non-empty list of Tensors",
    ):
      torch.cat([])

  def test_cat_scalar_input(self):
    with et.assert_raises_message(
        RuntimeError,
        "zero-dimensional tensor (at position 0) cannot be concatenated",
    ):
      torch.cat([torch.tensor(1.0, device=et.device())])
    with et.assert_raises_message(
        RuntimeError,
        "zero-dimensional tensor (at position 1) cannot be concatenated",
    ):
      torch.cat([
          torch.tensor([], dtype=torch.float32, device=et.device()),
          torch.tensor(1.0, device=et.device()),
      ])

  def test_cat_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        "Dimension out of range (expected to be in range of [-1, 0], but"
        " got 1)",
    ):
      t0 = torch.tensor([], dtype=torch.float32, device=et.device())
      t3 = torch.tensor([1, 2, 3], device=et.device())
      torch.cat([t0, t3], dim=1)

  def test_cat_mismatched_dims(self):
    t3 = torch.tensor([1, 2, 3], device=et.device())
    t1x3 = torch.tensor([[1, 2, 3]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        "Tensors must have same number of dimensions: got 1 and 2",
    ):
      torch.cat([t3, t1x3])

  def test_cat_mismatched_dim_sizes(self):
    t2x2 = torch.tensor([[1, 2], [3, 4]], device=et.device())
    t2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]], device=et.device())
    t3x2 = torch.tensor([[1, 2], [3, 4], [5, 6]], device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        "Sizes of tensors must match except in dimension 0. Expected size 2 but"
        " got size 3 for tensor number 1 in the list.",
    ):
      torch.cat([t2x2, t2x3])
    with et.assert_raises_message(
        RuntimeError,
        "Sizes of tensors must match except in dimension 1. Expected size 2 but"
        " got size 3 for tensor number 1 in the list.",
    ):
      torch.cat([t2x2, t3x2], dim=1)

  def test_addcmul_bool_error(self):
    """Tests that addcmul errors out with bool inputs."""
    self_tensor = torch.tensor([[True] * 5] * 5, dtype=torch.bool).to(
        et.device()
    )
    tensor1 = torch.tensor(True, dtype=torch.bool).to(et.device())
    tensor2 = torch.tensor(True, dtype=torch.bool).to(et.device())
    value = 1
    with et.assert_raises_message(
        RuntimeError,
        cpu="\"addcmul_cpu_out\" not implemented for 'Bool'",
        tpu=(
            "addcmul(): bool tensors are not supported, got input: bool,"
            " tensor1: bool, tensor2: bool"
        ),
    ):
      torch.addcmul(self_tensor, tensor1, tensor2, value=value)

  def test_index_put_too_many_indices_error(self):
    # TODO(mkkhanna): Fix exception type for TPU.
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="too many indices for tensor of dimension 1 (got 2)",
        tpu="index_put_(): too many indices for tensor of dimension 1, got 2",
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (
              torch.tensor([0], device=et.device()),
              torch.tensor([0], device=et.device()),
          ),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_index_dtype_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="tensors used as indices must be long, int, byte or bool tensors",
        tpu=(
            "index_put_(): tensors used as indices must be long, int, byte or"
            " bool tensors, got float32 at index 0"
        ),
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([0], dtype=torch.float32, device=et.device()),),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_broadcast_indices_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "shape mismatch: indexing tensors could not be"
            " broadcast together with shapes [2], [3]"
        ),
        tpu=(
            "index_put_(): index tensors not broadcastable, got index tensor"
            " shape [3] and broadcast shape [2]: The size of tensor a (2) must"
            " match the size of tensor b (3) at non-singleton dimension 0"
        ),
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1, 1], device=et.device()),
          ),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_broadcast_values_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "shape mismatch: value tensor of shape [2, 2] cannot be broadcast"
            " to indexing result of shape [2]"
        ),
        tpu=(
            "index_put_(): value tensor of shape [2, 2] cannot be broadcast"
            " to indexing result of shape [2]"
        ),
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1], device=et.device()),
          ),
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
      )

  def test_index_put_dtype_mismatch_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Index put requires the source and destination dtypes"
            " match, got Int for the destination and Long for the source."
        ),
        tpu=(
            "index_put_(): dtypes of values and destination must be the same,"
            " got int64 and int32"
        ),
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=et.device()),
          (
              torch.tensor([0, 1], device=et.device()),
              torch.tensor([0, 1], device=et.device()),
          ),
          torch.tensor([0], dtype=torch.int64, device=et.device()),
      )

  def test_index_put_index_or_indices_must_be_specified_error(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu="sym_strides() called on an undefined Tensor",
        tpu="index_put_(): indices must be specified",
    ):
      torch.index_put_(
          torch.tensor([[0, 1], [2, 3]], device=et.device()),
          (),
          torch.tensor([0], device=et.device()),
      )

  def test_index_put_decompose_with_mask_mask_shape_mismatch(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "The shape of the mask [3, 5] at index 0 does not"
            " match the shape of the indexed tensor [2, 5] at index 0"
        ),
        tpu=(
            "index_put_(): the shape of the mask at index 0 must match the"
            " shape of the indexed tensor at index 0, got mask shape [3, 5] and"
            " indexed tensor shape [2, 5]"
        ),
    ):
      tensor = torch.arange(10).view(2, 5).to(et.device())
      tensor_other = torch.arange(15).view(3, 5).to(et.device())
      boolean_mask = tensor_other % 2 != 0
      tensor[boolean_mask] = 100

  def test_index_put_decompose_with_mask_error_mask_dim_more_than_indexed_tensor_dim(
      self,
  ):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu=(
            "index_put_(): the shape of the mask at index 1 must match the"
            " shape of the indexed tensor at index 1, got mask shape [2, 2] and"
            " indexed tensor shape [2]"
        ),
    ):
      torch.index_put_(
          torch.tensor([0, 1], device=et.device()),
          (torch.tensor([[True, False], [False, True]], device=et.device()),),
          torch.tensor(0, device=et.device()),
      )

  def test_index_put_decompose_with_multiple_mask_error(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "The shape of the mask [3] at index 0 does not match"
            " the shape of the indexed tensor [2, 3, 5, 9] at index 2"
        ),
        tpu=(
            "index_put_(): the shape of the mask at index 0 must match the"
            " shape of the indexed tensor at index 2, got mask shape [3] and"
            " indexed tensor shape [2, 3, 5, 9]"
        ),
    ):
      tensor = torch.arange(270).view(2, 3, 5, 9).to(et.device())
      boolean_mask_dim1 = tensor[0, :, 0, 0] % 2 != 0
      boolean_mask_dim3 = tensor[0, 0, 0, :] % 2 != 0
      tensor[:, :, boolean_mask_dim1, boolean_mask_dim3] = 100

  def test_index_select_index_must_be_1d(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="index_select(): Index is supposed to be a vector",
        tpu="index_select(): index must be 1D, got shape [2, 3]",
    ):
      torch.index_select(
          torch.ones(2, 3, device=et.device()),
          1,
          torch.ones(2, 3, device=et.device(), dtype=torch.long),
      )

  def test_index_select_dim_out_of_bounds(self):
    with et.assert_raises_message(
        IndexError,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu=(
            # This error is generated by PyTorch and we cannot easily replace
            # it.
            "index_select(): Dimension out of range (expected to be in range of"
            " [-1, 0], but got 1)"
        ),
    ):
      torch.index_select(
          torch.ones(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_input(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu="index_select(): dim must be 0 for scalar input, got 1",
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          1,
          torch.tensor([0], device=et.device(), dtype=torch.long),
      )

  def test_index_select_scalar_index(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "index_select(): Index to scalar can have only 1 value, got 2"
            " value(s)"
        ),
        tpu=(
            "index_select(): index must be 1D of size 1 for scalar input, got"
            " shape [2]"
        ),
    ):
      torch.index_select(
          torch.tensor(1, device=et.device()),
          0,
          torch.tensor([0, 0], device=et.device(), dtype=torch.long),
      )

  def test_cumsum_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.device().type == "tpu" else RuntimeError,
        cpu=(
            "Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got"
            " float instead"
        ),
        tpu="cumsum(): TorchTPU does not yet support dtype int1",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, dtype=torch.int1, out=output)

  def test_cumsum_dimension_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu=(
            # This error is generated by PyTorch and we cannot easily replace
            # it.
            "cumsum(): Dimension out of range (expected to be in range of"
            " [-1, 0], but got 1)"
        ),
    ):
      t = torch.ones(1, device=et.device())
      output = torch.empty_like(t)
      torch.cumsum(t, dim=1, out=output)

  def test_prod_out_with_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError if et.device().type == "tpu" else RuntimeError,
        cpu=(
            "Expected out tensor to have dtype c10::dummy_int1_7_t<1>, but got"
            " float instead"
        ),
        tpu="prod(): TorchTPU does not yet support dtype int1",
        message_reviewed_by="wan",
    ):
      t = torch.ones(2, 2, device=et.device())
      output = torch.empty_like(t)
      torch.prod(t, dim=1, dtype=torch.int1, out=output)

  def test_index_add_rank_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "index_add_(): Number of indices (1) should be equal to"
            " source.size(dim): (2), for dim: 0"
        ),
        tpu=(
            "index_add(): self and source must have the same number of"
            " dimensions, got 2 and 1"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(2, device=et.device())
      )

  def test_index_add_index_rank_not_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "index_add_(): Index is supposed to be a vector, but got dim: 2"
            " with type: Long and size: [1, 1]"
        ),
        tpu="index_add(): index must be 1D, got shape [1, 1]",
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([[0]], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_dim_out_of_range(self):
    with et.assert_raises_message(
        IndexError,
        cpu=(
            "Dimension out of range (expected to be in range of [-2, 1], but"
            " got 2)"
        ),
        tpu=(
            # This error is generated by PyTorch and we cannot easily replace
            # it.
            "index_add(): Dimension out of range (expected to be in range of"
            " [-2, 1], but got 2)"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 2, device=et.device())
      torch.index_add(
          t, 2, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_source_dim_ne_index_size(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "index_add_(): Number of indices (1) should be equal to"
            " source.size(dim): (2), for dim: 0"
        ),
        tpu=(
            "index_add(): source must have the same number of elements as the"
            " index along dimension 0, got 2 and 1"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(2, 2, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_self_source_size_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "source tensor shape must match self tensor shape, excluding the"
            " specified dimension. Got self.shape = [2, 2] source.shape ="
            " [1, 3]"
        ),
        tpu=(
            "index_add(): self and source must have the same size along"
            " dimension 1, got 2 and 3"
        ),
    ):
      t = torch.ones(2, 2, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.ones(1, 3, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.ones(1, device=et.device())
      )

  def test_index_add_scalar_dim_out_of_range(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "Dimension out of range (expected to be in range of [-1, 0], but"
            " got 1)"
        ),
        tpu="index_add(): dim must be 0 for scalar input, got 1",
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_add(
          t, 1, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_add_scalar_source_not_scalar(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "source tensor shape must match self tensor shape, excluding the"
            " specified dimension. Got self.shape = [] source.shape = [1]"
        ),
        tpu=(
            "index_add(): source shape must match self shape, excluding the"
            " specified dimension, got source shape [1] and self shape []"
        ),
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0], device=et.device(), dtype=torch.long)
      source = torch.tensor([1], device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_index_add_scalar_index_size_ne_1(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu="Dimension specified as 0 but tensor has no dimensions",
        tpu=(
            "index_add(): index must be 1D of size 1 for scalar input, got"
            " shape [2]"
        ),
    ):
      t = torch.tensor(1, device=et.device())
      index = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
      source = torch.tensor(1, device=et.device())
      torch.index_add(
          t, 0, index, source, out=torch.tensor(1, device=et.device())
      )

  def test_addmm_input_rank_larger_than_matrices(self):
    # Arrange
    input_ = torch.ones(2, 2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)

    cpu_msg = (
        "expand(torch.FloatTensor{[2, 2, 2]}, size=[2, 2]): the number of sizes"
        " provided (2) must be greater or equal to the number of dimensions in"
        " the tensor (3)"
    )
    tpu_msg = (
        "addmm(): input tensor should not have more dimensions than the "
        "product of mat1 @ mat2, got 3-D input and 2-D product of mat1 @ mat2"
    )

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_not_broadcastable_to_matmul_result(self):
    # Arrange
    input_ = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    cpu_msg = (
        "The expanded size of the tensor (2) must match the existing size (3)"
        " at non-singleton dimension 1.  Target sizes: [2, 2].  Tensor sizes:"
        " [2, 3]"
    )
    tpu_msg = (
        "addmm(): input tensor shape [2, 3] cannot be broadcasted to matmul "
        "result shape [2, 2]"
    )

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_input_on_bool_tensor(self):
    # Arrange
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.bool)
    beta = True
    alpha = True
    cpu_msg = "\"addmm_impl_cpu_\" not implemented for 'Bool'"
    tpu_msg = "addmm(): boolean dtypes are not supported"

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, beta=beta, alpha=alpha)

  def test_addmm_on_non_matrix_mat1(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        "mat1 must be a matrix, got 1-D tensor",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_non_matrix_mat2(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        "mat2 must be a matrix, got 1-D tensor",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mat1_mat2_mismatch_contracting_dimension(self):
    input_ = torch.ones(13, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(3, 13, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(11, 2, device=et.device(), dtype=torch.float32)
    # This error is created by pytorch before our kernel is called.
    with et.assert_raises_message(
        RuntimeError,
        "mat1 and mat2 shapes cannot be multiplied (3x13 and 11x2)",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_addmm_on_mismatched_input_and_out_dtypes(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.int32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    tpu_msg = (
        "addmm(): input and out tensors expected to have the same dtype, got"
        " input dtype float32 and out dtype int32"
    )
    cpu_msg = "Expected out tensor to have dtype float, but got int instead"

    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, out=out)

  def test_addmm_outdtype_must_match_out_dtype(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int32
    tpu_msg = (
        "addmm(): out dtype should match out_dtype, got out dtype"
        " float32 and out_dtype int32"
    )
    cpu_msg = re.compile(
        r"Could not run 'aten::addmm.dtype_out' with arguments from the 'CPU'"
        r" backend.*",
        re.DOTALL,
    )

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(RuntimeError, cpu=cpu_msg, tpu=tpu_msg):
      torch.addmm(input_, mat1, mat2, out=out, out_dtype=out_dtype)

  def test_addmm_out_dtype_unsupported(self):
    input_ = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat1 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    mat2 = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    out_dtype = torch.int1
    tpu_msg = "addmm(): TorchTPU does not yet support the output dtype int1"
    cpu_msg = re.compile(
        r"^Could not run 'aten::addmm.dtype' with arguments from the 'CPU'"
        r" backend.*$",
        re.DOTALL,
    )

    # CPU raises NotImplementedError, a subclass of RuntimeError.
    with et.assert_raises_message(
        RuntimeError, cpu=cpu_msg, tpu=tpu_msg, message_reviewed_by="wan"
    ):
      torch.addmm(input_, mat1, mat2, out_dtype=out_dtype)

  def test_empty_strided_size_stride_mismatch(self):
    """Tests that empty_strided fails with expected error when size and stride arrays have different lengths."""
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "dimensionality of sizes (2) must match dimensionality of"
            " strides (1)"
        ),
        tpu=(
            "empty_strided(): the dimensionality of sizes must be the same as"
            " strides, got size [2] and stride [1]"
        ),
    ):
      torch.empty_strided((2, 3), (1,), device=et.device(), dtype=torch.float32)

  def test_empty_strided_negative_size(self):
    """Tests that empty_strided fails with expected error when size is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu="Trying to create tensor with negative dimension -1: [-1, 2]",
        tpu="empty_strided(): size must be nonnegative, got sizes [-1, 2]",
    ):
      torch.empty_strided(
          (-1, 2), (2, 1), device=et.device(), dtype=torch.float32
      )

  def test_empty_strided_negative_stride(self):
    """Tests that empty_strided fails with expected error when stride is negative."""
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Storage size calculation overflowed with sizes=[2, 2] and"
            " strides=[2, -1]"
        ),
        tpu="empty_strided(): stride must be nonnegative, got strides [2, -1]",
    ):
      torch.empty_strided(
          (2, 2), (2, -1), device=et.device(), dtype=torch.float32
      )

  def test_index_put_too_many_indices_after_expanding_boolean_tensors(self):
    err_type = RuntimeError if et.device().type == "tpu" else IndexError
    with et.assert_raises_message(
        err_type,
        cpu=(
            "Dimension out of range (expected to be in range of [-2, 1], but"
            " got 2)"
        ),
        tpu=(
            "index_put_(): too many indices for tensor of dimension 2, got 3"
            " index tensors after expanding boolean indices"
        ),
    ):
      t = torch.zeros(10, 20, device=et.device())
      mask = torch.zeros(10, 20, dtype=torch.bool, device=et.device())
      mask[0, 0] = True
      int_index = torch.tensor([0], device=et.device())
      values = torch.tensor([5.0], device=et.device())
      # The boolean mask has dimension 2, so it expands to 2 index tensors.
      # Together with int_index, we have 3 index tensors in total, which is
      # greater than the dimension of t (2).
      t.index_put_((mask, int_index), values)

  def test_softmax_backward_data_out_unsupported_dtype(self):
    tensor = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    tensor.requires_grad_()
    tensor_int = tensor.to(torch.int32)
    dim = 1
    with et.assert_raises_message(
        NotImplementedError,
        cpu="\"softmax_lastdim_kernel_impl\" not implemented for 'Int'",
        tpu="softmax(): not implemented for input type int32",
    ):
      torch.nn.functional.softmax(tensor_int, dim).backward(
          torch.randn(2, 3, device=et.device())
      )

  def test_log_softmax_backward_data_out_unsupported_dtype(self):
    tensor = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    tensor.requires_grad_()
    tensor_int = tensor.to(torch.int32)
    dim = 1
    with et.assert_raises_message(
        NotImplementedError,
        cpu="\"log_softmax_lastdim_kernel_impl\" not implemented for 'Int'",
        tpu="log_softmax(): not implemented for input type int32",
    ):
      torch.nn.functional.log_softmax(tensor_int, dim).backward(
          torch.randn(2, 3, device=et.device())
      )

  def test_trunc_unsupported_boolean_dtype(self):
    t = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="trunc(): does not support boolean types",
        cpu="\"trunc_vml_cpu\" not implemented for 'Bool'",
    ):
      torch.trunc(t)

  def test_elu_unsupported_dtypes(self):
    t_bool = torch.tensor([True, False], device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="elu(): only float dtypes are supported",
        cpu="\"elu_cpu\" not implemented for 'Bool'",
    ):
      torch.nn.functional.elu(t_bool)

    t_long = torch.tensor([0, 0], device=et.device(), dtype=torch.long)
    with et.assert_raises_message(
        RuntimeError,
        tpu="elu(): only float dtypes are supported",
        cpu="\"elu_cpu\" not implemented for 'Long'",
    ):
      torch.nn.functional.elu(t_long)

  def test_gelu_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="gelu(): unsupported approximate argument: invalid",
        cpu="approximate argument must be either none or tanh.",
    ):
      torch.nn.functional.gelu(t, approximate="invalid")

  def test_gelu_backward_grad_input_unsupported_approximation_type(self):
    t = torch.randn(2, 3, device=et.device(), dtype=torch.float32)
    grad_input = torch.empty_like(t)
    with et.assert_raises_message(
        RuntimeError,
        tpu="gelu_backward(): unsupported approximate argument: invalid",
        cpu="approximate argument must be either none or tanh.",
    ):
      torch.ops.aten.gelu_backward.grad_input(
          t, t, approximate="invalid", grad_input=grad_input
      )

  def test_group_norm_backward_grad_out_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Expected dY.numel() == N * C * HxW to be true, but got false."
            "  (Could this error message be improved?"
            "  If so, please report an enhancement request to PyTorch.)"
        ),
        tpu=(
            "native_group_norm_backward(): expected grad_out to"
            " have 18 elements, got 24"
        ),
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 4, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_input_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Expected X.numel() == N * C * HxW to be true, but got false."
            "  (Could this error message be improved?"
            "  If so, please report an enhancement request to PyTorch.)"
        ),
        tpu=(
            "native_group_norm_backward(): expected input to have 24 elements,"
            " got 18"
        ),
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 4, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          4,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_mean_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Expected mean.numel() == N * group to be true, but got false."
            "  (Could this error message be improved?"
            "  If so, please report an enhancement request to PyTorch.)"
        ),
        tpu=(
            "native_group_norm_backward(): expected mean to have shape [1, 2],"
            " got [1, 3]"
        ),
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_mean_rstd_shape_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Expected rstd.numel() == N * group to be true, but got false."
            "  (Could this error message be improved?"
            "  If so, please report an enhancement request to PyTorch.)"
        ),
        tpu=(
            "native_group_norm_backward():"
            " expected mean and rstd to have the same shape,"
            " got mean size [1, 2] and rstd size [1, 3]"
        ),
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 3, device=et.device()),
          torch.ones(6, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  def test_group_norm_backward_weight_numel_mismatch(self):
    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "Expected !gamma.defined() || gamma.numel() == C to be true, but"
            " got false.  (Could this error message be improved?  If so, please"
            " report an enhancement request to PyTorch.)"
        ),
        tpu=(
            "native_group_norm_backward(): expected weight to have 6 elements,"
            " got 5"
        ),
    ):
      torch.ops.aten.native_group_norm_backward(
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 6, 3, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(1, 2, device=et.device()),
          torch.ones(5, device=et.device()),
          1,
          6,
          3,
          2,
          [True, True, True],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_lhs_shape",
          lhs_arg=(2,),
          rhs_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="ragged_dot(): lhs must be 2D, got dim: 1",
      ),
      dict(
          testcase_name="invalid_rhs_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="ragged_dot(): rhs must be 3D, got dim: 2",
      ),
      dict(
          testcase_name="invalid_group_sizes_shape",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[[1, 1]],
          expected_error="ragged_dot(): group_sizes must be 1D, got dim: 2",
      ),
      dict(
          testcase_name="mismatched_contracting_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 3, 2),
          group_sizes_arg=[1, 1],
          expected_error=(
              "ragged_dot(): contracting dimension should be the same, got: 2"
              " vs 3"
          ),
      ),
      dict(
          testcase_name="mismatched_group_dims",
          lhs_arg=(3, 2),
          rhs_arg=(2, 2, 2),
          group_sizes_arg=[1, 1, 1],
          expected_error=(
              "ragged_dot(): lhs and group_sizes should have the same number of"
              " groups, got: 2 vs 3"
          ),
      ),
  )
  def test_ragged_dot(self, lhs_arg, rhs_arg, group_sizes_arg, expected_error):
    """Tests that torch_tpu.ragged_dot fails with expected errors.

    This test checks various error conditions for the `torch_tpu.ragged_dot`
    operation, such as invalid input shapes, mismatched dimensions, and
    incorrect group sizes.

    Args:
      lhs_arg: The shape argument for the left-hand side tensor.
      rhs_arg: The shape argument for the right-hand side tensor.
      group_sizes_arg: The argument for the group sizes tensor.
      expected_error: The expected error message substring.
    """
    lhs = torch.ones(*lhs_arg, dtype=torch.float32, device=et.device())
    rhs = torch.ones(*rhs_arg, dtype=torch.float32, device=et.device())
    group_sizes = torch.tensor(
        group_sizes_arg, dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_error,
        cpu=expected_error,
    ):
      torch.ops.torch_tpu.ragged_dot(lhs, rhs, group_sizes)

  def test_max_pool2d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="max_pool2d_with_indices(): bool dtype is not supported",
        cpu="\"max_pool2d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.max_pool2d(t_bool, kernel_size=3)

  def test_masked_scatter_invalid_mask_dtype(self):
    device = et.device()
    t = torch.randn(4, 4, device=device, dtype=torch.float32)
    source = torch.randn(16, device=device, dtype=torch.float32)
    mask_int = torch.ones(4, 4, device=device, dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "masked_scatter_ only supports boolean masks,"
            " but got mask with dtype Int"
        ),
        tpu="masked_scatter_(): expected Boolean tensor for mask, got int32",
    ):
      torch.masked_scatter(t, mask_int, source)

  def test_masked_scatter_dtype_mismatch(self):
    device = et.device()
    t = torch.randn(4, 4, device=device, dtype=torch.float32)
    mask = torch.ones(4, 4, device=device, dtype=torch.bool)
    source = torch.randint(0, 10, (16,), device=device, dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        cpu=(
            "masked_scatter: expected self and source to have same dtypes but"
            " gotFloat and Int"
        ),
        tpu=(
            "masked_scatter_(): expected same dtype for self and source,"
            " got self dtype float32 and source dtype int32"
        ),
    ):
      torch.masked_scatter(t, mask, source)

  def test_arange_zero_step(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="arange(): step must be non-zero",
        cpu="step must be nonzero",
    ):
      torch.arange(1, 10, 0, device=et.device())

  def test_arange_infinite_inputs(self):
    """Tests that arange fails on infinite inputs with expected error.

    The test for when `step` is infinite is skipped because PyTorch CPU does not
    error in this case. Such a test is in
    `TpuErrorsTest.test_arange_infinite_step`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected [start, end) interval to have finite bounds,"
            " got [inf, 0)"
        ),
        cpu="unsupported range: inf -> 0",
    ):
      torch.arange(float("inf"), 0, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected [start, end) interval to have finite bounds,"
            " got [0, inf)"
        ),
        cpu="unsupported range: 0 -> inf",
    ):
      torch.arange(0, float("inf"), 1, device=et.device())

  def test_arange_invalid_inputs(self):
    """Tests that arange fails on invalid sets of inputs with expected error.

    Tests whether the given `step` is valid for the given interval [start, end].
    i.e. `step` should be negative iff `start` > `end`, but positive iff
    `start` < `end`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected step to be positive since start (0) < end (10),"
            " got step=-1"
        ),
        cpu="upper bound and lower bound inconsistent with step sign",
    ):
      torch.arange(0, 10, -1, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected step to be negative since start (10) > end (0),"
            " got step=1"
        ),
        cpu="upper bound and lower bound inconsistent with step sign",
    ):
      torch.arange(10, 0, 1, device=et.device())

  def test_arange_invalid_inputs_with_infinite_step(self):
    """Tests that arange fails on invalid sets of inputs with infinite step.

    Tests whether the given `step` (infinite) sign is valid for the given
    interval [start, end]. i.e. `step` should be negative infinity iff `start` >
    `end`, but positive infinity iff `start` < `end`.
    """

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected step to be positive since start (0) < end (10),"
            " got step=-inf"
        ),
        cpu="upper bound and lower bound inconsistent with step sign",
    ):
      torch.arange(0, 10, float("-inf"), device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "arange(): expected step to be negative since start (10) > end (0),"
            " got step=inf"
        ),
        cpu="upper bound and lower bound inconsistent with step sign",
    ):
      torch.arange(10, 0, float("inf"), device=et.device())

  def test_max_pool3d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.bool)
    with et.assert_raises_message(
        RuntimeError,
        tpu="max_pool3d_with_indices(): bool dtype is not supported",
        cpu="\"max_pool3d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.max_pool3d(t_bool, kernel_size=3)

  def test_cdist_forward_unsupported_dtypes(self):
    x1_bf16 = torch.randn(2, 2, device=et.device(), dtype=torch.bfloat16)
    x1_f16 = torch.randn(2, 2, device=et.device(), dtype=torch.float16)
    x1_int32 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
    x2_bf16 = torch.randn(2, 2, device=et.device(), dtype=torch.bfloat16)
    x2_f16 = torch.randn(2, 2, device=et.device(), dtype=torch.float16)
    x2_int32 = torch.ones(2, 2, device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "cdist_forward(): expected floating-point dtypes,"
            " got x1 dtype int32"
        ),
        cpu="cdist only supports floating-point dtypes, X1 got: Int",
    ):
      torch.cdist(x1_int32, x2_int32, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "cdist_forward(): bfloat16 and float16 dtypes are not supported,"
            " got x1 dtype bfloat16 and x2 dtype bfloat16"
        ),
        cpu="\"cdist\" not implemented for 'BFloat16'",
    ):
      torch.cdist(x1_bf16, x2_bf16, p=1.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "cdist_forward(): bfloat16 and float16 dtypes are not supported,"
            " got x1 dtype float16 and x2 dtype float16"
        ),
        cpu="\"cdist\" not implemented for 'Half'",
    ):
      torch.cdist(x1_f16, x2_f16, p=1.0)

  def test_cdist_forward_unsupported_p(self):
    x1 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)
    x2 = torch.randn(2, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="cdist_forward(): expected p value to be >= 0, got -1",
        cpu="cdist only supports non-negative p values",
    ):
      torch.cdist(x1, x2, p=-1.0)

  def test_exponential_unsupported_dtypes(self):
    device = et.device()
    t_int = torch.ones((2, 2), device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "exponential_(): expected input tensor dtype to be a floating-point"
            " real type, got int32"
        ),
        cpu=(
            "Exponential distribution is a continuous probability distribution."
            " dtype must be a floating point but you specified Int"
        ),
    ):
      t_int.exponential_()

    t_complex = torch.ones((2, 2), device=device, dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "exponential_(): expected input tensor dtype to be a floating-point"
            " real type, got complex64"
        ),
        cpu=(
            "Exponential distribution is a continuous probability distribution."
            " dtype must be a floating point but you specified ComplexFloat"
        ),
    ):
      t_complex.exponential_()

  def test_add_smaller_out_alias(self):
    """Tests that add fails when the out tensor is a smaller alias of an input."""
    a = torch.ones(4, device=et.device())
    b = torch.ones(4, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "add(): output with shape [4] doesn't match the broadcast shape of"
            " the tensor being operated on in-place, which has shape [1]"
        ),
        cpu=(
            "unsupported operation: some elements of the input tensor and the"
            " written-to tensor refer to a single memory location. Please"
            " clone() the tensor before performing the operation."
        ),
    ):
      torch.add(a, b, out=a[1:2])

  def test_avg_pool2d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool2d(): not yet implemented for uint8, int8, int16, int32,"
            " and complex64 dtypes, got complex64"
        ),
        cpu="\"avg_pool2d\" not implemented for 'ComplexFloat'",
    ):
      torch.nn.functional.avg_pool2d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool2d(): not yet implemented for uint8, int8, int16, int32,"
            " and complex64 dtypes, got uint8"
        ),
        cpu="\"avg_pool2d\" not implemented for 'Byte'",
    ):
      torch.nn.functional.avg_pool2d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool2d(): not yet implemented for uint8, int8, int16, int32,"
            " and complex64 dtypes, got int8"
        ),
        cpu="\"avg_pool2d\" not implemented for 'Char'",
    ):
      torch.nn.functional.avg_pool2d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool2d(): not yet implemented for uint8, int8, int16, int32,"
            " and complex64 dtypes, got int16"
        ),
        cpu="\"avg_pool2d\" not implemented for 'Short'",
    ):
      torch.nn.functional.avg_pool2d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool2d(): not yet implemented for uint8, int8, int16, int32,"
            " and complex64 dtypes, got int32"
        ),
        cpu="\"avg_pool2d\" not implemented for 'Int'",
    ):
      torch.nn.functional.avg_pool2d(t_int32, kernel_size=3)

  def test_avg_pool3d_unsupported_dtypes(self):
    t_bool = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.bool)
    t_bf16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.bfloat16
    )
    t_f16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.float16
    )
    t_complex = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.uint8
    )
    t_int8 = torch.zeros((1, 1, 4, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.int16
    )
    t_int32 = torch.zeros(
        (1, 1, 4, 4, 4), device=et.device(), dtype=torch.int32
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got bool"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Bool'",
    ):
      torch.nn.functional.avg_pool3d(t_bool, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got bfloat16"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'BFloat16'",
    ):
      torch.nn.functional.avg_pool3d(t_bf16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got float16"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Half'",
    ):
      torch.nn.functional.avg_pool3d(t_f16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got complex64"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'ComplexFloat'",
    ):
      torch.nn.functional.avg_pool3d(t_complex, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got uint8"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Byte'",
    ):
      torch.nn.functional.avg_pool3d(t_uint8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got int8"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Char'",
    ):
      torch.nn.functional.avg_pool3d(t_int8, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got int16"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Short'",
    ):
      torch.nn.functional.avg_pool3d(t_int16, kernel_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "avg_pool3d(): not yet implemented for bool, bfloat16, float16,"
            " uint8, int8, int16, int32, and complex64 dtypes, got int32"
        ),
        cpu="\"avg_pool3d_out_frame\" not implemented for 'Int'",
    ):
      torch.nn.functional.avg_pool3d(t_int32, kernel_size=3)

  def test_pdist_forward_unsupported_dtypes(self):
    t_bf16 = torch.randn(2, 2, device=et.device(), dtype=torch.bfloat16)
    t_f16 = torch.randn(2, 2, device=et.device(), dtype=torch.float16)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "pdist_forward(): bfloat16 and float16 dtypes are not supported,"
            " got self dtype bfloat16"
        ),
        cpu="\"pdist\" not implemented for 'BFloat16'",
    ):
      torch.nn.functional.pdist(t_bf16, p=2.0)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "pdist_forward(): bfloat16 and float16 dtypes are not supported,"
            " got self dtype float16"
        ),
        cpu="\"pdist\" not implemented for 'Half'",
    ):
      torch.nn.functional.pdist(t_f16, p=2.0)

  # TODO(lwh): fix this test once G3 pytorch version is updated
  @unittest.skip("Disabled due to pytorch version mismatch")
  def test_replication_pad_backward(self):
    with et.assert_raises_message(
        RuntimeError,
        "Mismatch in shape: grad_output[0] has a shape of torch.Size([1])"
        " and output[0] has a shape of torch.Size([1, 6, 4]).",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 2,"
        " 3, 4]) and output[0] has a shape of torch.Size([1, 6, 4]).",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 2, 3, 4, device=et.device()))
    # Incorrectly sized padding input passes forward pass and fails with a
    # pytorch assertion error on the backward pass.
    with et.assert_raises_message(
        AssertionError,
        "",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    # Empty padding input
    with et.assert_raises_message(
        RuntimeError,
        "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported"
        " for now",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 6, 4])"
        " and output[0] has a shape of torch.Size([1, 6, 13]).",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[4, 5],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "input (W: 4) is too small. Calculated output W: -2",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[-1, -5],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

  def test_replication_pad_backward_unsupported_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="replication_pad1d(): not implemented for 'Bool'",
        cpu="\"replication_pad1d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="replication_pad2d(): not implemented for 'Bool'",
        cpu="\"replication_pad2d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="replication_pad3d(): not implemented for 'Bool'",
        cpu="\"replication_pad3d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0, 0, 0],
          mode="replicate",
      ).backward(torch.randn(1, 6, 4, 4, 4, device=et.device()))

  def test_reflection_pad_backward(self):
    with et.assert_raises_message(
        RuntimeError,
        "Mismatch in shape: grad_output[0] has a shape of torch.Size([1])"
        " and output[0] has a shape of torch.Size([1, 6, 4]).",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "Mismatch in shape: grad_output[0] has a shape of torch.Size([1, 2,"
        " 3, 4]) and output[0] has a shape of torch.Size([1, 6, 4]).",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 2, 3, 4, device=et.device()))
    # Incorrectly sized padding input passes forward pass and fails with a
    # pytorch assertion error on the backward pass.
    with et.assert_raises_message(
        RuntimeError,
        "element 0 of tensors does not require grad and does not have a grad_fn"
        if _is_internal()
        else "",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    # Empty padding input
    with et.assert_raises_message(
        RuntimeError,
        "Padding size 0 is not supported for 3D input"
        " tensor.\nSupported combinations for non-constant padding:\n "
        " - 2D or 3D input: padding size = 2 (pads last dimension)\n  -"
        " 3D or 4D input: padding size = 4 (pads last 2 dimensions)\n "
        " - 4D or 5D input: padding size = 6 (pads last 3 dimensions)",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "Argument #4: Padding size should be less than the corresponding"
        " input dimension, but got: padding (4, 5) at dimension 2 of input"
        " [1, 6, 4]",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[4, 5],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))
    with et.assert_raises_message(
        RuntimeError,
        "input (W: 4) is too small. Calculated output W: -2",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device()),
          pad=[-1, -5],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

  def test_reflection_pad_backward_unsupported_dtypes(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="reflection_pad1d(): not implemented for bool",
        cpu="\"reflection_pad1d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="reflection_pad2d(): not implemented for bool",
        cpu="\"reflection_pad2d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, 4, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu="reflection_pad3d(): not implemented for bool",
        cpu="\"reflection_pad3d\" not implemented for 'Bool'",
    ):
      torch.nn.functional.pad(
          input=torch.ones(1, 6, 4, 4, 4, device=et.device(), dtype=torch.bool),
          pad=[0, 0, 0, 0, 0, 0],
          mode="reflect",
      ).backward(torch.randn(1, 6, 4, 4, 4, device=et.device()))

  def test_adaptive_avg_pool2d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)
    t_int64 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got complex64"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'ComplexFloat'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_complex, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got uint8"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'Byte'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_uint8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int8"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'Char'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int8, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int16"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'Short'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int16, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int32"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'Int'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int32, output_size=2)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool2d(): not yet implemented for uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int64"
        ),
        cpu="\"adaptive_avg_pool2d\" not implemented for 'Long'",
    ):
      torch.nn.functional.adaptive_avg_pool2d(t_int64, output_size=2)

  def test_adaptive_avg_pool3d_unsupported_dtypes(self):
    t_complex = torch.zeros(
        (1, 1, 4, 4), device=et.device(), dtype=torch.complex64
    )
    t_uint8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.uint8)
    t_int8 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int8)
    t_int16 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int16)
    t_int32 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int32)
    t_int64 = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.int64)
    t_bool = torch.zeros((1, 1, 4, 4), device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got bool"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Bool'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_bool, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got complex64"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'ComplexFloat'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_complex, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got uint8"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Byte'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_uint8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int8"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Char'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int8, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int16"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Short'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int16, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int32"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Int'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int32, output_size=3)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "adaptive_avg_pool3d(): not yet implemented for bool, uint8, int8,"
            " int16, int32, int64, and complex64 dtypes, got int64"
        ),
        cpu="\"adaptive_avg_pool3d_cpu\" not implemented for 'Long'",
    ):
      torch.nn.functional.adaptive_avg_pool3d(t_int64, output_size=3)

  def test_floor_divide_complex64(self):
    lhs = torch.arange(5, device=et.device())
    rhs = torch.arange(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "floor_divide(): expected dtype of the first argument to be"
            " neither complex nor bool, got complex64"
        ),
        cpu="\"div_floor_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "floor_divide(): expected dtype of the second argument to be"
            " neither complex nor bool, got complex64"
        ),
        cpu="\"div_floor_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      torch.floor_divide(lhs, rhs.to(torch.complex64))

  def test_atan2_complex(self):
    x = torch.tensor([1.0, 2.0], device=et.device())
    y = torch.tensor([1.0, 2.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "atan2(): expected the dtype of the first argument not to be"
            " complex, got complex64"
        ),
        cpu="\"atan2_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      torch.atan2(x.to(torch.complex64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "atan2(): expected the dtype of the second argument not to be"
            " complex, got complex64"
        ),
        cpu="\"atan2_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      torch.atan2(x, y.to(torch.complex64))

  @parameterized.named_parameters(
      {
          "testcase_name": "bitwise_and",
          "op_name": "bitwise_and",
          "op": torch.bitwise_and,
      },
      {
          "testcase_name": "bitwise_or",
          "op_name": "bitwise_or",
          "op": torch.bitwise_or,
      },
      {
          "testcase_name": "bitwise_xor",
          "op_name": "bitwise_xor",
          "op": torch.bitwise_xor,
      },
  )
  def test_bitwise_ops_float64(self, op_name: str, op: Any):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    y = torch.ones(5, dtype=torch.int64, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dtype of the first argument to be"
            " neither floating-point nor complex, got float64"
        ),
        cpu=f"\"{op_name}_cpu\" not implemented for 'Double'",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dtype of the second argument to be"
            " neither floating-point nor complex, got float64"
        ),
        cpu=f"\"{op_name}_cpu\" not implemented for 'Double'",
        message_reviewed_by="wan",
    ):
      op(x, y.to(torch.float64))

  @parameterized.named_parameters(
      {
          "testcase_name": "bitwise_left_shift",
          "op_name_tpu": "bitwise_left_shift",
          "op_name_cpu": "lshift",
          "op": torch.bitwise_left_shift,
      },
      {
          "testcase_name": "bitwise_right_shift",
          "op_name_tpu": "bitwise_right_shift",
          "op_name_cpu": "rshift",
          "op": torch.bitwise_right_shift,
      },
  )
  def test_bitwise_shift_float64(
      self, op_name_tpu: str, op_name_cpu: str, op: Any
  ):
    x = torch.ones(5, dtype=torch.int64, device=et.device())
    y = torch.ones(5, dtype=torch.int64, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name_tpu}(): expected the dtype of the first argument to"
            " be integer, got float64"
        ),
        cpu=f"\"{op_name_cpu}_cpu\" not implemented for 'Double'",
        message_reviewed_by="wan",
    ):
      op(x.to(torch.float64), y)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name_tpu}(): expected the dtype of the second argument to"
            " be integer, got float64"
        ),
        cpu=f"\"{op_name_cpu}_cpu\" not implemented for 'Double'",
        message_reviewed_by="wan",
    ):
      op(x, y.to(torch.float64))

  def test_col2im_output_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected output_size to have 2 dimensions, got 3",
        cpu="It is expected output_size equals to 2, but got size 3",
    ):
      torch.ops.aten.col2im(img, (5, 5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected kernel_size to have 2 dimensions, got 3",
        cpu="It is expected kernel_size equals to 2, but got size 3",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_dilation_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected dilation to have 2 dimensions, got 3",
        cpu="It is expected dilation equals to 2, but got size 3",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1, 1), (0, 0), (1, 1))

  def test_col2im_padding_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected padding to have 2 dimensions, got 3",
        cpu="It is expected padding equals to 2, but got size 3",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0, 0), (1, 1))

  def test_col2im_stride_must_be_2d(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected stride to have 2 dimensions, got 3",
        cpu="It is expected stride equals to 2, but got size 3",
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1, 1))

  def test_col2im_input_must_be_3d(self):
    img = torch.randn(1, 4, 16, 1, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "col2im(): expected input to have 3 dimensions (batch, channels,"
            " length), got 4"
        ),
        cpu=(
            "Expected 2D or 3D (batch mode) tensor for input with possibly 0"
            " batch size and non-zero dimensions for input, but got: [1, 4,"
            " 16, 1]"
        ),
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_kernel_size_must_be_positive(self):
    img = torch.randn(1, 4, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="col2im(): expected kernel size to be positive, got 0",
        cpu=(
            "kernel size should be greater than zero, but got kernel_height: 0"
            " kernel_width: 2"
        ),
    ):
      torch.ops.aten.col2im(img, (5, 5), (0, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_channels_divisibility(self):
    img = torch.randn(1, 5, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "col2im(): expected input channels to be divisible by kernel"
            " product (4), got 5"
        ),
        cpu=(
            "Expected size of input's dimension 1 to be divisible by the"
            " product of kernel_size, but got input.size(1)=5 and"
            " kernel_size=(2, 2)."
        ),
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  def test_col2im_length_mismatch(self):
    # output=(5,5), k=(2,2), stride=(1,1), pad=(0,0)
    # -> col_h, col_w = (4, 4) -> L=16
    img = torch.randn(1, 4, 15, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "col2im(): expected input length to be divisible by col size (4 * 4"
            " = 16), got 15"
        ),
        cpu=(
            "Given output_size=(5, 5), kernel_size=(2, 2), dilation=(1, 1),"
            " padding=(0, 0), stride=(1, 1), expected size of input's dimension"
            " 2 to match the calculated number of sliding blocks 4 * 4 = 16,"
            " but got input.size(2)=15."
        ),
    ):
      torch.ops.aten.col2im(img, (5, 5), (2, 2), (1, 1), (0, 0), (1, 1))

  @parameterized.named_parameters(
      {"testcase_name": "ge", "op_name": "ge", "op": torch.ge},
      {"testcase_name": "gt", "op_name": "gt", "op": torch.gt},
      {"testcase_name": "le", "op_name": "le", "op": torch.le},
      {"testcase_name": "lt", "op_name": "lt", "op": torch.lt},
  )
  def test_comparison_ops_complex(self, op_name: str, op: Any):
    lhs = torch.tensor([1.0, 2.0], device=et.device())
    rhs = torch.tensor([1.0, 2.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dtype of the first argument not to be"
            " complex, got complex64"
        ),
        cpu=f"\"{op_name}_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      op(lhs.to(torch.complex64), rhs)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dtype of the second argument not to"
            " be complex, got complex64"
        ),
        cpu=f"\"{op_name}_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      op(lhs, rhs.to(torch.complex64))

    # TODO: b/478955517 dtype checks should run after dtype promotion.
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dtype of the second argument not to"
            " be complex, got complex128"
        ),
        cpu=f"\"{op_name}_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      op(lhs, 1j)

  def test_remainder_complex(self):
    c64 = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    f32 = torch.tensor([1.0], device=et.device())

    test_cases = [
        (1, c64),
        (1j, f32),
        (c64, f32),
    ]

    for args in test_cases:
      with et.assert_raises_message(
          RuntimeError,
          tpu=(
              "remainder(): expected the dtype of the output (promoted inputs"
              " dtype) to be neither bool nor complex, got complex64"
          ),
          cpu="\"remainder_cpu\" not implemented for 'ComplexFloat'",
          message_reviewed_by="wan",
      ):
        torch.remainder(*args)

  def test_foreach_add_int_tensors_float_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "foreach_add(): expected alpha to be integral for integral input"
            " tensors, got float64"
        ),
        cpu=(
            "For integral input tensors, argument alpha must not be a floating"
            " point number."
        ),
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=1.5)

  def test_foreach_add_int_tensors_bool_alpha(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    other_list = [torch.tensor([3, 4], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "foreach_add(): expected input tensor dtypes to be bool when alpha"
            " dtype is bool, got int32 and int32"
        ),
        cpu="Boolean alpha only supported for Boolean results.",
        message_reviewed_by="wan",
    ):
      torch._foreach_add(self_list, other_list, alpha=True)

  def test_inplace_foreach_add_int_and_float(self):
    self_list = [torch.tensor([1, 2], dtype=torch.int32, device=et.device())]
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "foreach_add_(): expected the scalar dtype to be castable to the"
            " tensor dtype (e.g. bool to int or int to float), got float64 and"
            " int32"
        ),
        cpu="result type Float can't be cast to the desired output type Int",
        message_reviewed_by="wan",
    ):
      torch._foreach_add_(self_list, 1.5)

  def test_inplace_foreach_add_bool_tensors_and_int_scalars(self):
    self_list = [
        torch.tensor([True, True], dtype=torch.bool, device=et.device()),
        torch.tensor([True, True], dtype=torch.bool, device=et.device()),
    ]
    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "foreach_add_(): expected the scalar dtype to be castable to the"
            " tensor dtype (e.g. bool to int or int to float), got int64 and"
            " bool"
        ),
        cpu="result type Long can't be cast to the desired output type Bool",
        message_reviewed_by="wan",
    ):
      torch._foreach_add_(self_list, [1, 1])

  def test_sub_bool(self):
    lhs = torch.tensor([1.0, 1.0], device=et.device())
    rhs = torch.tensor([1.0, 1.0], device=et.device())
    out = torch.tensor([0.0, 0.0], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="sub(): the dtype of the first argument cannot be bool",
        cpu=(
            "Subtraction, the `-` operator, with a bool tensor is not"
            " supported. If you are trying to invert a mask, use the `~` or"
            " `logical_not()` operator instead."
        ),
        message_reviewed_by="wan",
    ):
      torch.sub(lhs.to(torch.bool), rhs, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu="sub(): the dtype of the second argument cannot be bool",
        cpu=(
            "Subtraction, the `-` operator, with a bool tensor is not"
            " supported. If you are trying to invert a mask, use the `~` or"
            " `logical_not()` operator instead."
        ),
        message_reviewed_by="wan",
    ):
      torch.sub(lhs, rhs.to(torch.bool), out=out)

  def _test_aminmax_output_dtype_mismatch_impl(
      self, op_name: str, op: Any, cpu: str
  ):
    tensor = torch.ones(5, device=et.device(), dtype=torch.int64)
    out = _get_aminmax_outputs(op, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected output tensor dtype to match the dtype of"
            " the first argument (int64), got complex64"
        ),
        cpu=cpu,
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  @parameterized.named_parameters(
      {"testcase_name": "amin", "op_name": "amin", "op": torch.amin},
      {"testcase_name": "amax", "op_name": "amax", "op": torch.amax},
  )
  def test_amin_amax_output_dtype_mismatch(self, op_name: str, op: Any):
    self._test_aminmax_output_dtype_mismatch_impl(
        op_name,
        op,
        cpu=(
            "Expected the dtype for input and out to match, but got Long for"
            " input's dtype and ComplexFloat for out's dtype."
        ),
    )

  def test_aminmax_output_dtype_mismatch(self):
    self._test_aminmax_output_dtype_mismatch_impl(
        op_name="aminmax",
        op=torch.aminmax,
        cpu=(
            "Expected out tensor to have dtype long, but got"
            " c10::complex<float> instead"
        ),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "amin",
          "op_name_cpu": "min_values",
          "op_name_tpu": "amin",
          "op": torch.amin,
      },
      {
          "testcase_name": "amax",
          "op_name_cpu": "max_values",
          "op_name_tpu": "amax",
          "op": torch.amax,
      },
      {
          "testcase_name": "aminmax",
          "op_name_cpu": "aminmax",
          "op_name_tpu": "aminmax",
          "op": torch.aminmax,
      },
  )
  def test_aminmax_complex(self, op_name_cpu: str, op_name_tpu: str, op: Any):
    tensor = torch.ones(5, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name_tpu}(): expected the dtype of the input not to be"
            " complex, got complex64"
        ),
        cpu=f"\"{op_name_cpu}_cpu\" not implemented for 'ComplexFloat'",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0)

  def test_complex_int(self):
    real = torch.ones(5, device=et.device(), dtype=torch.float32)
    img = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "complex(): expected the dtype of the first argument to be float32"
            " or float64, got int32"
        ),
        cpu=(
            "Expected both inputs to be Half, Float or Double tensors but got"
            " Int and Float"
        ),
        message_reviewed_by="wan",
    ):
      torch.complex(real.to(torch.int32), img, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "complex(): expected the dtype of the second argument to be float32"
            " or float64, got int32"
        ),
        cpu=(
            "Expected both inputs to be Half, Float or Double tensors but got"
            " Float and Int"
        ),
        message_reviewed_by="wan",
    ):
      torch.complex(real, img.to(torch.int32), out=out)

  def test_polar_int(self):
    absv = torch.tensor([1.0, 2.0], device=et.device())
    angle = torch.tensor([1.0, 2.0], device=et.device())
    out = torch.empty(2, device=et.device(), dtype=torch.complex64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "polar(): expected the dtype of the first argument to be float32 or"
            " float64, got int32"
        ),
        cpu=(
            "Expected both inputs to be Half, Float or Double tensors but got"
            " Int and Float"
        ),
        message_reviewed_by="wan",
    ):
      torch.polar(absv.to(torch.int32), angle, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "polar(): expected the dtype of the second argument to be float32"
            " or float64, got int32"
        ),
        cpu=(
            "Expected both inputs to be Half, Float or Double tensors but got"
            " Float and Int"
        ),
        message_reviewed_by="wan",
    ):
      torch.polar(absv, angle.to(torch.int32), out=out)

  def test_addmv_bool(self):
    t = torch.ones(5, device=et.device(), dtype=torch.bool)
    mat = torch.ones(5, 5, device=et.device(), dtype=torch.bool)
    vec = torch.ones(5, device=et.device(), dtype=torch.bool)

    with et.assert_raises_message(
        RuntimeError,
        tpu="addmv(): the dtype of the first argument cannot be bool",
        cpu="\"addmv_impl_cpu\" not implemented for 'Bool'",
        message_reviewed_by="wan",
    ):
      torch.addmv(t.to(torch.bool), mat, vec)

  def test_addmv_not_a_matrix(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, 5, device=et.device())
    vec = torch.ones(5, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "addmv(): expected the second argument to be a matrix (2D tensor),"
            " got 3D tensor"
        ),
        cpu="vector + matrix @ vector expected, got 1, 3, 1",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_not_a_vector(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, device=et.device())
    vec = torch.ones(5, 5, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "addmv(): expected the third argument to be a vector (1D tensor),"
            " got 2D tensor"
        ),
        cpu="vector + matrix @ vector expected, got 1, 2, 2",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_invalid_mm_dimensions(self):
    inp = torch.ones(5, device=et.device())
    mat = torch.ones(5, 5, device=et.device())
    vec = torch.ones(4, device=et.device())
    out = torch.empty(5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "addmv(): expected the last dimension of the second argument"
            " (matrix of size [5, 5]) to match the first dimension of the third"
            " argument (vector of size [4]), got 5 vs 4"
        ),
        cpu="size mismatch, got input (5), mat (5x5), vec (4)",
        message_reviewed_by="wan",
    ):
      torch.addmv(inp, mat, vec, out=out)

  def test_addmv_scalars_complex(self):
    args = (
        torch.ones(5, device=et.device()),
        torch.ones(5, 5, device=et.device()),
        torch.ones(5, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "addmv(): expected the dtype of alpha to be neither complex nor"
            " bool, got complex128"
        ),
        cpu="value cannot be converted to type float without overflow",
        message_reviewed_by="wan",
    ):
      torch.addmv(*args, alpha=1j)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "addmv(): expected the dtype of beta to be neither complex nor"
            " bool, got complex128"
        ),
        cpu="value cannot be converted to type float without overflow",
        message_reviewed_by="wan",
    ):
      torch.addmv(*args, beta=1j)

  def test_clamp_cant_cast(self):
    inp = torch.tensor([-1.0, 2.0], device=et.device(), dtype=torch.complex64)
    out = torch.empty(2, device=et.device(), dtype=torch.float64)

    # Scalar and Tensor inputs go through different execution path, so test that
    # both of them raise the same error.
    min_max_values = [
        (1.0, 2.0),
        (
            torch.zeros(2, device=et.device(), dtype=torch.float64),
            torch.ones(2, device=et.device(), dtype=torch.float64),
        ),
    ]

    for minv, maxv in min_max_values:
      with et.assert_raises_message(
          RuntimeError,
          tpu=(
              "clamp(): unable to cast complex128, the promotion of the dtypes"
              " of the inputs (complex64, min: float64, max: float64), to the"
              " output dtype float64"
          ),
          cpu="clamp is not supported for complex types",
          message_reviewed_by="wan",
      ):
        torch.clamp(inp, min=minv, max=maxv, out=out)

  def test_bmm_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="bmm(): the dtype of the first argument cannot be bool",
        cpu="\"bmm\" not implemented for 'Bool'",
        message_reviewed_by="wan",
    ):
      torch.bmm(a.to(torch.bool), b)

    with et.assert_raises_message(
        RuntimeError,
        tpu="bmm(): the dtype of the second argument cannot be bool",
        cpu="Expected out tensor to have dtype bool, but got float instead",
        message_reviewed_by="wan",
    ):
      # Call `bmm.out` op.
      #
      # Otherwise, it will trigger the output tensor dtype check first. This
      # happens because of 2 reasons:
      #   1. output dtype is checked first
      #   2. PyTorch generated code sets the output dtype to be whatever `b`
      #      dtype is (bool)
      torch.bmm(a, b.to(torch.bool), out=out)

  def test_bmm_output_bool(self):
    a = torch.ones(1, 2, 3, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 3, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.bool, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="bmm(): the dtype of the output tensor cannot be bool",
        cpu="Expected out tensor to have dtype float, but got bool instead",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_bmm_not_batch_of_matrices(self):
    a = torch.ones(1, 2, 3, 4, dtype=torch.float32, device=et.device())
    b = torch.ones(1, 4, 2, dtype=torch.float32, device=et.device())
    out = torch.ones(1, 2, 2, dtype=torch.float32, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "bmm(): expected the first argument to be a 3D tensor (batch of"
            " matrices), got 4D"
        ),
        cpu="batch1 must be a 3D tensor",
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "bmm(): expected the second argument to be a 3D tensor (batch of"
            " matrices), got 4D"
        ),
        cpu="batch2 must be a 3D tensor",
        message_reviewed_by="wan",
    ):
      torch.bmm(b, a, out=out)

  def test_bmm_mismatch_batch_dimensions(self):
    a = torch.ones(1, 2, 3, device=et.device())
    b = torch.ones(2, 3, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "bmm(): expected the batch dimension of the first argument [1, 2,"
            " 3] to match the batch dimension of the second argument [2, 3, 2],"
            " got 1 vs 2"
        ),
        cpu=(
            "Expected size for first two dimensions of batch2 tensor to be: [1,"
            " 3] but got: [2, 3]."
        ),
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  def test_bmm_mismatch_mm_contracting_dimension(self):
    a = torch.ones(1, 2, 3, device=et.device())
    b = torch.ones(1, 2, 2, device=et.device())
    out = torch.ones(1, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "bmm(): expected the last dimension of the first argument [1, 2, 3]"
            " to match the second dimension of the second argument [1, 2, 2],"
            " got 3 vs 2"
        ),
        cpu=(
            "Expected size for first two dimensions of batch2 tensor to be: [1,"
            " 3] but got: [1, 2]."
        ),
        message_reviewed_by="wan",
    ):
      torch.bmm(a, b, out=out)

  @_parameterize_convolution_fwd_bwd(
      forward={
          "tpu_fn": "convolution",
          "cpu_fn": "slow_conv2d_cpu",
      },
      backward={
          "tpu_fn": "convolution_backward",
          "cpu_fn": "slow_conv2d_cpu_grad_input",
      },
  )
  def test_convolution_bool(self, convolution, tpu_fn: str, cpu_fn: str):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the dtype of the input"
            " tensor to be neither long nor bool, got bool"
        ),
        cpu=f"\"{cpu_fn}\" not implemented for 'Bool'",
        message_reviewed_by="wan",
    ):
      convolution(inp.to(torch.bool), w)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the dtype of the weight"
            " tensor to be neither long nor bool, got bool"
        ),
        cpu="expected scalar type Float but found Bool",
        message_reviewed_by="wan",
    ):
      convolution(inp, w.to(torch.bool))

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_input_invalid_rank(self, convolution, tpu_fn: str):
    inp = torch.ones(10, 10, device=et.device())

    # PyTorch CPU implementation will check the weight tensor before the input
    # tensor. So, it must have >3 dimensions.
    w = torch.ones(2, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the input to have >= 3 dimensions of shape"
            " [batch, in channels, ... spatial dimensions ...], got shape"
            " [10, 10]"
        ),
        cpu=(
            "Expected 3-dimensional input for 3-dimensional weight [2, 3, 3],"
            " but got 2-dimensional input of size [10, 10] instead"
        ),
        message_reviewed_by="wan",
    ):
      convolution(inp, w)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_spatial_dimensions_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    def test_arg(arg_name: str):
      with et.assert_raises_message(
          RuntimeError,
          tpu=(
              f"{tpu_fn}(): expected {arg_name} to be either an integer or a"
              " 2-element list that matches the convolution dimensions, got"
              " [1, 1, 1]"
          ),
          cpu=(
              f"expected {arg_name} to be a single integer value or a list of 2"
              " values to match the convolution dimensions, but got"
              f" {arg_name}=[1, 1, 1]"
          ),
          message_reviewed_by="wan",
      ):
        convolution(inp, w, **{arg_name: (1, 1, 1)})

    # Each of these parameters go through the same error check.
    # Since this test is already parameterized for forward and backward, we use
    # subTest() for checking each parameter.
    for arg_name in ("stride", "padding", "dilation", "output_padding"):
      with self.subTest(arg_name=arg_name):
        test_arg(arg_name)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_weight_invalid_rank(self, convolution, tpu_fn: str):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the weight tensor to have 4 dimensions of"
            " shape [out channels, in channels per group, ... 2 spatial"
            " dimensions ...], got shape [1, 3, 3, 3, 3]"
        ),
        cpu=(
            "Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3,"
            " 3], but got 4-dimensional input of size [2, 3, 10, 10] instead"
        ),
        message_reviewed_by="wan",
    ):
      convolution(inp, w)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_convolution_weight_dimension_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the second dimension of the weight tensor of"
            " shape [1, 3, 3, 3] to be 1 (3 in channels divided by 3 groups),"
            " got 3"
        ),
        cpu=(
            "Given groups=3, expected weight to be at least 3 at dimension 0,"
            " but got weight of size [1, 3, 3, 3] instead"
        ),
        message_reviewed_by="wan",
    ):
      convolution(inp, w, groups=3)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_transposed_convolution_weight_invalid_rank(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the weight tensor to have 4 dimensions of"
            " shape [in channels, out channels per group, ... 2 spatial"
            " dimensions ...], got shape [1, 3, 3, 3, 3]"
        ),
        cpu=(
            "Expected 5-dimensional input for 5-dimensional weight [1, 3, 3, 3,"
            " 3], but got 4-dimensional input of size [2, 3, 10, 10] instead"
        ),
        message_reviewed_by="wan",
    ):
      convolution(inp, w, transposed=True)

  @_parameterize_convolution_fwd_bwd(
      forward={"tpu_fn": "convolution"},
      backward={"tpu_fn": "convolution_backward"},
  )
  def test_transposed_convolution_weight_dimension_mismatch(
      self, convolution, tpu_fn: str
  ):
    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{tpu_fn}(): expected the first dimension of the weight tensor of"
            " shape [1, 3, 3, 3] to be 3 (number of in channels), got 1"
        ),
        cpu=(
            "Given groups=3, expected weight to be at least 3 at dimension 0,"
            " but got weight of size [1, 3, 3, 3] instead"
        ),
        message_reviewed_by="wan",
    ):
      convolution(inp, w, groups=3, transposed=True)

  def test_convolution_bias_invalid_dimensions(self):
    # Why isn't this test also run on backward?
    # =========================================
    #
    # PyTorch native device implementation doesn't error on invalid bias_sizes
    # values. This might actually be a bug on PyTorch upstream.

    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "convolution(): expected the bias tensor to have 1 dimension of"
            " shape [1 (out channels)], got shape [1, 1]"
        ),
        cpu=(
            "Given weight of size [1, 3, 3, 3], expected bias to be"
            " 1-dimensional with 1 elements, but got bias of size [1, 1]"
            " instead"
        ),
        message_reviewed_by="wan",
    ):
      _run_convolution(inp, w, bias=torch.ones(1, 1, device=et.device()))

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "convolution(): expected the bias tensor to have 1 dimension of"
            " shape [1 (out channels)], got shape [5]"
        ),
        cpu=(
            "Given weight of size [1, 3, 3, 3], expected bias to be"
            " 1-dimensional with 1 elements, but got bias of size [5] instead"
        ),
        message_reviewed_by="wan",
    ):
      _run_convolution(inp, w, bias=torch.ones(5, device=et.device()))

  def test_convolution_backward_grad_bool(self):
    # This test compliments the parameterized test `test_convolution_bool()`,
    # which does the same for `inp` and `w`. Since `grad` only exists in the
    # backward, this specifically tests it.
    grad = torch.ones(2, 1, 8, 8, device=et.device(), dtype=torch.bool)

    inp = torch.ones(2, 3, 10, 10, device=et.device())
    w = torch.ones(1, 3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "convolution_backward(): expected the dtype of the grad tensor to"
            " be neither long nor bool, got bool"
        ),
        cpu="expected scalar type Float but found Bool",
        message_reviewed_by="wan",
    ):
      _run_convolution_backward(grad, inp, w)

  @parameterized.named_parameters(
      {"testcase_name": "min", "op": torch.min, "op_name": "min"},
      {"testcase_name": "max", "op": torch.max, "op_name": "max"},
  )
  def test_min_max_input_with_zero_elements(self, op, op_name: str):
    inp = torch.ones(2, 2, 0, 5, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            f"{op_name}(): expected the dim argument to be specified when the"
            " input tensor has 0 elements"
        ),
        cpu=(
            f"{op_name}(): Expected reduction dim to be specified for"
            " input.numel() == 0. Specify the reduction dim with the 'dim'"
            " argument."
        ),
        message_reviewed_by="wan",
    ):
      op(inp)

  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op": torch.argmin, "op_name": "argmin"},
      {"testcase_name": "argmax", "op": torch.argmax, "op_name": "argmax"},
  )
  def test_argmin_float_output(self, op, op_name: str):
    inp = torch.ones(2, 2, device=et.device())
    out = torch.empty(2, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"{op_name}(): expected the output dtype to be int64, got float32",
        cpu="Expected out tensor to have dtype long, but got float instead",
        message_reviewed_by="wan",
    ):
      op(inp, out=out)

  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op": torch.argmin, "op_name": "argmin"},
      {"testcase_name": "argmax", "op": torch.argmax, "op_name": "argmax"},
  )
  def test_argmin_invalid_dtypes(self, op, op_name: str):
    # We need to call the out-of-place variant of `argmin` (`argmax`) op, so
    # that we don't go through the fallback. Otherwise, the meta kernel will
    # catch this error before it reaches TorchTPU implementation.
    out = torch.empty(2, device=et.device(), dtype=torch.int64)

    def test_with(dtype: torch.dtype, tpu: str, cpu: str):
      """Tests the `op` with the input tensor of the given `dtype`.

      Tests that running `argmin` (`argmax`) with the given `dtype` will result
      in the expected error.

      Args:
        dtype: The dtype of the op input tensor.
        tpu: String representation for `dtype` to be used in the error message
          of the TPU kernel.
        cpu: String representation for `dtype` to be used in the error message
          of the CPU kernel.
      """

      inp = torch.ones(2, 2, device=et.device(), dtype=dtype)

      with et.assert_raises_message(
          RuntimeError,
          tpu=(
              f"{op_name}(): expected the input dtype to be neither complex nor"
              f" bool, got {tpu}"
          ),
          cpu=f"{op_name}(): does not support {cpu} input",
          message_reviewed_by="wan",
      ):
        op(inp, out=out)

    with self.subTest(dtype=torch.bool):
      test_with(torch.bool, tpu="bool", cpu="bool")
    with self.subTest(dtype=torch.complex64):
      test_with(torch.complex64, tpu="complex64", cpu="complex")

  def test_mm_output_dtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    rhs = torch.ones(4, 5, device=et.device())
    out = torch.ones(3, 5, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "mm(): expected the inputs and the output to have the same dtype,"
            " got float32 vs float64"
        ),
        cpu="Expected out tensor to have dtype float, but got double instead",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_mm_inputs_dtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float32)
    rhs = torch.ones(4, 5, device=et.device(), dtype=torch.float64)

    # Call the out-of-place variant of `mm()` op.
    out = torch.ones(3, 6, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "mm(): expected the two arguments to have the same dtype, got"
            " float32 vs float64"
        ),
        cpu=(
            "expected m1 and m2 to have the same dtype, but got: float !="
            " double"
        ),
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_mm_inputs_are_not_matrices(self):
    not_a_matrix_tensor = torch.ones(3, 4, 5, device=et.device())
    matrix_tensor = torch.ones(4, 4, device=et.device())

    # Call the out-of-place variant of `mm()` op.
    out = torch.ones(4, 4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "mm(): expected the first argument to be a 2D tensor (matrix),"
            " got 3D of shape [3, 4, 5]"
        ),
        cpu="self must be a matrix",
        message_reviewed_by="wan",
    ):
      torch.mm(not_a_matrix_tensor, matrix_tensor, out=out)

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "mm(): expected the second argument to be a 2D tensor (matrix),"
            " got 3D of shape [3, 4, 5]"
        ),
        cpu="mat2 must be a matrix",
        message_reviewed_by="wan",
    ):
      torch.mm(matrix_tensor, not_a_matrix_tensor, out=out)

  def test_mm_inputs_dimension_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device())
    rhs = torch.ones(5, 6, device=et.device())

    # Call the out-of-place variant of `mm()` op.
    out = torch.ones(3, 6, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "mm(): expected the column size of the first matrix to match the"
            " row size of the second matrix, got shape [3, 4] vs [5, 6] where 4"
            " != 5"
        ),
        cpu="mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)",
        message_reviewed_by="wan",
    ):
      torch.mm(lhs, rhs, out=out)

  def test_linalg_lu_factor_ex_no_pivoting(self):
    a = torch.ones(1, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_factor_ex(): non-pivoting decomposition is not supported"
        ),
        cpu=(
            "linalg.lu_factor: LU without pivoting is not implemented on"
            " the CPU"
        ),
        message_reviewed_by="wan",
    ):
      torch.linalg.lu_factor_ex(a, pivot=False)

  def test_linalg_lu_factor_ex_rank_too_low(self):
    a = torch.ones(4, device=et.device())

    # We need to call the out-of-place variant of linalg.lu_factor_ex() op, so
    # that we don't go through the fallback. Otherwise, the meta kernel will
    # catch this error before it reaches TorchTPU implementation.
    out = (
        torch.empty(4, device=et.device()),
        torch.empty(4, device=et.device()),
        torch.empty(4, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_factor_ex(): input tensor expected to have at least 2"
            " dimensions, got 1"
        ),
        cpu=(
            "torch.lu_factor: Expected tensor with 2 or more dimensions. Got"
            " size: [4] instead"
        ),
    ):
      torch.linalg.lu_factor_ex(a, out=out)

  def test_lu_unpack_data_rank_too_low(self):
    data = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)

    # Call the out-of-place variant of linalg.lu_unpack() op.
    out = _make_lu_unpack_outputs(p=(4,), l=(4,), u=(4,))

    with et.assert_raises_message(
        RuntimeError,
        tpu="lu_unpack(): lu_data must have at least 2 dimensions, got 1",
        cpu=(
            "torch.lu_unpack: Expected tensor with 2 or more dimensions. Got"
            " size: [4] instead"
        ),
    ):
      torch.lu_unpack(data, pivots, out=out)

  def test_lu_solve_rank_too_low(self):
    lu = torch.ones(4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="linalg_lu_solve(): lu must have at least 2 dimensions, got 1",
        cpu=(
            "torch.linalg.lu_solve: The input tensor A must have at least 2"
            " dimensions."
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_rectangular_matrix(self):
    lu = torch.ones(4, 2, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="linalg_lu_solve(): lu must be square, got 4 and 2",
        cpu=(
            "torch.linalg.lu_solve: A must be batches of square matrices, but"
            " they are 4 by 2 matrices"
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_dimensions_mismatch(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 4, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_solve(): b must have compatible dimensions with lu, got"
            " b.shape[-2:]=(3, 4) and lu.shape[-2:]=(4, 4), and left=1"
        ),
        cpu=(
            "linalg.lu_solve: Incompatible shapes of A and B for the equation"
            " AX = B (4x4 and 3x4)"
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_pivots_invalid_dimensions(self):
    lu = torch.ones(3, 3, device=et.device())
    pivots = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_solve(): pivots must have one less dimension than the"
            " tensor, got 2 and 2"
        ),
        cpu=(
            "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the"
            " same, but got pivots with shape [2, 3] instead"
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_batch_dimensions_mismatch(self):
    lu = torch.ones(3, 3, 3, device=et.device())
    pivots = torch.ones(2, 3, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, 3, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(3, 3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_solve(): pivots and tensor must have the same batch"
            " dimensions, got [3] and [2]"
        ),
        cpu=(
            "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the"
            " same, but got pivots with shape [2, 3] instead"
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  def test_lu_solve_pivots_dimension_too_high(self):
    lu = torch.ones(3, 3, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(3, 3, device=et.device())

    # Call the out-of-place variant of linalg.lu_solve() op.
    out = torch.empty(3, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=(
            "linalg_lu_solve(): pivots size must be less than or equal to the"
            " size of the matrix, got 4 and 3"
        ),
        cpu=(
            "linalg.lu_solve: Number of pivots per batch should be same as the"
            " dimension of the matrix"
        ),
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)


if __name__ == "__main__":
  absltest.main()
