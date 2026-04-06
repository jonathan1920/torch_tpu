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

"""Directly test the PyBind11 API for compiled mode, without using Dynamo."""

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile


@torch.compile
def global_compile_plus_one(x):
  return x + 1


def get_mock_lookup_backend():
  return absltest.mock.patch(
      'torch._dynamo.backends.registry.lookup_backend',
      wraps=torch._dynamo.backends.registry.lookup_backend,
  )


def eager_mode_defer_all():
  """Enable EagerMode.INTERNAL_DEFER_ALL.

  This prevents tensor materialization during Python execution and enables
  `build_mlir` to trace the deferred ops and generate MLIR.

  Returns:
    A context manager for setting the execution mode.
  """
  return execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL)


# TODO: add more test coverage for the direct compile API.
class CompileApiTest(absltest.TestCase):

  def test_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=api.tpu_device())
      y = torch.ones(10, device='cpu').to(device=api.tpu_device())
      z = x + y
    result_tensors = [z]
    argument_tensors = [x, y]
    donate_args = (1,)
    mlir = tpu_torch_compile.build_mlir(
        result_tensors, argument_tensors, donate_args=donate_args
    )
    mlir_str = str(mlir)
    self.assertIn(
        'func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>'
        ' {jax.buffer_donor = true})',
        mlir_str,
    )
    self.assertIn('stablehlo.add', mlir_str)

  def test_extra_input_to_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=api.tpu_device())
      y = torch.ones(10, device='cpu').to(device=api.tpu_device())
      extra = torch.ones(10, device='cpu').to(device=api.tpu_device())
      z = x + y
    result_tensors = [z]
    argument_tensors = [x, y, extra]
    mlir = tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    mlir_str = str(mlir)
    self.assertIn('func @main', mlir_str)
    self.assertIn('%arg2', mlir_str)

  def test_missing_input_to_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=api.tpu_device())
      y = torch.ones(10, device='cpu').to(device=api.tpu_device())
      z = x + y
    result_tensors = [z]
    argument_tensors = [x]
    with self.assertRaises(RuntimeError) as err:
      tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    self.assertIn(
        'identified an input that was not provided',
        str(err.exception),
    )

  def test_compile_backend_defaults_to_tpu(self):
    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10, device=api.tpu_device())
      torch.compile(lambda arg: arg + 1)(x)

      mock_lookup_backend.assert_called_with('tpu')

  def test_global_compile_decorator_backend_defaults_to_tpu(self):
    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10, device=api.tpu_device())
      global_compile_plus_one(x)

      mock_lookup_backend.assert_called_with('tpu')

  def test_compile_explicit_backend_is_respected(self):

    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10)
      torch.compile(lambda arg: arg + 1, backend='aot_eager')(x)

      mock_lookup_backend.assert_called_with('aot_eager')

  def test_compile_default_backend_no_recompilations(self):
    num_calls = 0

    def user_context() -> str:
      nonlocal num_calls
      num_calls += 1
      return 'user_context: ' + str(num_calls)

    torch._dynamo.register_hook_for_recompile_user_context(user_context)

    def f(x):
      return x + 1

    for _ in range(10):
      torch.compile(f)(torch.randn(1, 5, device=api.tpu_device()))
    self.assertEqual(num_calls, 1)


if __name__ == '__main__':
  absltest.main()
