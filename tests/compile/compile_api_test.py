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
from torch_tpu._internal.compile import tpu_torch_compile


# TODO: add more test coverage for the direct compile API.
class CompileApiTest(absltest.TestCase):

  def test_build_mlir(self):
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


if __name__ == '__main__':
  absltest.main()
