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

"""Tests PyTorch distributed ops with torch.compile()."""
import unittest
from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal import env
from torch_tpu._internal.compile import TpuBackend
from torch_tpu._internal.distributed import tpu_env
from torch_tpu._internal.utils import utils
from torch_tpu.shims.g3_multiprocessing import g3_multiprocessing


def xfail_in_oss(func):
  if not env.IS_INTERNAL_TORCH_TPU:
    return unittest.expectedFailure(func)
  return func


def run_all_reduce_with_torch_compile(rank: int, world_size: int) -> None:  # pylint: disable=unused-argument
  """Tests all-reduce functionality."""

  def func(x):
    x = x + torch.zeros_like(x)
    # TODO: b/478944489 - Force a graph break to avoid the crash in the
    # distributed runtime in OSS.
    torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM)
    x = x + torch.abs(x)
    x = x + torch.ones_like(x)
    return x

  input_tpu = torch.tensor(
      [0.0, 1.0, float(rank), float(rank**2)], device="tpu"
  )
  backend = TpuBackend(debug=True)
  compiled = torch.compile(func, backend=backend)
  output_compiled = compiled(input_tpu)
  output_compiled_cpu = output_compiled.to("cpu")
  utils.assert_close(
      output_compiled_cpu, torch.tensor([1.0, 17.0, 57.0, 281.0])
  )

  # torch.distributed.ops usually introduce a graph break, so we are expecting
  # two graphs in the cache
  v = backend._compiled_executables
  assert len(v) == 2, "Expected 2 graphs, got %d" % len(v)

  # debug mode enabled so expect graphs to be set and in plaintext
  assert "torch.ops.aten.abs" not in v[0].graph_module_debug_str
  assert "stablehlo.abs" not in str(v[0].mlir_graph)

  assert "torch.ops.aten.abs" in v[1].graph_module_debug_str
  assert "stablehlo.abs" in str(v[1].mlir_graph)

  assert v[0].graph_module_debug_str != v[1].graph_module_debug_str
  assert v[0].mlir_graph != v[1].mlir_graph


class MultiTpuTorchCompileTest(absltest.TestCase):

  @xfail_in_oss
  def test_all_reduce_with_torch_compile(self):
    tpu_env.run_in_workers(8, run_all_reduce_with_torch_compile)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
