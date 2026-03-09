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
import os
import unittest
from absl.testing import absltest
import torch
from torch import distributed as dist
# from torch.google import distributed as g3_distributed
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import env
from torch_tpu._internal.compile import TpuBackend
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from torch_tpu.shims.g3_multiprocessing import g3_multiprocessing


def expected_to_fail_in_oss(func):
  """Annotates a test to fail in OSS."""

  return func if env.IS_INTERNAL_TORCH_TPU else unittest.expectedFailure(func)


def run_all_reduce_with_torch_compile() -> None:
  """Tests all-reduce functionality."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])

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

  # TODO: fix the failure "compile_mlir(): failed to validate and reorder
  # inputs" error in OSS.
  @expected_to_fail_in_oss
  def test_all_reduce_with_torch_compile(self):
    dist.torchrun(
        singlehost_wrapper.tpu_env_wrapper(run_all_reduce_with_torch_compile),
        nproc_per_node=8,
    )()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
