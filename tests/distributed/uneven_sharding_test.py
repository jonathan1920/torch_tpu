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

import os

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def _replicate_tensor(
    tensor_size: int, compiled: bool, pass_shape: bool = False
) -> None:
  """Redistributes a tensor from shard to replicate."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))

  def shard_and_redistribute(x):
    if pass_shape:
      global_shape = torch.Size((tensor_size,))
      global_stride = torch.empty(global_shape, device="meta").stride()
      dtensor = dt.DTensor.from_local(
          x,
          device_mesh,
          [dt.Shard(0)],
          shape=global_shape,
          stride=global_stride,
      )
    else:
      dtensor = dt.DTensor.from_local(x, device_mesh, [dt.Shard(0)])

    replicated = dtensor.redistribute(device_mesh, [dt.Replicate()])
    return replicated.to_local()

  # Create local tensor. We follow the torch.chunk strategy so that DTensor can
  # be correctly reconstructed. torch.chunk splits into max-size chunks, leaving
  # the last few ranks -- if any -- empty (size 0).
  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start
  local_tensor = torch.ones(local_size, dtype=torch.float32, device="tpu")

  if compiled:
    backend = tt_compile.TpuBackend(debug=True)
    func = torch.compile(
        shard_and_redistribute, backend=backend, fullgraph=True
    )
  else:
    func = shard_and_redistribute

  output = func(local_tensor)
  out = output.cpu()

  expected = torch.ones(tensor_size, dtype=torch.float32)
  utils.assert_close(out, expected)


class UnevenShardingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("eager_small", False, 4),
      ("compiled_small", True, 4),
      ("eager_uneven", False, 12),
      ("compiled_uneven", True, 12),
  )
  def test_replicate_tensor_with_shape(self, compiled, tensor_size):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(_replicate_tensor, world_size=8),
        tensor_size=tensor_size,
        compiled=compiled,
        pass_shape=True,
    )

  @parameterized.named_parameters(
      ("compiled_fail_small", True, 4, "Tensor shape mismatch"),
      ("eager_fail_small", False, 4, "Tensor shape mismatch"),
      ("compiled_fail_uneven", True, 12, "Tensor shape mismatch"),
      ("eager_fail_uneven", False, 12, "Tensor shape mismatch"),
  )
  def test_replicate_tensor_without_shape(
      self, compiled, tensor_size, expected_regex
  ):
    with self.assertRaisesRegex(Exception, expected_regex):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _replicate_tensor, world_size=8
          ),
          tensor_size=tensor_size,
          compiled=compiled,
          pass_shape=False,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
