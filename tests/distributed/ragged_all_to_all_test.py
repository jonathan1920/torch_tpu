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

"""Tests for ragged_all_to_all in a distributed environment on TPU."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_ragged_all_to_all_test() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])

  #  This is a single group of 4.
  replica_groups = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64).to("tpu")

  # Correct output offsets for single group of 4: [rank, rank, rank, rank]
  output_offsets = torch.tensor([rank, rank, rank, rank], dtype=torch.int32).to(
      "tpu"
  )

  send_sizes = torch.tensor([1, 1, 1, 1], dtype=torch.int32).to("tpu")
  input_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("tpu")
  recv_sizes = torch.tensor([1, 1, 1, 1], dtype=torch.int32).to("tpu")

  operand = torch.full((4, 1, 128), rank, dtype=torch.int32).to("tpu")
  output = torch.zeros(4, 1, 128, dtype=torch.int32).to("tpu")

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      replica_groups,
  )

  expected = torch.tensor([0, 1, 2, 3], dtype=torch.int32).to("tpu")
  utils.assert_close(result[:, 0, 0], expected)

  dist.barrier()
  dist.destroy_process_group()


class RaggedAllToAllTest(absltest.TestCase):
  """Tests the ragged_all_to_all TPU collective operation.

  This test initializes a distributed environment and performs a
  ragged_all_to_all operation on a single group of 4 TPUs. Each TPU sends its
  rank to all other TPUs in the group.
  """

  def test_ragged_all_to_all(self):
    distributed_utils.dist_run(
        nproc_per_node=4,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_all_to_all_test),
    )


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
