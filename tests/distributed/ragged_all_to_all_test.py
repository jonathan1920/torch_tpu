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
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_ragged_all_to_all_test() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  output_offsets = torch.tensor([rank] * world_size, dtype=torch.int32).tpu()

  send_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(range(world_size), dtype=torch.int32).tpu()
  recv_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()

  offset = rank * world_size
  operand = torch.arange(offset, offset + world_size, dtype=torch.int32).tpu()
  output = torch.zeros(world_size, dtype=torch.int32).tpu()

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Simply a transpose.
  expected = torch.arange(
      rank,
      rank + world_size * world_size,
      world_size,
      dtype=torch.int32,
  )

  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


def run_ragged_all_to_all_uneven_test() -> None:
  """Tests the lower-level torch.ops.tpu.ragged_all_to_all custom call with uneven splits.

  This test directly exercises the underlying TPU custom call by manually
  computing:
    1. send_sizes and recv_sizes: non-uniform slice sizes where rank i sends
       2 elements to rank (i+1)%world_size and 1 element to all other ranks.
    2. input_offsets: the starting index in the local `operand` tensor for each
       destination rank's outgoing slice.
    3. output_offsets: the exact memory offset in the destination rank's
       `output` buffer where this rank's slice should be placed (accounting
       for the cumulative size of slices received from preceding ranks
       src < rank).
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Communication split matrix: matrix[src][dst] is the number of elements
  # sent from src to dst. Rank i sends 2 elements to (i+1)%world_size, 1 to others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  send_sizes_list = matrix[rank]
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  # Calculate local input offsets for each destination slice in operand.
  input_offsets_list = []
  curr = 0
  for s in send_sizes_list:
    input_offsets_list.append(curr)
    curr += s

  # Calculate remote output offsets: where each outgoing slice will be placed
  # on destination `dst`. On rank `dst`, slices from src=0..rank-1 arrive before `rank`.
  output_offsets_list = []
  for dst in range(world_size):
    dst_offset = sum(matrix[src][dst] for src in range(rank))
    output_offsets_list.append(dst_offset)

  send_sizes = torch.tensor(send_sizes_list, dtype=torch.int32).tpu()
  recv_sizes = torch.tensor(recv_sizes_list, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(input_offsets_list, dtype=torch.int32).tpu()
  output_offsets = torch.tensor(output_offsets_list, dtype=torch.int32).tpu()

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # Initialize unique input elements: rank 0 gets 0..N, rank 1 gets 100..100+N, etc.
  operand = torch.arange(
      rank * 100, rank * 100 + total_send, dtype=torch.int32
  ).tpu()
  output = torch.zeros(total_recv, dtype=torch.int32).tpu()

  # Invoke the ragged_all_to_all ATen custom call.
  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Construct expected output buffer from received slices.
  expected_slices = []
  for src in range(world_size):
    count = matrix[src][rank]
    src_send_offset = sum(matrix[src][d] for d in range(rank))
    expected_slices.append(
        torch.arange(
            src * 100 + src_send_offset,
            src * 100 + src_send_offset + count,
            dtype=torch.int32,
        )
    )
  expected = torch.cat(expected_slices)
  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


def run_dist_all_to_all_single_uneven_test() -> None:
  """Tests high-level torch.distributed.all_to_all_single with uneven split sizes.

  Validates that ProcessGroupTpu automatically calculates input/output offsets,
  synchronizes remote destination offsets across ranks via c10d::Store, and
  dispatches the underlying ragged_all_to_all custom call seamlessly.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # 1. Define the uneven communication pattern.
  # matrix[src][dst] defines the number of elements sent from `src` to `dst`.
  # Here, rank i sends 2 elements to peer (i+1)%world_size, and 1 element to all others.
  matrix = [
      [(2 if j == (i + 1) % world_size else 1) for j in range(world_size)]
      for i in range(world_size)
  ]
  # Slice sizes sent from this rank to all destination ranks.
  send_sizes_list = matrix[rank]
  # Slice sizes this rank expects to receive from all source ranks.
  recv_sizes_list = [matrix[src][rank] for src in range(world_size)]

  total_send = sum(send_sizes_list)
  total_recv = sum(recv_sizes_list)

  # 2. Populate unique, rank-identifiable input data.
  # For instance, rank 0 has values 0..total_send, rank 1 has 100..100+total_send, etc.
  operand = torch.arange(
      rank * 100, rank * 100 + total_send, dtype=torch.int32
  ).tpu()
  # Pre-allocate output buffer matching total received element count across all sources.
  output = torch.zeros(total_recv, dtype=torch.int32).tpu()

  # 3. Invoke standard PyTorch collective with non-uniform split sizes.
  # ProcessGroupTpu calculates local/remote offsets and lowers to ragged_all_to_all.
  torch.distributed.all_to_all_single(
      output,
      operand,
      recv_sizes_list,
      send_sizes_list,
  )

  # 4. Reconstruct ground-truth expected buffer from the split matrix.
  # For each source rank `src`, extract the exact slice that `src` sent to `rank`.
  expected_slices = []
  for src in range(world_size):
    count = matrix[src][rank]
    # Calculate offset in source rank's operand where the slice sent to `rank` begins.
    src_send_offset = sum(matrix[src][d] for d in range(rank))
    expected_slices.append(
        torch.arange(
            src * 100 + src_send_offset,
            src * 100 + src_send_offset + count,
            dtype=torch.int32,
        )
    )
  expected = torch.cat(expected_slices)

  # 5. Verify the received output matches the expected tensor values.
  utils.assert_close(output.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


class RaggedAllToAllTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests the ragged_all_to_all TPU collective operation.

  This test initializes a distributed environment and performs a
  ragged_all_to_all operation on a single group of 8 TPUs. Each TPU sends its
  rank to all other TPUs in the group.
  """

  def test_ragged_all_to_all(self):
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_all_to_all_test),
    )

  def test_ragged_all_to_all_uneven(self):
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_ragged_all_to_all_uneven_test
        ),
    )

  def test_dist_all_to_all_single_uneven(self):
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dist_all_to_all_single_uneven_test
        ),
    )


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
