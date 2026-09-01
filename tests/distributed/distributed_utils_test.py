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

"""Tests for distributed_utils module."""

import os
import random
from unittest import mock

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _worker_seed_check(
    expected_base_seed: int,
    output_dir: str,
) -> None:
  rank = int(os.environ["RANK"])
  expected_seed = expected_base_seed + rank

  # Check python random
  py_random_val = random.random()
  expected_py_val = random.Random(expected_seed).random()

  # Check torch random
  torch_val = torch.rand(4)
  torch.manual_seed(expected_seed)
  expected_torch_val = torch.rand(4)

  assert (
      py_random_val == expected_py_val
  ), f"Rank {rank}: expected py rand {expected_py_val}, got {py_random_val}"
  assert torch.equal(
      torch_val, expected_torch_val
  ), f"Rank {rank}: expected torch rand {expected_torch_val}, got {torch_val}"

  rank_file = os.path.join(output_dir, f"rank_{rank}.txt")
  with open(rank_file, "w") as f:
    f.write(f"{py_random_val},{torch_val.tolist()}\n")


def _worker_xla_flags_check(output_dir: str) -> None:
  rank = os.environ["RANK"]
  xla_flags = os.environ.get("XLA_FLAGS", "")
  rank_file = os.path.join(output_dir, f"rank_{rank}_xla_flags.txt")
  with open(rank_file, "w") as f:
    f.write(xla_flags)


def _worker_env_check() -> None:
  local_rank = int(os.environ["LOCAL_RANK"])
  node_rank = int(os.environ["GROUP_RANK"])
  nproc = int(os.environ["LOCAL_WORLD_SIZE"])
  world_size = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["RANK"])

  assert rank == node_rank * nproc + local_rank
  assert world_size == 2
  assert nproc == 2


class DistributedUtilsTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_automatic_per_rank_seeding(self) -> None:
    """Verifies default per-rank RNG seeding based on RepeatableTest seed."""
    output_dir = self.create_tempdir().full_path
    base_seed = seed_test_utils.RepeatableTest.choose_seed()

    distributed_utils.dist_run(
        2,
        _worker_seed_check,
        base_seed,
        output_dir,
    )

    rank0_file = os.path.join(output_dir, "rank_0.txt")
    rank1_file = os.path.join(output_dir, "rank_1.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      rank0_content = f.read().strip()
    with open(rank1_file) as f:
      rank1_content = f.read().strip()

    # Rank 0 and Rank 1 must have different random streams (rank diversity)
    self.assertNotEqual(rank0_content, rank1_content)

  def test_explicit_base_seed(self) -> None:
    """Verifies that an explicitly passed base_seed is used."""
    output_dir = self.create_tempdir().full_path
    custom_seed = 9876

    distributed_utils.dist_run(
        2,
        _worker_seed_check,
        custom_seed,
        output_dir,
        base_seed=custom_seed,
    )

    rank0_file = os.path.join(output_dir, "rank_0.txt")
    rank1_file = os.path.join(output_dir, "rank_1.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      rank0_content = f.read().strip()
    with open(rank1_file) as f:
      rank1_content = f.read().strip()

    self.assertNotEqual(rank0_content, rank1_content)

  def test_env_variables_set(self) -> None:
    """Verifies distributed environment variables set in worker."""
    distributed_utils.dist_run(
        2,
        _worker_env_check,
    )

  def test_xla_flags_expansion(self) -> None:
    """Verifies that $RANK / ${RANK} in XLA_FLAGS is expanded per worker."""
    output_dir = self.create_tempdir().full_path
    template_flag = "--xla_dump_to=/tmp/dumps/rank_${RANK} --xla_tpu_rank=$RANK"
    with mock.patch.dict(os.environ, {"XLA_FLAGS": template_flag}):
      distributed_utils.dist_run(
          2,
          _worker_xla_flags_check,
          output_dir,
      )

    rank0_file = os.path.join(output_dir, "rank_0_xla_flags.txt")
    rank1_file = os.path.join(output_dir, "rank_1_xla_flags.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      self.assertEqual(
          f.read().strip(),
          "--xla_dump_to=/tmp/dumps/rank_0 --xla_tpu_rank=0",
      )
    with open(rank1_file) as f:
      self.assertEqual(
          f.read().strip(),
          "--xla_dump_to=/tmp/dumps/rank_1 --xla_tpu_rank=1",
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
