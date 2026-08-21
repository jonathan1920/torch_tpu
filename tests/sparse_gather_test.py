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

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from tests import op_testing

PAD_WIDTH = 8
INT_MAX = 2**31 - 1


def pad_length(arr: list[int]) -> list[int]:
  """Pads the array to the nearest multiple of PAD_WIDTH."""
  pad_len = (PAD_WIDTH - len(arr)) % PAD_WIDTH
  assert 0 <= pad_len < PAD_WIDTH
  assert (len(arr) + pad_len) % PAD_WIDTH == 0
  return arr + [INT_MAX] * pad_len


def round_up(x: int, y: int) -> int:
  """Rounds up x to the nearest multiple of y."""
  return math.ceil(x / y) * y


def build_csr(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
  """Builds CSR row_pointers and indices from a list of lists (COO format)."""
  row_ptrs_list = []
  indices_list = []
  curr_row_end = 0
  for row in rows:
    curr_row_end += len(row)
    padded_row = pad_length(row)
    indices_list.extend(padded_row)
    row_ptrs_list.append(curr_row_end)
    curr_row_end = round_up(curr_row_end, PAD_WIDTH)

  return (
      torch.tensor(row_ptrs_list, dtype=torch.int32),
      torch.tensor(indices_list, dtype=torch.int32),
  )


class SparseGatherTest(op_testing.TorchTpuTestBase):
  """Tests for sparse_gather op on TPU SparseCore."""

  def _get_inputs(
      self,
      device: torch.device,
      table_depth: int = 1024,
      table_width: int = 16,
      num_rows: int = 16,
      max_non_zeroes_per_row: int = 64,
  ):
    torch.manual_seed(42)
    np.random.seed(42)

    operand = (
        torch.arange(0, table_depth, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, table_width)
    )

    coos = []
    for _ in range(num_rows):
      row_size = np.random.randint(1, max_non_zeroes_per_row // 2 + 1)
      row_indices = np.random.randint(0, table_depth, size=row_size).tolist()
      coos.append(row_indices)

    row_pointers, indices = build_csr(coos)
    pad_needed = num_rows * max_non_zeroes_per_row - len(indices)
    if pad_needed > 0:
      indices = torch.cat(
          [indices, torch.full((pad_needed,), INT_MAX, dtype=torch.int32)]
      )

    return (
        row_pointers.to(device),
        indices.to(device),
        operand.to(device),
        coos,
        max_non_zeroes_per_row,
    )

  @parameterized.parameters(False, True)
  def test_sparse_gather_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, indices, operand, coos, max_non_zeroes_per_row = (
        self._get_inputs(device)
    )

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes_per_row
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)

    # Compute expected result
    all_indices = []
    for row in coos:
      all_indices.extend(row)
    expected = operand.cpu()[torch.tensor(all_indices, dtype=torch.long)]

    # Verified output slice
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu()[: expected.shape[0]],
    )


if __name__ == "__main__":
  absltest.main()
