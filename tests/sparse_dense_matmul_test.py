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

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import op_testing


class SparseDenseMatmulTest(
    op_testing.TorchTpuTestBase, parameterized.TestCase
):
  """Tests for sparse_dense_matmul op.

  This operator requires TPU v5e hardware because it uses SparseCore.
  """

  def _get_inputs(self, device):
    row_pointers = torch.tensor(
        [
            3,
            9,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,  # input for SC 0
            3,
            9,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,  # input for SC 1
        ],
        dtype=torch.int32,
        device=device,
    )
    embedding_ids = torch.tensor(
        [
            0,
            1,
            2,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            1,
            3,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
        ],
        dtype=torch.int32,
        device=device,
    )
    sample_ids = torch.tensor(
        [
            3,
            0,
            2,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2,
            3,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
        ],
        dtype=torch.int32,
        device=device,
    )
    gains = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            1.0,
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ],
        dtype=torch.float32,
        device=device,
    )
    embedding_table = (
        torch.arange(32, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .repeat(1, 8)
    )
    return row_pointers, embedding_ids, sample_ids, gains, embedding_table

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.ops.torch_tpu.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=16,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      matmul_fn = torch.compile(matmul_fn, fullgraph=True)

    out = matmul_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table,
    )

    expected = torch.tensor(
        [
            [1.0] * 8,
            [16.0] * 8,
            [2.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [17.0] * 8,
            [3.0] * 8,
            [0.0] * 8,
            [1.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
        ],
        dtype=torch.float32,
    )

    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())


if __name__ == "__main__":
  absltest.main()
