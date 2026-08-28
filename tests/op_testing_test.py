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

"""Unit tests for the op_testing framework."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from torch.testing._internal import common_methods_invocations
from torch_tpu._internal.utils import test_utils
from tests import op_testing


op_db = common_methods_invocations.op_db


class FakeSample:
  """A fake test sample for an op."""

  def __init__(
      self,
      name: str,
      input_value: torch.Tensor,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ):
    self.name = name
    self.input = input_value
    self.args = args
    self.kwargs = kwargs


class OpTestingTest(op_testing.OpInfoTestBase):
  """Tests for the op_testing framework itself."""

  def test_torch_tpu_vs_gpu_max_samples(self):
    op = next(op for op in op_db if op.name == "add")
    fake_samples = [
        (
            op_testing.OpInput(FakeSample(f"s{i}", torch.zeros(1), (), {})),
            op_testing.OpOutput(torch.zeros(1)),
        )
        for i in range(5)
    ]
    golden_data = op_testing.GoldenGpuData()
    for op_input, op_output in fake_samples:
      golden_data.add(
          self._testMethodName,
          op_testing.OpVariant.BASE,
          torch.float32,
          op_input,
          op_output,
      )
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
    ):
      # 1. max_samples argument limits returned samples.
      res = self._get_golden_input_output_pairs(
          op=op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=2,
      )
      self.assertEqual(res, fake_samples[:2])

      # 2. When max_samples is None, --max_samples_per_op_dtype flag is respected.
      with flagsaver.flagsaver(max_samples_per_op_dtype=3):
        res = self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=None,
        )
        self.assertEqual(res, fake_samples[:3])

      # 3. Default or negative max_samples returns all samples.
      with flagsaver.flagsaver(max_samples_per_op_dtype=-1):
        res = self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=None,
        )
        self.assertEqual(res, fake_samples)

  @parameterized.named_parameters(
      ("coo", lambda t: t.to_sparse()),
      ("csr", lambda t: t.to_sparse_csr()),
      ("csc", lambda t: t.to_sparse_csc()),
      ("bsr", lambda t: t.to_sparse_bsr(blocksize=(1, 1))),
      ("bsc", lambda t: t.to_sparse_bsc(blocksize=(1, 1))),
  )
  def test_plistlib_sparse_tensor_serialization(self, to_sparse_fn):
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sparse = to_sparse_fn(dense)
    encoded = op_testing._to_plistlib_compatible(sparse)
    restored = op_testing._from_plistlib_compatible(encoded)
    test_utils.assert_close(restored, dense)

  def test_supports_out(self):
    add_op = next(op for op in op_db if op.name == "add")
    self.assertTrue(op_testing._supports_out(add_op))

    foreach_abs_op = op_testing._get_op("_foreach_abs")
    self.assertFalse(op_testing._supports_out(foreach_abs_op))

  def test_find_cast_pairs(self):
    # Distinct dtypes with both allowed upcast (int32 -> float32) and illegal downcast (float32 -> int32).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.int32])
    self.assertEqual(pairs.allowed_pair, (torch.int32, torch.float32))
    self.assertEqual(pairs.illegal_pair, (torch.float32, torch.int32))

    # All pairs allowed (e.g. float32 <-> float64).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.float64])
    self.assertEqual(pairs.allowed_pair, (torch.float32, torch.float64))
    self.assertIsNone(pairs.illegal_pair)

  def test_find_cast_pairs_duplicate_dtypes(self):
    # Duplicate entries of the same dtype should not yield identical pairs (e.g. float32 -> float32).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.float32])
    self.assertIsNone(pairs.allowed_pair)
    self.assertIsNone(pairs.illegal_pair)

    # Duplicates mixed with other dtypes should yield distinct pairs.
    pairs = op_testing.find_cast_pairs(
        [torch.float32, torch.float32, torch.int32]
    )
    self.assertEqual(pairs.allowed_pair, (torch.int32, torch.float32))
    self.assertEqual(pairs.illegal_pair, (torch.float32, torch.int32))

  def test_find_cast_pairs_insufficient_dtypes(self):
    self.assertEqual(
        op_testing.find_cast_pairs([]),
        op_testing.CastPairs(illegal_pair=None, allowed_pair=None),
    )
    self.assertEqual(
        op_testing.find_cast_pairs([torch.float32]),
        op_testing.CastPairs(illegal_pair=None, allowed_pair=None),
    )


if __name__ == "__main__":
  absltest.main()
