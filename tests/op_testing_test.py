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
import torch
from torch.testing._internal import common_methods_invocations
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
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.dict(
            op_testing._GOLDEN_GPU_DATA,
            {
                self._testMethodName: {
                    op_testing.OpVariant.BASE.value: {
                        torch.float32: fake_samples
                    }
                }
            },
            clear=True,
        ),
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


if __name__ == "__main__":
  absltest.main()
