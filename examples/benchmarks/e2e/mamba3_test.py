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

"""Tests for mamba3."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from examples.benchmarks.e2e import mamba3
from tests import seed_test_utils


class Mamba3Test(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 1},
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 2},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 1},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 2},
  )
  def test_forward_eval(self, is_mimo: bool, mimo_rank: int, ngroups: int = 1):
    layer = mamba3.Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=ngroups,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
    )
    layer.eval()
    u = torch.randn(2, 16, 64)
    with torch.no_grad():
      out = layer(u)
    self.assertEqual(out.shape, (2, 16, 64))
    self.assertFalse(torch.isnan(out).any())

  @parameterized.parameters(
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 1},
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 2},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 1},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 2},
  )
  def test_forward_backward_train(
      self, is_mimo: bool, mimo_rank: int, ngroups: int = 1
  ):
    layer = mamba3.Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=ngroups,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
    )
    layer.train()
    u = torch.randn(2, 16, 64, requires_grad=True)
    out = layer(u)
    loss = out.sum()
    loss.backward()
    self.assertIsNotNone(u.grad)
    self.assertEqual(u.grad.shape, (2, 16, 64))
    self.assertFalse(torch.isnan(u.grad).any())

    for name, param in layer.named_parameters():
      if param.requires_grad:
        self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
        self.assertFalse(
            torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        )

  def test_invalid_ngroups_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates ngroups divisibility.
        ValueError, "nheads must be divisible by ngroups"
    ):
      mamba3.Mamba3(
          d_model=64,
          d_state=32,
          expand=2,
          headdim=32,  # nheads = (2 * 64) // 32 = 4
          ngroups=3,  # 4 % 3 != 0
      )


if __name__ == "__main__":
  absltest.main()
