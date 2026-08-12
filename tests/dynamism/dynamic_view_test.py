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

"""Test demonstrating dynamic execution through torch.compile() with input/output as dynamic views."""

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend


class DynamicViewTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  @absltest.skip("Dynamic view as input not supported")
  def test_static_size_dynamic_stride_input(self):
    class SlidingWindowCacheUpdate(torch.nn.Module):

      def __init__(self, sliding_window=128):
        super().__init__()
        self.sliding_window = sliding_window

      def forward(self, new_k, past_k=None):
        if past_k is not None:
          full_k = torch.cat([past_k, new_k], dim=2)
        else:
          full_k = new_k

        # Only retain the last sliding_window - 1 tokens in the returned cache.
        next_past_k = full_k[:, :, -self.sliding_window + 1 :, :]
        return next_past_k

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    model = (
        SlidingWindowCacheUpdate(sliding_window=128)
        .to(self.device, dtype=torch.float32)
        .eval()
    )
    compiled_model = torch.compile(model, backend=tpu_backend)

    with torch.no_grad():
      # 1. Prefill (seq_len = 256 >= sliding_window 128):
      # Model returns past_k_1 of length sliding_window - 1 = 127.
      new_k_prefill = torch.randn(
          16, 8, 256, 8, device="cpu", dtype=torch.float32
      ).to(self.device)
      past_k_1 = compiled_model(new_k_prefill, None)

      # 2. Decode Step 1 (past_k length 127 -> full_k length 128 -> past_k_2 length 127):
      # Compiles static decode frame for past_k size 127.
      new_k_token_1 = torch.randn(
          16, 8, 1, 8, device="cpu", dtype=torch.float32
      ).to(self.device)
      past_k_2 = compiled_model(new_k_token_1, past_k_1)

      # 3. Decode Step 2 (past_k length 127 -> full_k length 128 -> dynamic frame):
      # past_k_2 was produced by model's slice from dynamic full_k_1.
      # When past_k_2 enters compiled_model as an input argument:
      #   - Size is static: torch.Size([16, 8, 127, 8]) (because length is always 127)
      #   - Stride is dynamic: (8*s0, s0, 8, 1) (inherited from the dynamic full_k buffer)
      new_k_token_2 = torch.randn(
          16, 8, 1, 8, device="cpu", dtype=torch.float32
      ).to(self.device)
      _ = compiled_model(new_k_token_2, past_k_2)

  @absltest.skip("Dynamic view as input not supported")
  def test_dynamic_size_dynamic_stride_input(self):
    class DynamicViewModule(torch.nn.Module):

      def forward(self, x):
        # Operates on an input tensor that is a dynamic non-contiguous view
        return x + 1.0

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    model = DynamicViewModule().to(self.device, dtype=torch.float32).eval()
    compiled_model = torch.compile(model, backend=tpu_backend)

    with torch.no_grad():
      # Step 1: Pass a dynamic non-contiguous view (transpose on dynamic seq_len)
      base_1 = torch.randn(16, 8, 20, 64, device="cpu", dtype=torch.float32)
      view_1 = base_1.to(self.device).transpose(
          1, 2
      )  # shape [16, 20, 8, 64], non-contiguous
      _ = compiled_model(view_1)

      # Step 2: Pass a different dynamic seq_len with the same dynamic non-contiguous view
      base_2 = torch.randn(16, 8, 30, 64, device="cpu", dtype=torch.float32)
      view_2 = base_2.to(self.device).transpose(
          1, 2
      )  # shape [16, 30, 8, 64], non-contiguous
      _ = compiled_model(view_2)

      # Step 3: Pass a different dynamic seq_len with the same dynamic non-contiguous view
      base_3 = torch.randn(16, 8, 40, 64, device="cpu", dtype=torch.float32)
      view_3 = base_3.to(self.device).transpose(
          1, 2
      )  # shape [16, 40, 8, 64], non-contiguous
      _ = compiled_model(view_3)


if __name__ == "__main__":
  absltest.main()
