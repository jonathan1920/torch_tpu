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

"""Tests for device_utils."""

from absl.testing import absltest
import torch
from examples.benchmarks.e2e import device_utils


class DeviceUtilsTest(absltest.TestCase):

  def test_get_peak_host_compilation_memory_mb(self):
    class SimpleModel(torch.nn.Module):

      def forward(self, x):
        return x + 1

    device = torch.device("tpu")
    model = SimpleModel().to(device)
    x = torch.randn(10, device=device)

    # We need to use torch.compile with the tpu backend
    compiled_model = torch.compile(model, backend="tpu")

    # Run it to trigger compilation
    _ = compiled_model(x)
    torch.tpu.synchronize()

    peak_mem = device_utils.get_peak_host_compilation_memory_mb("tpu")
    self.assertIsNotNone(peak_mem)
    self.assertGreater(peak_mem, 0.0)


if __name__ == "__main__":
  absltest.main()
