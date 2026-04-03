# Copyright 2025 Google LLC
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

"""Tests TpuModule."""

from absl.testing import absltest
import torch
from torch.testing._internal import common_utils
from torch_tpu import api
from torch_tpu.api import _device_module

_DevicePythonModule = _device_module._DeviceModule


class TpuModuleTest(absltest.TestCase, common_utils.TestCase):

  def setUp(self):
    super().setUp()
    # Reset class variables before each test
    _DevicePythonModule._autocast_enabled = False
    _DevicePythonModule._autocast_dtype = torch.float16
    # Ensure TPU backend is initialized and registered
    api.tpu_device()

  def test_memory_stats(self):
    stats = torch.accelerator.memory_stats()
    self.assertIn("active_bytes.all.current", stats)
    self.assertIn("active_bytes.all.peak", stats)
    self.assertIn("reserved_bytes.all.current", stats)
    self.assertIn("reserved_bytes.all.peak", stats)

  def test_memory_allocation(self):
    # Get initial state
    allocated_before = torch.accelerator.memory_allocated()
    max_allocated_before = torch.accelerator.max_memory_allocated()
    self.assertGreaterEqual(allocated_before, 0)
    self.assertEqual(max_allocated_before, allocated_before)

    # Allocate a known amount of memory (e.g., 4MB)
    # float32 is 4 bytes, so 1024 * 1024 elements is 4MB
    size_in_bytes = 4 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(num_elements, device="tpu")
    t = t + 1
    # Synchronize by copying to CPU (forces execution)
    _ = t.cpu()

    # Verify allocation increased
    allocated_after = torch.accelerator.memory_allocated()
    max_allocated_after = torch.accelerator.max_memory_allocated()

    self.assertGreater(allocated_after, allocated_before)
    self.assertGreaterEqual(allocated_after, allocated_before + size_in_bytes)

    # Verify max memory allocated updated
    self.assertGreater(max_allocated_after, 0)
    self.assertGreater(max_allocated_after, max_allocated_before)
    self.assertGreaterEqual(max_allocated_after, allocated_after)


if __name__ == "__main__":
  absltest.main()
