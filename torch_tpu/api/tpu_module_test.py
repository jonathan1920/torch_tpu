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
from torch_tpu.api import _device_module

_DevicePythonModule = _device_module._DeviceModule


class TpuModuleTest(absltest.TestCase, common_utils.TestCase):

  def setUp(self):
    super().setUp()
    # Reset class variables before each test
    _DevicePythonModule._autocast_enabled = False
    _DevicePythonModule._autocast_dtype = torch.float16

  def test_init_tpu_runtime(self):
    # Test TPU initialization
    _DevicePythonModule._init_runtime(device_type="tpu")
    self.assertTrue(_DevicePythonModule.is_initialized())
    self.assertGreater(_DevicePythonModule.device_count(), 0)


if __name__ == "__main__":
  absltest.main()
