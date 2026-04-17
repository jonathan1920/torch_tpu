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

"""Tests for the device module with TPU backend."""

from absl.testing import absltest
import torch
from tests import device_module_testing


# Test cases are defined in device_module_testing.DeviceModuleBase
class TpuDeviceModuleTest(
    device_module_testing.DeviceModuleBase, absltest.TestCase
):

  @property
  def device_module(self):
    return torch.tpu


if __name__ == "__main__":
  absltest.main()
