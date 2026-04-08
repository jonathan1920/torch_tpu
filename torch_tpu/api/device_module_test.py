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


from unittest import mock

from absl.testing import absltest
import torch
from torch.testing._internal import common_utils
from torch_tpu.api import _device_module

_DevicePythonModule = _device_module._DeviceModule


class DeviceModuleTest(absltest.TestCase, common_utils.TestCase):

  def setUp(self):
    super().setUp()
    # Reset class variables before each test
    _DevicePythonModule._autocast_enabled = False
    _DevicePythonModule._autocast_dtype = torch.float16

  def test_get_amp_supported_dtype(self):
    supported_dtypes = _DevicePythonModule.get_amp_supported_dtype()
    self.assertIsInstance(supported_dtypes, list)
    self.assertNotEmpty(supported_dtypes)
    for dtype in supported_dtypes:
      self.assertIsInstance(dtype, torch.dtype)

  def test_is_autocast_enabled(self):
    self.assertFalse(_DevicePythonModule._is_autocast_enabled())
    _DevicePythonModule._set_autocast_enabled(True)
    self.assertTrue(_DevicePythonModule._is_autocast_enabled())

  def test_get_autocast_dtype(self):
    self.assertEqual(_DevicePythonModule._get_autocast_dtype(), torch.float16)
    _DevicePythonModule._set_autocast_dtype(torch.bfloat16)
    self.assertEqual(_DevicePythonModule._get_autocast_dtype(), torch.bfloat16)

  def test_set_autocast_enabled(self):
    _DevicePythonModule._set_autocast_enabled(True)
    self.assertTrue(_DevicePythonModule._autocast_enabled)
    _DevicePythonModule._set_autocast_enabled(False)
    self.assertFalse(_DevicePythonModule._autocast_enabled)

  def test_set_autocast_dtype(self):
    supported_dtypes = _DevicePythonModule.get_amp_supported_dtype()
    for dtype in supported_dtypes:
      _DevicePythonModule._set_autocast_dtype(dtype)
      self.assertEqual(_DevicePythonModule._autocast_dtype, dtype)

    with self.assertRaises(ValueError):
      _DevicePythonModule._set_autocast_dtype(None)

  def test_dump_on_miss(self):
    # Default should be False
    self.assertFalse(_DevicePythonModule._get_dump_on_cache_miss())
    _DevicePythonModule._set_dump_on_cache_miss(True)
    self.assertTrue(_DevicePythonModule._get_dump_on_cache_miss())
    _DevicePythonModule._set_dump_on_cache_miss(False)
    self.assertFalse(_DevicePythonModule._get_dump_on_cache_miss())

  def test_rng_validate_device_index(self):
    # Test valid string and int.
    _device_module._rng_validate_device_index("tpu", 0)
    _device_module._rng_validate_device_index(0, 0)

    # Test valid torch.device.
    _device_module._rng_validate_device_index(
        mock.MagicMock(spec=torch.device, type="tpu", index=0), 0
    )

    # Test invalid device type.
    with self.assertRaisesRegex(
        ValueError, "RNG state can only be accessed on TPU"
    ):
      _device_module._rng_validate_device_index("cpu", 0)

    # Test invalid device index logs warning.
    with mock.patch(
        "torch_tpu.api._device_module.logging.warning"
    ) as mock_warning:
      _device_module._rng_validate_device_index(1, 0)
      mock_warning.assert_called_once()


if __name__ == "__main__":
  absltest.main()
