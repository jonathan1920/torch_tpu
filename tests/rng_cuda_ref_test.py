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

import functools
import struct
from typing import Any
import unittest

from absl import flags
from absl.testing import absltest
import torch
from tests import oss_utils
from tests import seed_test_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


def _get_backend_module(backend_name: str) -> Any:
  if backend_name == "gpu":
    return torch.cuda
  elif backend_name == "tpu":
    return torch.tpu
  raise ValueError(
      f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
      " 'tpu'"
  )


def _get_device(backend_name: str, device_idx: int) -> torch.device:
  if backend_name == "gpu":
    return torch.device(f"cuda:{device_idx}")
  elif backend_name == "tpu":
    return torch.device(f"tpu:{device_idx}")
  raise ValueError(
      f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
      " 'tpu'"
  )


def _fail_on_tpu(reason: str):
  """Decorator that asserts the test fails on TPU."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.backend == "tpu":
        try:
          func(self, *args, **kwargs)
        except unittest.SkipTest:
          raise
        except Exception:
          # Expected failure on TPU due to backend limitation or missing feature.
          return
        self.fail(
            f"Expected test to fail on TPU ({reason}), but it unexpectedly"
            " succeeded."
        )
      return func(self, *args, **kwargs)

    return wrapper

  return decorator


class CpuRngTest(seed_test_utils.RepeatableTest):
  """Tests default CPU RNG behaviors.

  This is not a reference test comparing TPU vs GPU. However, the CPU RNG
  behaviors verified here are directly relied upon by other test cases to
  validate their test setups. It also ensures that the device backend
  implementation does not introduce unexpected side effects that alter native
  PyTorch CPU RNG behaviors.
  """

  def test_cpu_manual_seed_initializes_state(self):
    """Verifies torch.manual_seed sets the initial CPU seed."""
    torch.manual_seed(1)
    self.assertEqual(torch.initial_seed(), 1)

    torch.manual_seed(42)
    self.assertEqual(torch.initial_seed(), 42)

  def test_cpu_rand_changes_state(self):
    """Verifies generating random numbers mutates the CPU generator state."""
    torch.manual_seed(42)
    initial_cpu_state = torch.get_rng_state()

    _ = torch.rand(100, device="cpu")
    self.assertFalse(torch.equal(torch.get_rng_state(), initial_cpu_state))


class RngCudaRefTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.backend = _BACKEND.value
    self.backend_mod = _get_backend_module(self.backend)
    self.device_idx = self.backend_mod.current_device()
    self.device = _get_device(self.backend, self.device_idx)

  def _unpack_device_rng_uint64(
      self, start_byte: int, end_byte: int, device_idx: int | None = None
  ) -> int:
    """Unpacks a uint64 value from a byte slice of the device RNG state."""
    if device_idx is None:
      device_idx = self.device_idx
    state = self.backend_mod.get_rng_state(device_idx)
    (val,) = struct.unpack("<Q", state[start_byte:end_byte].numpy().tobytes())
    return val

  def _get_device_rng_seed(self, device_idx: int | None = None) -> int:
    """Returns the RNG seed for the specified device.

    Args:
      device_idx: The index of the device to query. Defaults to the current
        device index (self.device_idx) if not specified.
    """
    return self._unpack_device_rng_uint64(0, 8, device_idx)

  def _get_device_rng_offset(self, device_idx: int | None = None) -> int:
    """Returns the RNG offset for the specified device.

    Args:
      device_idx: The index of the device to query. Defaults to the current
        device index (self.device_idx) if not specified.
    """
    return self._unpack_device_rng_uint64(8, 16, device_idx)

  def test_initial_seed_return_current_device_seed(self):
    """Verifies backend_mod.initial_seed returns current device seed."""
    torch.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  def test_manual_seed_sets_current_device_seed(self):
    """Verifies torch.manual_seed sets initial seed on device backend."""
    torch.manual_seed(1)
    self.assertEqual(self.backend_mod.initial_seed(), 1)

    torch.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  # TODO(b/547900660): Remove _fail_on_tpu once querying non-current device RNG state is supported.
  @_fail_on_tpu(
      "TPU backend does not support querying non-current device RNG state."
  )
  # TODO: Enable multi-device execution in OSS via the 'exclusive' tag and remove this skip.
  @oss_utils.skip_in_oss("OSS CI runners isolate tests to a single TPU chip.")
  def test_torch_manual_seed_sets_all_device_seeds(self):
    """Verifies torch.manual_seed seeds all devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding all devices.",
    )
    torch.manual_seed(42)

    for i in range(num_devices):
      self.assertEqual(self._get_device_rng_seed(i), 42)
      self.assertEqual(self._get_device_rng_offset(i), 0)

  def test_torch_manual_seed_different_seeds_produce_different_tensors(self):
    """Verifies different seeds produce distinct random tensors."""
    torch.manual_seed(1)
    t1 = torch.rand(100, device=self.device)

    torch.manual_seed(2)
    t2 = torch.rand(100, device=self.device)

    self.assertFalse(torch.equal(t1, t2))

  def test_torch_manual_seed_reproducibility(self):
    """Verifies setting the same seed produces identical random tensors."""
    torch.manual_seed(42)
    t1 = torch.rand(100, device=self.device)

    torch.manual_seed(42)
    t2 = torch.rand(100, device=self.device)

    self.assertTrue(torch.equal(t1, t2))

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  def test_backend_manual_seed_sets_current_device_seed(self):
    """Verifies backend_mod.manual_seed sets initial seed on current device."""
    self.backend_mod.manual_seed(1)
    self.assertEqual(self.backend_mod.initial_seed(), 1)
    self.assertEqual(self._get_device_rng_seed(), 1)
    self.assertEqual(self._get_device_rng_offset(), 0)

    self.backend_mod.manual_seed(42)
    self.assertEqual(self.backend_mod.initial_seed(), 42)
    self.assertEqual(self._get_device_rng_seed(), 42)
    self.assertEqual(self._get_device_rng_offset(), 0)

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  def test_backend_manual_seed_resets_offset_after_rand(self):
    """Verifies re-seeding with backend_mod.manual_seed resets device offset back to 0."""
    self.backend_mod.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

    self.backend_mod.manual_seed(77)
    self.assertEqual(self._get_device_rng_offset(), 0)

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  def test_backend_manual_seed_different_seeds_produce_different_tensors(self):
    """Verifies different seeds with backend_mod.manual_seed produce distinct tensors."""
    self.backend_mod.manual_seed(1)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed(2)
    t2 = torch.rand(100, device=self.device)

    self.assertFalse(torch.equal(t1, t2))

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  def test_backend_manual_seed_reproducibility(self):
    """Verifies setting the same seed with backend_mod.manual_seed produces identical tensors."""
    self.backend_mod.manual_seed(42)
    t1 = torch.rand(100, device=self.device)

    self.backend_mod.manual_seed(42)
    t2 = torch.rand(100, device=self.device)

    self.assertTrue(torch.equal(t1, t2))

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  def test_backend_manual_seed_does_not_change_cpu_seed(self):
    """Verifies backend_mod.manual_seed does not change CPU seed or state."""
    torch.manual_seed(10)
    cpu_seed_before = torch.initial_seed()
    cpu_state_before = torch.get_rng_state()

    self.backend_mod.manual_seed(42)

    self.assertEqual(torch.initial_seed(), cpu_seed_before)
    self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state_before))

  # TODO(b/548110551): Remove _fail_on_tpu once `torch.tpu.manual_seed` is implemented.
  @_fail_on_tpu("torch.tpu does not implement manual_seed().")
  # TODO: Enable multi-device execution in OSS via the 'exclusive' tag and remove this skip.
  @oss_utils.skip_in_oss("OSS CI runners isolate tests to a single TPU chip.")
  def test_backend_manual_seed_does_not_change_other_devices(self):
    """Verifies backend_mod.manual_seed does not modify other devices."""
    num_devices = self.backend_mod.device_count()
    self.assertGreater(
        num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " seeding behavior across devices.",
    )
    torch.manual_seed(10)
    other_devices_state_before = {
        i: (self._get_device_rng_seed(i), self._get_device_rng_offset(i))
        for i in range(num_devices)
        if i != self.device_idx
    }

    self.backend_mod.manual_seed(42)

    self.assertEqual(self._get_device_rng_seed(self.device_idx), 42)
    self.assertEqual(self._get_device_rng_offset(self.device_idx), 0)
    for i, (seed_before, offset_before) in other_devices_state_before.items():
      self.assertEqual(
          self._get_device_rng_seed(i),
          seed_before,
          msg=f"Device {i} seed mismatch",
      )
      self.assertEqual(
          self._get_device_rng_offset(i),
          offset_before,
          msg=f"Device {i} offset mismatch",
      )

  def test_rand_does_not_change_device_seed(self):
    """Verifies torch.rand on device does not change initial_seed."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertEqual(self.backend_mod.initial_seed(), 42)

  def test_rand_advances_offset(self):
    """Verifies torch.rand on device advances the offset."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    self.assertGreater(self._get_device_rng_offset(), 0)

  def test_manual_seed_resets_offset_after_rand(self):
    """Verifies re-seeding resets device offset back to 0."""
    torch.manual_seed(42)
    _ = torch.rand(100, device=self.device)
    torch.manual_seed(77)
    self.assertEqual(self._get_device_rng_offset(), 0)


if __name__ == "__main__":
  absltest.main()
