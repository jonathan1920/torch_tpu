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

"""Device Python Module.

This module provides a Python interface to initialize and shutdown the device
and runtime.
"""

import atexit
from typing import Any, List

from absl import logging
import torch
import torch_tpu._internal.precision as _precision_module
from torch_tpu.api import _device_ops_backend
from torch_tpu.api import streams


# TODO: add more dtypes here. Initial list was chosen arbitrarily.
_AMP_SUPPORTED_DTYPES = (
    # go/keep-sorted start
    torch.bfloat16,
    torch.bool,
    torch.complex128,
    torch.complex64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.uint8,
    # go/keep-sorted end
)


def _rng_validate_device_index(
    device: int | str | torch.device, device_idx: int
) -> None:
  """Validates that the provided device is this process's TPU.

  Args:
    device: The device to validate. Can be an int (device index), "tpu", or a
      `torch.device("tpu")` object.
    device_idx: The index of the current process's TPU device.

  Raises:
    ValueError: If the device is not a TPU device or if the index does not
      match the current device index.
  """
  provided_str, provided_idx = None, -1
  if isinstance(device, str):
    provided_str = device
  elif isinstance(device, torch.device):
    provided_str = device.type
    if provided_idx is not None:
      provided_idx = device.index
  elif isinstance(device, int):
    provided_idx = device
  else:
    raise TypeError(f"Got unrecognized device type, {device}")

  if provided_str is not None and provided_str != "tpu":
    raise ValueError(
        f"RNG state can only be accessed on TPU device, got {provided_str}"
    )
  if provided_idx is not None and provided_idx != device_idx:
    # TODO(b/496180326): allow cross-device RNG access.
    logging.warning(
        "Accessing RNG state of another process's TPU is not supported"
    )


class _DeviceModule:
  """torch_tpu device equivalent to functions in "torch/cuda/__init__.py".

  https://github.com/pytorch/pytorch/blob/v2.7.0/torch/cuda/__init__.py

  For Automatic Mixed Presicion (AMP), see:
  https://docs.pytorch.org/docs/stable/amp.html

  Since methods of this class are accessible via getattr(torch, "tpu"), we need
  to make sure they are not accidentally exposed to the public API by adding a
  leading underscore to the names of internal methods.
  """

  # Invariant: if _is_initialized is True, then _device_count and
  # _current_device must not be None. Otherwise they must be None.
  _is_initialized: bool = False
  _device_count: int | None = None
  _current_device: int | None = None
  _autocast_enabled: bool = False
  _autocast_dtype: torch.dtype | None = torch.float16

  Precision = _precision_module.Precision  # pylint: disable=invalid-name
  precision = _precision_module.precision

  @classmethod
  def current_device(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the index of the currently selected device."""
    if not cls._is_initialized:
      raise RuntimeError(
          "torch.tpu.current_device() not available because the TPU runtime is"
          " not initialized. Please initialize it by calling"
          " torch_tpu.api.tpu_device()."
      )
    assert (
        cls._current_device is not None
    ), "_DeviceModule is initialized but has no current device."
    return cls._current_device

  @classmethod
  def is_available(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns if the TPU backend is currently available."""
    return True

  @classmethod
  def is_initialized(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns whether PyTorch's TPU state has been initialized."""
    return cls._is_initialized

  @classmethod
  def _is_in_bad_fork(cls) -> bool:
    """Returns whether we are in a bad fork.

    This is a CUDA specific concept, related to how the CUDA context and runtime
    interacts with multiprocesses. An implementation is required to support
    torch.manual_seed(), as it is called in _seed_custom_device() in
    py/torch/random.py.
    """
    # TODO(b/456545545): unda - revisit this.
    return False

  @classmethod
  def manual_seed_all(cls, seed: int) -> None:
    """Sets the seed for generating random numbers on all devices."""
    _device_ops_backend.manual_seed_all(seed)

  @classmethod
  def get_rng_state(
      cls, device: int | str | torch.device = "tpu"
  ) -> torch.Tensor:
    _rng_validate_device_index(device, cls.current_device())
    return _device_ops_backend.get_rng_state(-1)

  @classmethod
  def set_rng_state(
      cls, new_state: torch.Tensor, device: int | str | torch.device = "tpu"
  ) -> None:
    _rng_validate_device_index(device, cls.current_device())
    _device_ops_backend.set_rng_state(new_state, -1)

  @classmethod
  def _init_runtime(
      cls,
      device_type="tpu",
  ):
    """Initializes the TPU runtime."""
    if not cls._is_initialized:
      init_result = _device_ops_backend._init_runtime(device_type)  # pylint: disable=protected-access
      cls._is_initialized = True
      cls._device_count = init_result.device_count
      cls._current_device = init_result.device_id
      if not hasattr(cls, "_atexit_registered"):
        atexit.register(cls._shutdown_runtime)
        cls._atexit_registered = True

  @classmethod
  def synchronize(cls, device: int | None = None) -> None:
    """Waits for all kernels in all streams on a TPU device to complete."""
    streams.synchronize(device)

  @classmethod
  def device_count(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the number of TPU devices."""
    if (maybe_count := cls._device_count) is None:
      raise RuntimeError(
          "torch.tpu.device_count() not available because PJRT not initialized."
          " Please initialize the TPU runtime by calling"
          " torch_tpu.api.tpu_device()."
      )
    return maybe_count

  @classmethod
  def stream(cls, stream: streams.TpuStream | None) -> "TpuStreamContext":
    """Wrapper for a context manager that selects a given TPUStream."""
    return TpuStreamContext(stream)

  @classmethod
  def Stream(cls, device=None, priority=0, **kwargs):  # pylint: disable=invalid-name
    """Device-level wrapper for TpuStream object."""
    return streams.TpuStream(device=device, priority=priority, **kwargs)

  @classmethod
  def Event(  # pylint: disable=invalid-name
      cls,
      enable_timing: bool = False,
      blocking: bool = False,
      interprocess: bool = False,
      external: bool = False,
  ) -> streams.TpuEvent:
    """Device-level wrapper for TpuEvent object."""
    return streams.TpuEvent(
        enable_timing=enable_timing,
        blocking=blocking,
        interprocess=interprocess,
        external=external,
    )

  @classmethod
  def set_stream(cls, stream: streams.TpuStream) -> None:  # pylint: disable=unused-argument
    """Sets the current stream. Currently a dummy method (b/452051142)."""
    return

  @classmethod
  def current_stream(
      cls, device: torch.device | None = None  # pylint: disable=unused-argument
  ) -> streams.TpuStream:  # pylint: disable=g-doc-args
    """Returns the active stream for a given device.

    Currently a dummy method (b/452051142).
    """
    return streams.TpuStream()

  @classmethod
  def default_stream(
      cls, device: torch.device | None = None  # pylint: disable=unused-argument
  ) -> streams.TpuStream:  # pylint: disable=g-doc-args
    """Returns the default stream for a given device.

    Currently a dummy method (b/452051142).
    """
    return streams.TpuStream()

  @classmethod
  def get_amp_supported_dtype(cls) -> List[torch.dtype]:  # Needed for AMP.
    return list(_AMP_SUPPORTED_DTYPES)

  @classmethod
  def _is_autocast_enabled(cls) -> bool:
    return cls._autocast_enabled

  @classmethod
  def _get_autocast_dtype(cls) -> torch.dtype | None:
    return cls._autocast_dtype

  @classmethod
  def _set_autocast_enabled(cls, enabled: bool) -> None:
    cls._autocast_enabled = enabled

  @classmethod
  def _set_autocast_dtype(cls, dtype: torch.dtype) -> None:
    if dtype not in cls.get_amp_supported_dtype():
      raise ValueError(
          f"Unsupported dtype {dtype} for AMP. Supported dtypes are:"
          f" {cls.get_amp_supported_dtype()}"
      )
    cls._autocast_dtype = dtype

  @classmethod
  def _set_allow_cache(cls, allow: bool = True):
    _device_ops_backend._set_allow_cache(allow)  # pylint: disable=protected-access

  @classmethod
  def _set_cache_only(cls, cache_only: bool = True):
    _device_ops_backend._set_cache_only(cache_only)  # pylint: disable=protected-access

  @classmethod
  def _get_cache_requests(cls):
    return _device_ops_backend._get_cache_requests()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_hits(cls):
    return _device_ops_backend._get_cache_hits()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_misses(cls):
    return _device_ops_backend._get_cache_misses()  # pylint: disable=protected-access

  @classmethod
  def _clear_cache(cls):
    _device_ops_backend._clear_cache()  # pylint: disable=protected-access

  @classmethod
  def _hbm_usage_summary(cls):
    return _device_ops_backend._hbm_usage_summary()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_stats(cls):
    return _device_ops_backend._get_cache_stats()  # pylint: disable=protected-access

  @classmethod
  def _shutdown_runtime(cls):
    if cls._is_initialized:
      _device_ops_backend._shutdown_runtime()  # pylint: disable=protected-access
      cls._is_initialized = False


class TpuStreamContext:
  """Operations within this context manager are enqueued on the given stream.

  Currently non-functional. See b/452051142.
  """

  def __init__(self, stream: streams.TpuStream | None):
    self.stream = stream
    self.prev_stream = None

  def __enter__(self):
    if self.stream is None:
      return
    self.prev_stream = torch.tpu.current_stream()
    torch.tpu.set_stream(self.stream)

  def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any):
    if self.stream is None:
      return
    torch.tpu.set_stream(self.prev_stream)
