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
import torch


def available_xla_device() -> torch.device | None:
  """Returns the initialized XLA device (xla_cpu, xla_cuda, or tpu) or None."""
  if hasattr(torch, "tpu"):
    return torch.tpu.current_device()
  if hasattr(torch, "xla_cpu"):
    return torch.xla_cpu.current_device()
  if hasattr(torch, "xla_cuda"):
    return torch.xla_cuda.current_device()
  return None


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "available_xla_device",
    # go/keep-sorted end
]
