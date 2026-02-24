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

"""Utilities for discovering TPU hardware topology.

This is for reading from the pcie devs, and then from that, determining the
generation of TPU that we have (and that we have a TPU at all). If the user
decides to call the wrapper to generate our two additional environment
variables, then we use this to decode their topology given the generation of the
device and the world size.
"""

import glob
import os
import pathlib
from typing import Optional

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TPU_PCI_DEVICE_IDS_TO_TOPOLOGY = {
    # tpu v5p
    "0x005e": {
        4: "2,2,1",
        8: "2,2,2",
    },
    # tpu v5e
    "0x0063": {
        4: "2,2,1",
        8: "2,2,2",
    },
    # tpu v6e
    "0x006f": {
        4: "2,2,1",
        8: "2,2,2",
    },
    # tpu v7
    "0x0076": {
        4: "1,2,1,2",
        8: "2,2,1,2",
    },
}


def get_tpu_topology(world_size: Optional[int] = None):
  """Returns the TPU topology string for an attached TPU device, or None.

  Args:
    world_size: The number of TPU devices expected. If None, it will be inferred
      from the number of TPU devices found.
  """
  count = 0
  topology_map = None

  for vendor_path in glob.glob("/sys/bus/pci/devices/*/vendor"):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), "device")
    device_id = pathlib.Path(device_path).read_text().strip()

    if device_id in _TPU_PCI_DEVICE_IDS_TO_TOPOLOGY:
      count += 1
      topology_map = _TPU_PCI_DEVICE_IDS_TO_TOPOLOGY[device_id]

  if topology_map:
    target_size = world_size if world_size else count
    return topology_map.get(target_size)

  return None
