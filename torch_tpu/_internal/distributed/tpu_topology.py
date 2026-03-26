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

from absl import logging


_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TPU_PCI_DEVICE_IDS_TO_TOPOLOGY = {
    # tpu v5p
    "0x0062": {
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
        8: "2,4,1",
    },
    # tpu v7
    "0x0076": {
        2: "1,1,1,2",
        4: "1,2,1,2",
        8: "2,2,1,2",
    },
}


def get_tpu_topology(
    world_size: int | None = None,
) -> tuple[str | None, int]:
  """Returns the topology string and device count for attached TPU devices.

  Args:
    world_size: If provided, use this as the target device count for topology
      lookup instead of the auto-detected count. Allows sub-slicing only on
      single-host, in supported topologies.

  Raises:
    RuntimeError: If a TPU device is detected but no topology is found for the
      `target_count` (either `world_size` or the auto-detected count).
  """
  detected_count = 0
  topology_map = None

  for vendor_path in glob.glob("/sys/bus/pci/devices/*/vendor"):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_dir = os.path.dirname(vendor_path)
    device_id = (
        pathlib.Path(os.path.join(device_dir, "device")).read_text().strip()
    )

    if device_id in _TPU_PCI_DEVICE_IDS_TO_TOPOLOGY:
      # Some devices have multiple tensorcores sharing the same bus.
      # We count only those that are exposed via VFIO.
      iommu_group_path = os.path.join(device_dir, "iommu_group")
      if os.path.exists(iommu_group_path):
        group_id = os.path.basename(os.readlink(iommu_group_path))
        if not os.path.exists(f"/dev/vfio/{group_id}"):
          continue

      detected_count += 1
      topology_map = _TPU_PCI_DEVICE_IDS_TO_TOPOLOGY[device_id]

  if world_size is not None:
    if world_size > detected_count:
      raise ValueError(
          "expected world_size to be less than or equal to detected TPUs"
          f" {detected_count}, got {world_size}"
      )
    elif world_size < detected_count:
      logging.warning(
          "using world_size: %d instead of detected count: %d",
          world_size,
          detected_count,
      )

  target_count = world_size if world_size is not None else detected_count

  if target_count == 0:
    return None, 0

  if topology_map:
    if target_count not in topology_map:
      raise RuntimeError(f"No TPU topology found for count: {target_count}")
    return topology_map[target_count], target_count

  return None, 0
