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

"""Basic single TPU tests for TPU process group backend."""

import datetime

from absl import logging
from absl.testing import absltest
import portpicker
import torch
import torch.accelerator
from torch_tpu._internal.distributed import tpu_distributed


class SingleTpuPGTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # tpu_dist is auto-registered when torch_tpu loads. world_size is passed as
    # an arg (the CUDA/NCCL way), not via env vars.
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://localhost:{portpicker.pick_unused_port()}",
        rank=0,
        world_size=1,
    )

  def test_init_pg(self):
    """A world_size=1 group initializes even when >1 TensorCore is visible.

    On TPU v7 each chip exposes two addressable cores, so a single process sees
    more than one addressable device. A single-rank group must still bind to one
    core (like cuda:0 with several GPUs visible) rather than rejecting init.
    """
    logging.info("TPU device: %s", torch.device("tpu"))
    logging.info("global_device_id: %s", tpu_distributed.global_device_id())
    self.assertTrue(torch.distributed.is_backend_available("tpu_dist"))
    self.assertTrue(torch.distributed.is_initialized())
    self.assertEqual(torch.distributed.get_world_size(), 1)
    self.assertEqual(torch.distributed.get_rank(), 0)

  def test_allreduce_is_identity(self):
    """A collective over a single rank is the identity (NCCL semantics)."""
    x = torch.arange(8, dtype=torch.float32, device="tpu")
    expected = x.clone()
    torch.distributed.all_reduce(x)  # SUM over one rank == identity
    torch.testing.assert_close(x.cpu(), expected.cpu())

  def test_allgather_into_tensor_is_identity(self):
    x = torch.arange(4, dtype=torch.float32, device="tpu") + 1
    out = torch.empty(4, dtype=torch.float32, device="tpu")
    torch.distributed.all_gather_into_tensor(out, x)
    torch.testing.assert_close(out.cpu(), x.cpu())

  def test_manual_pg(self):
    """Test manual/explicit ProcessGroup creation."""

    device = torch.device("tpu")
    logging.info("TPU device: %s", device)

    store = torch.distributed.TCPStore(
        host_name="localhost",
        port=portpicker.pick_unused_port(),
        world_size=1,
        is_master=True,
    )
    pg = tpu_distributed.create_process_group(
        store, 0, 1, datetime.timedelta(seconds=5)
    )
    logging.info("ProcessGroup: %s", pg)
    self.assertEqual(pg._get_backend_name(), "tpu_dist")


if __name__ == "__main__":
  absltest.main()
