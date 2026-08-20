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

"""Unit tests for TorchCommsTPU communicator and backend registration."""

import os
from unittest import mock
from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch_tpu import _loader
from torch_tpu._internal.distributed import torchcomm_tpu
from tests import seed_test_utils

_loader._init_device("tpu")


class TorchCommsTPUTest(seed_test_utils.RepeatableTest):

  def test_communicator_properties(self):
    dev = torch.device("tpu:0")
    comm = torchcomm_tpu.TorchCommsTPU(
        backend_name="tpu",
        device=dev,
        name="test_comm_0",
        group_rank=2,
        group_size=8,
    )

    self.assertEqual(comm.rank, 2)
    self.assertEqual(comm.size, 8)
    self.assertEqual(comm.device, dev)
    self.assertEqual(comm.name, "test_comm_0")

    self.assertEqual(comm.get_rank(), 2)
    self.assertEqual(comm.get_size(), 8)
    self.assertEqual(comm.get_device(), dev)
    self.assertEqual(comm.get_name(), "test_comm_0")
    self.assertFalse(comm.is_finalized())

  def test_env_var_configuration(self):
    env_patch = {
        "TORCHCOMM_RANK": "3",
        "TORCHCOMM_SIZE": "16",
    }
    with mock.patch.dict(os.environ, env_patch):
      comm = torchcomm_tpu.TorchCommsTPU()
      self.assertEqual(comm.rank, 3)
      self.assertEqual(comm.size, 16)

  def test_rank_size_fallback_to_rank_world_size(self):
    env_patch = {
        "RANK": "1",
        "WORLD_SIZE": "4",
    }
    # Clear TORCHCOMM_* if set
    with mock.patch.dict(os.environ, env_patch, clear=False), mock.patch.dict(
        os.environ, {"TORCHCOMM_RANK": "", "TORCHCOMM_SIZE": ""}
    ):
      os.environ.pop("TORCHCOMM_RANK", None)
      os.environ.pop("TORCHCOMM_SIZE", None)
      comm = torchcomm_tpu.TorchCommsTPU()
      self.assertEqual(comm.rank, 1)
      self.assertEqual(comm.size, 4)

  def test_finalize_lifecycle(self):
    comm = torchcomm_tpu.TorchCommsTPU(group_rank=0, group_size=1)
    self.assertFalse(comm.is_finalized())

    comm.finalize()
    self.assertTrue(comm.is_finalized())
    self.assertIsNone(comm.get_backend())

    # Finalize should be idempotent
    comm.finalize()
    self.assertTrue(comm.is_finalized())

    # Operations should raise error after finalize
    dummy_tensor = torch.tensor([1.0])
    with self.assertRaisesRegex(RuntimeError, "finalized"):
      comm.all_reduce(dummy_tensor)

    with self.assertRaisesRegex(RuntimeError, "Cannot split"):
      comm.split(color=1, key=0)

  def test_split_subgroup(self):
    comm = torchcomm_tpu.TorchCommsTPU(
        name="parent_mesh",
        group_rank=0,
        group_size=4,
    )
    sub_comm = comm.split(color=1, key=2)
    self.assertEqual(sub_comm.rank, 2)
    self.assertEqual(sub_comm.size, 4)
    self.assertEqual(sub_comm.name, "parent_mesh_split_1_2")
    self.assertFalse(sub_comm.is_finalized())

  def test_create_torchcomm_tpu_factory(self):
    dev = torch.device("tpu:0")
    comm = torchcomm_tpu.create_torchcomms_tpu(
        backend_str="tpu",
        device=dev,
        name="factory_comm",
        hints={"custom_hint": 123},
    )
    self.assertIsInstance(comm, torchcomm_tpu.TorchCommsTPU)
    self.assertEqual(comm.name, "factory_comm")
    self.assertEqual(comm.device, dev)

  def test_backend_wrapper_compatibility(self):
    class FakeBackendWrapper:

      def __init__(self, comm):
        self._comm = comm

      def get_comm(self):
        return self._comm

    comm = torchcomm_tpu.TorchCommsTPU(group_rank=0, group_size=1)
    wrapper = FakeBackendWrapper(comm)
    self.assertIs(wrapper.get_comm(), comm)

  def test_collectives_delegation(self):
    mock_pg = mock.MagicMock()
    comm = torchcomm_tpu.TorchCommsTPU(group_rank=0, group_size=2, pg=mock_pg)

    self.assertIs(comm.get_backend(), mock_pg)
    self.assertIs(comm.unsafe_get_backend(), mock_pg)

    t = torch.tensor([1.0, 2.0])
    out = torch.empty([2, 2])

    # all_reduce
    comm.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    self.assertTrue(mock_pg.allreduce.called)
    call_args = mock_pg.allreduce.call_args[0]
    self.assertTrue(call_args[1].asyncOp)

    # all_gather
    t_list = [torch.empty_like(t), torch.empty_like(t)]
    comm.all_gather(t_list, t)
    self.assertTrue(mock_pg.allgather.called)

    # all_gather_single & alias
    comm.all_gather_single(out, t)
    self.assertTrue(mock_pg._allgather_base.called)
    comm.all_gather_into_tensor(out, t)

    # reduce_scatter
    comm.reduce_scatter(t, t_list, op=dist.ReduceOp.SUM)
    self.assertTrue(mock_pg.reduce_scatter.called)

    # reduce_scatter_single & alias
    comm.reduce_scatter_single(t, out, op=dist.ReduceOp.SUM)
    self.assertTrue(mock_pg._reduce_scatter_base.called)
    comm.reduce_scatter_tensor(t, out, op=dist.ReduceOp.SUM)

    # broadcast
    comm.broadcast(t, src=0)
    self.assertTrue(mock_pg.broadcast.called)

    # barrier
    comm.barrier()
    self.assertTrue(mock_pg.barrier.called)

    # send / recv
    mock_pg.experimental_send = mock.MagicMock()
    mock_pg.experimental_recv = mock.MagicMock()
    comm.send(t, dst=1, tag=42)
    mock_pg.experimental_send.assert_called_once_with([t], 1, 42)
    comm.recv(t, src=1, tag=42)
    mock_pg.experimental_recv.assert_called_once_with([t], 1, 42)

    # all_to_all
    comm.all_to_all(t_list, t_list)
    self.assertTrue(mock_pg.alltoall.called)

    # all_to_all_single
    comm.all_to_all_single(out, out, [1, 1], [1, 1])
    self.assertTrue(mock_pg.alltoall_base.called)

  def test_registration_with_torchcomms(self):
    fake_torchcomms = mock.MagicMock()
    registered_backends = {}

    def fake_register(name, creator, devices):
      registered_backends[name] = (creator, devices)

    fake_torchcomms.register_backend = fake_register
    fake_torchcomms.is_backend_registered = lambda n: n in registered_backends

    with mock.patch.dict("sys.modules", {"torchcomms": fake_torchcomms}):
      registered = torchcomm_tpu.register_torchcomms_tpu()
      self.assertTrue(registered)
      self.assertIn("tpu", registered_backends)
      self.assertIn("tpu_dist", registered_backends)
      self.assertEqual(registered_backends["tpu"][1], ["tpu"])

      # Test new_comm creation via registered creator
      creator, _ = registered_backends["tpu"]
      comm = creator("tpu", torch.device("tpu:0"), name="tc_tpu")
      self.assertIsInstance(comm, torchcomm_tpu.TorchCommTPU)

  def test_torchcomms_handles_backend_routing(self):
    # Verify that PyTorch c10d _torchcomms_handles_backend recognizes tpu
    with mock.patch.multiple(
        c10d,
        _TORCHCOMM_AVAILABLE=True,
        _torchcomms_is_backend_registered=lambda name: name
        in ("tpu", "tpu_dist"),
        _torchcomms_is_backend_built=lambda name: False,
        create=True,
    ):
      self.assertTrue(c10d._torchcomms_handles_backend("tpu"))
      self.assertTrue(c10d._torchcomms_handles_backend("tpu_dist"))
      self.assertTrue(c10d._torchcomms_handles_backend("TPU"))
      self.assertFalse(c10d._torchcomms_handles_backend("unknown_backend"))

  def test_c10d_backend_availability(self):
    mock_distributed = mock.MagicMock()
    with mock.patch(
        "torch_tpu._internal.distributed.torchcomm_tpu.tpu_distributed",
        mock_distributed,
    ):
      torchcomm_tpu.register_torchcomms_tpu()
      self.assertTrue(dist.is_backend_available("tpu_dist"))
      self.assertTrue(dist.is_backend_available("tpu"))
      self.assertEqual(
          dist.Backend.default_device_backend_map.get("tpu"), "tpu"
      )

  def test_torchcomms_new_comm_standard_initialization(self):
    fake_torchcomms = mock.MagicMock()
    registered_backends = {}

    def fake_register(name, creator, devices):
      registered_backends[name] = (creator, devices)

    def fake_new_comm(backend_str, *args, **kwargs):
      if backend_str not in registered_backends:
        raise ValueError(f"Backend {backend_str} not registered")
      creator, _ = registered_backends[backend_str]
      return creator(backend_str, *args, **kwargs)

    fake_torchcomms.register_backend = fake_register
    fake_torchcomms.is_backend_registered = lambda n: n in registered_backends
    fake_torchcomms.new_comm = fake_new_comm

    with mock.patch.dict("sys.modules", {"torchcomms": fake_torchcomms}):
      torchcomm_tpu.register_torchcomms_tpu()
      # Call standard torchcomms.new_comm entry point
      comm = fake_torchcomms.new_comm(
          "tpu",
          device=torch.device("tpu:0"),
          name="std_tpu_comm",
      )
      self.assertIsInstance(comm, torchcomm_tpu.TorchCommTPU)
      self.assertEqual(comm.name, "std_tpu_comm")
      self.assertEqual(comm.device, torch.device("tpu:0"))

  def test_is_torchcomms_registered(self):
    # 1. When torchcomms is not available
    with mock.patch.dict("sys.modules", {"torchcomms": None}):
      self.assertFalse(torchcomm_tpu.is_torchcomms_registered("tpu"))

    # 2. When torchcomms has _is_backend_registered
    fake_torchcomms_1 = mock.MagicMock(spec=["_is_backend_registered"])
    fake_torchcomms_1._is_backend_registered = lambda backend: backend in (
        "tpu",
        "tpu_dist",
    )
    with mock.patch.dict("sys.modules", {"torchcomms": fake_torchcomms_1}):
      self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu"))
      self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu_dist"))
      self.assertFalse(torchcomm_tpu.is_torchcomms_registered("nccl"))

    # 3. When torchcomms has is_backend_registered
    fake_torchcomms_2 = mock.MagicMock(spec=["is_backend_registered"])
    fake_torchcomms_2.is_backend_registered = lambda backend: backend == "tpu"
    with mock.patch.dict("sys.modules", {"torchcomms": fake_torchcomms_2}):
      self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu"))
      self.assertFalse(torchcomm_tpu.is_torchcomms_registered("cuda"))

    # 4. When torchcomms has is_backend_built
    fake_torchcomms_3 = mock.MagicMock(spec=["is_backend_built"])
    fake_torchcomms_3.is_backend_built = lambda backend: backend == "tpu"
    with mock.patch.dict("sys.modules", {"torchcomms": fake_torchcomms_3}):
      self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu"))
      self.assertFalse(torchcomm_tpu.is_torchcomms_registered("gloo"))

  def test_unsupported_stubs_raise_not_implemented(self):
    mock_pg = mock.MagicMock()
    comm = torchcomm_tpu.TorchCommTPU(
        backend_name="tpu",
        device=torch.device("tpu:0"),
        pg=mock_pg,
        group_rank=0,
        group_size=2,
    )
    t = torch.zeros(4, device="cpu")

    self.assertEqual(comm.backend_name, "tpu")
    self.assertEqual(comm.get_backend_name(), "tpu")
    self.assertEqual(comm.device_type, "tpu")
    self.assertEqual(comm.hints, {})
    self.assertIsNone(comm.store)
    self.assertIsNotNone(comm.timeout)

    with self.assertRaises(NotImplementedError):
      comm.create_pair_comm(1)

    with self.assertRaises(NotImplementedError):
      comm.split_group([0, 1])

    with self.assertRaises(NotImplementedError):
      comm.reconfigure(ranks=[0, 1], size=2)

    with self.assertRaises(NotImplementedError):
      comm.create_window(t)

    with self.assertRaises(NotImplementedError):
      comm.map_remote_tensor(t, peer_rank=1)

    with self.assertRaises(NotImplementedError):
      comm.create_batch()

    with self.assertRaises(NotImplementedError):
      comm.all_reduce_coalesced([t])

    with self.assertRaises(NotImplementedError):
      comm.all_gather_coalesced([[t]], [t])

    with self.assertRaises(NotImplementedError):
      comm.reduce_scatter_coalesced([t], [[t]])

    with self.assertRaises(NotImplementedError):
      comm.all_to_all_coalesced([[t]], [[t]])

    with self.assertRaises(NotImplementedError):
      comm.reduce(t, dst=0)

    with self.assertRaises(NotImplementedError):
      comm.gather(input_tensor=t, dst=0)

    with self.assertRaises(NotImplementedError):
      comm.scatter(output_tensor=t, src=0)

    with self.assertRaises(NotImplementedError):
      comm.monitored_barrier()

    with self.assertRaises(NotImplementedError):
      comm.register_flight_recorder_hook(mock.MagicMock())

    with self.assertRaises(NotImplementedError):
      comm.get_flight_recorder_traces()

    with self.assertRaises(NotImplementedError):
      comm.broadcast_object_list([1, 2, 3])

    with self.assertRaises(NotImplementedError):
      comm.all_gather_object([], {"a": 1})

    with self.assertRaises(NotImplementedError):
      comm.gather_object({"a": 1})

    with self.assertRaises(NotImplementedError):
      comm.scatter_object_list([])

  def test_context_manager_finalization(self):
    mock_pg = mock.MagicMock()
    comm = torchcomm_tpu.TorchCommsTPU(
        backend_name="tpu",
        device=torch.device("tpu:0"),
        pg=mock_pg,
        group_rank=0,
        group_size=2,
    )
    with comm as c:
      self.assertIs(c, comm)
      self.assertFalse(c.is_finalized())
    self.assertTrue(comm.is_finalized())


# Alias for backward compatibility
TorchCommTPUTest = TorchCommsTPUTest

if __name__ == "__main__":
  absltest.main()
