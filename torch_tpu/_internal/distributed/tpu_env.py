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

"""TPU-specific boilerplate we need to run on multiple TPUs.

TODO(vladbelous): We should simplify this and make it work on Cloud TPUs,
before open-sourcing.
"""

import os
import time
from typing import Any, Callable, Tuple

from absl import logging
import portpicker
import torch
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed import tpu_topology


class WorldSpec:
  """Specification of the world (i.e. the set of worker processes).

  All workers agree on this specification. An instance of this class is created
  by the main process and sent to each of the worker processes.
  """

  def __init__(self, world_size: int):
    # All workers must agree on the world size and which ports they'll use
    # to communicate.
    # TODO(b/469558756): Reconcile how we pass around world size.
    self.world_size = world_size
    self.c10d_store_port = portpicker.pick_unused_port()

    # Get open ports on this host for slicebuilder.
    self.sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    self.sb_addresses = ",".join([f"localhost:{p}" for p in self.sb_ports])

    # Detect TPU topology.
    self.topology = tpu_topology.get_tpu_topology(world_size)


def init_worker(
    rank: int, world_spec: WorldSpec, xprof_session_id: str
) -> None:
  """Initializes the current process as a worker.

  This must be run in each worker process. It simulates how a real
  TorchTPU user would start multiple PyTorch trainer processes. In reality
  the details would vary depending on the systems being used to create
  and manage multiple processes.

  Args:
    rank: The rank of this worker process.
    world_spec: The specification of the world.
    xprof_session_id: The XProf session ID for this run.
  """

  os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
  # Set the logging output directory to what the TEST_UNDECLARED_OUTPUTS_DIR
  # env var points to. This makes the logs show up in Sponge.
  logging.get_absl_handler().start_logging_to_file(
      program_name="torch_tpu",
      log_dir=os.getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
  )

  logging.info(
      "init_worker: rank=%d, world_size=%d", rank, world_spec.world_size
  )

  # Initialize TPU and get global device id (for this rank).
  _ = api.tpu_device()
  logging.info("device debug string: %s", tpu_distributed.device_debug_string())
  logging.info("global device id: %s", tpu_distributed.global_device_id())
  logging.info(
      "all global device ids: %s", tpu_distributed.all_global_device_ids()
  )

  # Initialize a default distributed process group.
  torch.distributed.init_process_group(
      # Tell PyTorch to use tpu_distributed.create_process_group to create
      # the process group.
      backend="tpu_dist",
      init_method=f"tcp://localhost:{world_spec.c10d_store_port}",
      rank=rank,
      world_size=world_spec.world_size,
  )
  assert torch.distributed.is_initialized()


def init_worker_and_run(
    rank: int,
    world_spec: WorldSpec,
    worker_func: Callable[..., Any],
    extra_worker_func_args: Tuple[Any, ...],
    xprof_session_id: str,
) -> None:
  """Initializes the current process as a worker and runs the given function.

  This must be run in each worker process.

  Args:
    rank: The rank of this worker process.
    world_spec: The specification of the world.
    worker_func: A function that will be run in this worker process. It expects
      required arguments (rank and world_size) and the optional extra arguments
      passed as a tuple in `extra_worker_func_args`.
    extra_worker_func_args: Additional arguments to pass to the worker function.
      These are passed as a tuple to the worker function after the required rank
      and world_size arguments.
    xprof_session_id: The uint64 XProf session ID to use for this run. We are
      using the same session ID for all workers. This should be nanoseconds
      since epoch.
  """

  # Set environment variables typically set by torchrun and other launchers.
  # This is necessary for proper initialization of the PJRT client.
  if "WORLD_SIZE" not in os.environ:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_spec.world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_spec.world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(world_spec.c10d_store_port)

    # Set slicebuilder and topology environment variables.
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = world_spec.sb_addresses
    if world_spec.topology:
      os.environ["TORCH_TPU_TOPOLOGY"] = world_spec.topology

  init_worker(rank, world_spec, xprof_session_id)
  worker_func(rank, world_spec.world_size, *extra_worker_func_args)


def run_in_workers(
    world_size: int, func: Callable[..., Any], worker_args: Tuple[Any, ...] = ()
) -> None:
  """Spawns worker processes for all ranks to run the given function.

  This must be run in the main process.

  Args:
    world_size: The number of worker processes to spawn.
    func: A function that will be run in each worker process. It should take two
      arguments (rank and world_size), plus any arguments in worker_args.
    worker_args: Additional arguments to pass to the worker function. These are
      passed as a tuple to the worker function.
  """
  logging.warning(
      "tpu_env: run_in_workers is being deprecated, please migrate to"
      " singlehost_wrapper and torchrun for distributed runs"
  )

  xprof_session_id = str(time.time_ns())
  logging.info("TORCH_TPU_XPROF_SESSION_ID: %s", xprof_session_id)
  mp.spawn(
      init_worker_and_run,
      args=(
          WorldSpec(world_size),
          func,
          worker_args,
          xprof_session_id,
      ),
      nprocs=world_size,
  )
