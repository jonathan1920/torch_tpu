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

"""Distributed testing utilities for TorchTPU."""

import os
from typing import Any, Callable

import portpicker
import torch.multiprocessing as mp


def _worker_fn(
    rank: int,
    world_size: int,
    master_port: int,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
  """Worker function for mp.spawn."""
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

  fn(*args, **kwargs)


def dist_run(fn: Callable[..., Any], nproc_per_node: int) -> Callable[..., Any]:
  """A programmatic API to launch distributed tests using mp.spawn.

  Args:
    fn: The function to be executed by each distributed worker.
    nproc_per_node: The number of processes to spawn on the current node.

  Returns:
    A wrapped function that, when called, launches the distributed processes.
  """

  def _run_distributed(*args: Any, **kwargs: Any) -> Any:
    master_port = portpicker.pick_unused_port()
    mp.spawn(
        _worker_fn,
        args=(nproc_per_node, master_port, fn, args, kwargs),
        nprocs=nproc_per_node,
        join=True,
    )
    return None

  return _run_distributed
