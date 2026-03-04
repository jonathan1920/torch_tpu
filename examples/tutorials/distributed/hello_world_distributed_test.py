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

"""Tests torch_tpu distributed example.

NOTE: You should not be running this script directly. Instead, run via:
  - `run_script.sh` or
  - `run_script.py` if using Bazel.
"""

import os
import torch
from torch import distributed as dist
from torch_tpu import api
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils


def run_all_reduce():
  """Tests all-reduce functionality."""
  _ = api.tpu_device()
  ws = int(os.environ["WORLD_SIZE"])
  dist.init_process_group(backend="tpu_dist", world_size=ws)

  x = torch.tensor([0.0, 1.0, ws], device="tpu")
  torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM)

  utils.assert_close(x.cpu(), torch.tensor([0.0, ws, ws * ws]))
  print("result: ", x)


if __name__ == "__main__":
  singlehost_wrapper.prepare_tpu_environment()
  run_all_reduce()
