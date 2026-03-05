#!/bin/bash
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

# A script to run the tensor parallel worker using standard torchrun.

# This is an example config for a v7x-8 machine.
export TORCH_TPU_TOPOLOGY="2,2,1,2"
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:50000,localhost:50001,localhost:50002,localhost:50003,localhost:50004,localhost:50005,localhost:50006,localhost:50007"
export MASTER_ADDR="localhost"
export MASTER_PORT=12355

python3 -m torch.distributed.run --nproc_per_node=8 ./tp_worker.py
