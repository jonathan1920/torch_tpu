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

# A script to run the distributed example.

export TORCH_TPU_TOPOLOGY="2,4,1"
export MASTER_ADDR="localhost"
export MASTER_PORT=12355

torchrun --nproc_per_node=8 ./examples/tutorials/distributed/hello_world_distributed_test.py
