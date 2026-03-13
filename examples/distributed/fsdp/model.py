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

"""Two layer model to demonstrate sharded data parallelism (FSDP)."""

from torch import nn


class SimpleNN(nn.Module):
  """Two linear layers and a nonlinear activation."""

  def __init__(self, in_features, out_features, bias=True):
    super().__init__()
    self.linear1 = nn.Linear(in_features, in_features, bias=bias)
    self.linear2 = nn.Linear(in_features, out_features, bias=bias)
    self.silu = nn.SiLU()

  def forward(self, x):
    x = self.linear1(x)
    x = self.silu(x)
    x = self.linear2(x)
    return x
