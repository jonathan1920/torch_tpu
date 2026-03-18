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

"""Common utilities for data parallel examples."""

import torch
from torch import nn
from torch import optim
from examples.distributed.data_parallel import simple_model


def make_model_and_data(
    batch_size: int,
    in_features: int,
    out_features: int,
    random_seed: int = 0,
) -> tuple[simple_model.SimpleNN, torch.Tensor, torch.Tensor]:
  """Deterministically initializes model and training data.

  Returns:
    Initialized model, input data (X), target data (y).

  Args:
    batch_size: the number of examples in each batch
    in_features: the number of input features
    out_features: the number of output features
    random_seed: for reproducibility of data and weights
  """
  torch.manual_seed(random_seed)
  model = simple_model.SimpleNN(
      in_features=in_features,
      out_features=out_features,
  )
  x = torch.randn(batch_size, in_features)
  y = torch.randn(batch_size, out_features)
  return model, x, y


def run_cpu_model_training_step(
    batch_size: int,
    in_features: int,
    out_features: int,
    lr: float,
    random_seed: int = 0,
) -> simple_model.SimpleNN:
  """Deterministically creates a model and runs one training step on CPU."""
  model, data, target = make_model_and_data(
      batch_size=batch_size,
      in_features=in_features,
      out_features=out_features,
      random_seed=random_seed,
  )
  optimizer = optim.SGD(model.parameters(), lr=lr)
  run_training_step(model=model, data=data, target=target, optimizer=optimizer)
  return model


def run_training_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module = nn.MSELoss(),
) -> None:
  optimizer.zero_grad()
  output = model(data)
  loss = loss_fn(output, target)
  loss.backward()
  optimizer.step()
