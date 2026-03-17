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

"""Shared layer configurations for benchmarks."""

import dataclasses


@dataclasses.dataclass
class LinearConfig:
  batch_size: int
  seq_len: int
  in_features: int
  out_features: int


LINEAR_CONFIGS = (
    # Default config for smoke test.
    LinearConfig(
        batch_size=1,
        seq_len=128,
        in_features=128,
        out_features=128,
    ),
    # Configs for Llama3 70B for MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=8192,
        out_features=28672,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=28672,
        out_features=8192,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=8192,
        out_features=28672,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=28672,
        out_features=8192,
    ),
    # Configs for Qwen3 480B MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=6144,
        out_features=2560,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=2560,
        out_features=6144,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=6144,
        out_features=2560,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=2560,
        out_features=6144,
    ),
    # Configs for Gemma3 27B MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=4608,
        out_features=36864,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=36864,
        out_features=4608,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=4608,
        out_features=36864,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=36864,
        out_features=4608,
    ),
    # Configs for BERT
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=768,
        out_features=768,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=768,
        out_features=3072,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=3072,
        out_features=768,
    ),
)


@dataclasses.dataclass
class BatchNormConfig:
  batch_size: int
  seq_len: int
  num_features: int


BATCH_NORM_CONFIGS = (
    # Default config for smoke test.
    BatchNormConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    # Larger configs.
    BatchNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
    # High-batch, more TPU friendly shape.
    BatchNormConfig(
        batch_size=2056,
        seq_len=512,
        num_features=1024,
    ),
)


@dataclasses.dataclass
class LayerNormConfig:
  batch_size: int
  seq_len: int
  num_features: int
  num_normalized_dims: int = 1

  @property
  def shape(self):
    return (self.batch_size, self.seq_len, self.num_features)

  @property
  def normalized_shape(self):
    return self.shape[-self.num_normalized_dims :]

  @property
  def reduction_axes(self):
    return range(len(self.shape) - self.num_normalized_dims, len(self.shape))


LAYER_NORM_CONFIGS = (
    # Default config for smoke test.
    LayerNormConfig(batch_size=1, seq_len=128, num_features=128),
    # Larger configs.
    LayerNormConfig(batch_size=32, seq_len=8192, num_features=8192),
    LayerNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
        num_normalized_dims=2,
    ),
    # BERT configs
    LayerNormConfig(
        batch_size=32,
        seq_len=128,
        num_features=768,
    ),
)


@dataclasses.dataclass
class Conv2dConfig:
  batch_size: int
  in_channels: int
  out_channels: int
  kernel_size: int
  stride: int
  padding: int
  height: int
  width: int


CONV2D_CONFIGS = (
    # Default config for smoke test.
    Conv2dConfig(
        batch_size=1,
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        height=128,
        width=128,
    ),
    # Larger configs.
    Conv2dConfig(
        batch_size=128,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        height=256,
        width=256,
    ),
)


@dataclasses.dataclass
class RmsNormConfig:
  batch_size: int
  seq_len: int
  num_features: int


RMS_NORM_CONFIGS = (
    # Default config for smoke test.
    RmsNormConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    # Configs for Llama3 70B for MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=8192,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
    # Configs for Qwen3 480B MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=6144,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=6144,
    ),
    # Configs for Gemma3 27B MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=4608,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=4608,
    ),
)


@dataclasses.dataclass(frozen=True)
class SdpaConfig:
  batch_size: int
  seq_len: int
  num_heads: int
  head_dim: int


SDPA_CONFIGS = (
    # Default config for smoke test.
    SdpaConfig(
        batch_size=1,
        seq_len=128,
        num_heads=8,
        head_dim=64,
    ),
    # Configs for Llama3 70B attention layers
    SdpaConfig(
        batch_size=1,
        seq_len=2048,
        num_heads=64,
        head_dim=128,
    ),
    SdpaConfig(
        batch_size=4,
        seq_len=2048,
        num_heads=64,
        head_dim=128,
    ),
    # Configs for Qwen3 480B attention layers
    SdpaConfig(
        batch_size=1,
        seq_len=2048,
        num_heads=96,
        head_dim=128,
    ),
    SdpaConfig(
        batch_size=4,
        seq_len=2048,
        num_heads=96,
        head_dim=128,
    ),
    # Configs for Gemma3 27B attention layers
    SdpaConfig(
        batch_size=1,
        seq_len=2048,
        num_heads=32,
        head_dim=128,
    ),
    SdpaConfig(
        batch_size=4,
        seq_len=2048,
        num_heads=32,
        head_dim=128,
    ),
)


@dataclasses.dataclass
class TopKConfig:
  batch_size: int
  seq_len: int
  num_features: int
  k: int
  dim: int


TOPK_CONFIGS = (
    TopKConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
        k=10,
        dim=-1,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
        k=100,
        dim=-1,
    ),
)


@dataclasses.dataclass
class NonZeroConfig:
  batch_size: int
  seq_len: int
  num_features: int


NONZERO_CONFIGS = (
    NonZeroConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    NonZeroConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
)
