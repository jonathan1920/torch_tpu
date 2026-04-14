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
from typing import Any
import torch


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
  embed_dim: int
  q_seq_len: int
  q_num_heads: int
  kv_num_heads: int
  qk_head_dim: int
  v_head_dim: int
  is_causal: bool
  enable_gqa: bool
  dtype: torch.dtype
  backend: Any = torch.nn.attention.SDPBackend.MATH

  @classmethod
  def get_base_configs(cls):
    return [
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=128,
            q_num_heads=8,
            kv_num_heads=8,
            qk_head_dim=64,
            v_head_dim=64,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Llama3 70B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=64,
            kv_num_heads=64,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=64,
            kv_num_heads=64,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Qwen3 480B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=96,
            kv_num_heads=96,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=96,
            kv_num_heads=96,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Gemma3 27B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=32,
            kv_num_heads=32,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=32,
            kv_num_heads=32,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # AFM v7 configs
        SdpaConfig(
            batch_size=4,
            embed_dim=2048,
            q_seq_len=2048,
            q_num_heads=16,
            kv_num_heads=2,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
    ]

  @classmethod
  def configs_with_backends(cls, *backends):
    for backend in backends:
      for config in cls.get_base_configs():
        yield dataclasses.replace(config, backend=backend)


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


@dataclasses.dataclass
class EmbeddingConfig:
  batch_size: int
  seq_len: int
  num_embeddings: int
  embedding_dim: int


EMBEDDING_CONFIGS = (
    # Default config for smoke test.
    EmbeddingConfig(
        batch_size=1,
        seq_len=128,
        num_embeddings=128,
        embedding_dim=128,
    ),
    # Configs for BERT
    EmbeddingConfig(
        batch_size=32,
        seq_len=128,
        num_embeddings=30522,
        embedding_dim=768,
    ),
    EmbeddingConfig(
        batch_size=1,
        seq_len=128,
        num_embeddings=512,
        embedding_dim=768,
    ),
    EmbeddingConfig(
        batch_size=32,
        seq_len=128,
        num_embeddings=2,
        embedding_dim=768,
    ),
)


@dataclasses.dataclass
class DropoutConfig:
  p: float
  shape: tuple[int, ...]


DROPOUT_CONFIGS = (
    # BERT configs
    DropoutConfig(
        p=0.1,
        shape=(32, 128, 768),
    ),
)


@dataclasses.dataclass
class TanhConfig:
  shape: tuple[int, ...]


TANH_CONFIGS = (
    # BERT configs
    TanhConfig(
        shape=(32, 768),
    ),
)


@dataclasses.dataclass
class BertLayerConfig:
  batch_size: int
  seq_len: int


BERT_LAYER_CONFIGS = (BertLayerConfig(batch_size=32, seq_len=128),)


@dataclasses.dataclass
class Qwen3Config:
  batch_size: int
  seq_len: int
  hidden_size: int = 128
  intermediate_size: int = 512
  head_dim: int = 128


QWEN3_CONFIGS = (Qwen3Config(batch_size=1, seq_len=128),)


@dataclasses.dataclass
class SiLUConfig:
  batch_size: int
  seq_len: int
  shape: tuple[int, ...] = (1, 128, 512)


SILU_CONFIGS = (SiLUConfig(batch_size=1, seq_len=128),)


@dataclasses.dataclass
class DeepSeekConfig:
  batch_size: int
  seq_len: int
  vocab_size: int = 1024
  dim: int = 128
  inter_dim: int = 512
  moe_inter_dim: int = 64
  n_layers: int = 1
  n_dense_layers: int = 1
  n_heads: int = 4
  n_routed_experts: int = 4
  n_shared_experts: int = 2
  n_activated_experts: int = 2
  in_features: int = 128
  out_features: int = 128


DEEPSEEK_CONFIGS = (DeepSeekConfig(batch_size=1, seq_len=128),)
