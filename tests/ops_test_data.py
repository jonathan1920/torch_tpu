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

"""Utility for SDPA tests."""

import dataclasses
import sys
from absl import flags
import torch

asdict = dataclasses.asdict

_RUN_CONFIGS = flags.DEFINE_enum(
    "run_configs",
    "all",
    ["first", "all"],
    "Which test configurations to run. 'first' runs only the first config,"
    " while 'all' runs all configurations.",
)


def generate_configs_for_parameterized(configs):
  """Generates test parameters from a list of configuration dataclasses."""
  # Parse flags if they haven't been parsed yet. This is necessary because
  # parameterized decorators are executed at module import time, before
  # absltest.main() has a chance to parse them.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

  for config in configs:
    config_dict = asdict(config)
    name_parts = (
        f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
        for k, v in config_dict.items()
    )
    testcase_name = "_".join(name_parts)
    yield dict(testcase_name=testcase_name, config=config)
    if _RUN_CONFIGS.value == "first":
      # We consider only the first configuration.
      break


@dataclasses.dataclass(frozen=True)
class SdpaConfig:
  """Configuration for SDPA tests."""
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
  scale: float | None = None
  use_math_backend: bool = False


SDPA_CONFIGS = [
    # Default config for smoke test.
    SdpaConfig(
        batch_size=1,
        embed_dim=512,
        q_seq_len=512,
        q_num_heads=8,
        kv_num_heads=8,
        qk_head_dim=64,
        v_head_dim=64,
        is_causal=True,
        enable_gqa=True,
        dtype=torch.bfloat16,
        scale=1.1 / 64**0.5,  # Slightly different from default 1.0 / sqrt(64)
    ),
    # Configs for Llama3 70B attention layers
    SdpaConfig(
        batch_size=1,
        embed_dim=8192,
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
        embed_dim=8192,
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
        embed_dim=6144,
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
        embed_dim=6144,
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
        embed_dim=2048,
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
        embed_dim=2048,
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
        batch_size=2,
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
