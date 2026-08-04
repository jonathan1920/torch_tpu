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

"""TPU optimized attention implementation for Gemma 4."""

import os
from typing import Any, Optional
from absl import logging
import torch
from torch import nn

try:
  from torch_tpu.ops import splash_attention
except (ImportError, ModuleNotFoundError):
  splash_attention = None
from transformers import modeling_utils

_DEBUG_ATTENTION = (
    os.environ.get("TORCH_TPU_DEBUG_ATTENTION", "False").lower() == "true"
)


def get_stats_str(t: torch.Tensor) -> str:
  t_cpu = t.cpu().float()
  return (
      f"dtype={t.dtype}, mean={t_cpu.mean().item():.6f},"
      f" std={t_cpu.std().item():.6f}, max={t_cpu.abs().max().item():.6f}"
  )


def torch_tpu_gemma4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """Attention forward implementation optimized for TPU."""
  del kwargs
  use_bidirectional_attn = getattr(
      module.config, "use_bidirectional_attention", None
  )
  skip_sliding_mask = use_bidirectional_attn == "vision"
  is_sliding = getattr(module, "is_sliding", False)
  use_swa_env = os.environ.get("TORCH_TPU_USE_SWA", "True").lower() == "true"
  use_swa = is_sliding and not skip_sliding_mask and use_swa_env

  if scaling is None:
    scaling = 1.0

  q_len = query.shape[-2]
  kv_len = key.shape[-2]
  device_type = query.device.type

  logging.debug(
      "[DEBUG] Device: %s, type: %s, is_sliding: %s, use_swa: %s",
      query.device,
      device_type,
      is_sliding,
      use_swa,
  )

  # Check if we can use Splash Attention (requires TPU and large enough sequence)
  use_splash = (
      device_type in ("xla", "tpu")
      and splash_attention is not None
      and use_swa
      and sliding_window is not None
      and kv_len > sliding_window
      and q_len >= 512
      and attention_mask is None
  )

  if use_splash:
    assert splash_attention is not None
    if module.training and dropout != 0.0:
      raise ValueError(
          "Splash Attention on TPU does not support dropout during training."
      )
    logging.debug(
        "[DEBUG_ATTN] Layer %s (SWA Splash) | q_len=%s >= 512",
        module.layer_idx,
        q_len,
    )
    if _DEBUG_ATTENTION:
      logging.debug(
          "[DEBUG_ATTN] Layer %s (SWA) | device: %s | Q: %s",
          module.layer_idx,
          query.device,
          get_stats_str(query),
      )
      logging.debug(
          "[DEBUG_ATTN] Layer %s (SWA) | device: %s | K: %s",
          module.layer_idx,
          query.device,
          get_stats_str(key),
      )
      logging.debug(
          "[DEBUG_ATTN] Layer %s (SWA) | device: %s | V: %s",
          module.layer_idx,
          query.device,
          get_stats_str(value),
      )

    num_key_value_groups = getattr(module, "num_key_value_groups", 1)
    attn_output = splash_attention.splash_sdpa(
        query,
        key,
        value,
        scale=scaling,
        is_causal=getattr(module, "is_causal", True),
        local_window_size=sliding_window - 1,
        enable_gqa=(num_key_value_groups > 1),
        block_q=1024,
        block_kv=1024,
        block_dkv=1024,
        block_kv_compute=1024,
        block_q_dkv=1024,
        block_kv_dkv=1024,
        block_kv_dkv_compute=1024,
    )
  else:
    # Fallback path: use standard SDPA (which is compile-friendly on all devices)
    logging.debug(
        "[DEBUG_ATTN] Layer %s (SDPA Fallback) | device=%s, use_swa=%s,"
        " q_len=%s, kv_len=%s",
        module.layer_idx,
        device_type,
        use_swa,
        q_len,
        kv_len,
    )
    if _DEBUG_ATTENTION:
      logging.debug(
          "[DEBUG_ATTN] Layer %s (Fallback) | device: %s | Q: %s",
          module.layer_idx,
          query.device,
          get_stats_str(query),
      )
      logging.debug(
          "[DEBUG_ATTN] Layer %s (Fallback) | device: %s | K: %s",
          module.layer_idx,
          query.device,
          get_stats_str(key),
      )
      logging.debug(
          "[DEBUG_ATTN] Layer %s (Fallback) | device: %s | V: %s",
          module.layer_idx,
          query.device,
          get_stats_str(value),
      )

    if use_swa:
      assert (
          sliding_window is not None
      ), "sliding_window must be provided for SWA"

      if kv_len <= sliding_window:
        logging.debug(
            "[DEBUG_ATTN] Layer %s (SWA->Causal fallback) | kv_len=%s <="
            " window=%s",
            module.layer_idx,
            kv_len,
            sliding_window,
        )
        if attention_mask is not None:
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              query,
              key,
              value,
              attn_mask=attention_mask,
              is_causal=False,
              dropout_p=dropout if module.training else 0.0,
              scale=scaling,
          )
        else:
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              query,
              key,
              value,
              is_causal=module.is_causal,
              dropout_p=dropout if module.training else 0.0,
              scale=scaling,
          )
      else:
        # Generate SWA mask
        kv_offset = kv_len - q_len
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool),
            diagonal=1 + kv_offset,
        )
        window_mask = torch.tril(
            torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool),
            diagonal=-sliding_window + kv_offset,
        )
        mask = ~(causal_mask | window_mask)

        if attention_mask is not None:
          if attention_mask.dtype == torch.bool:
            mask = mask & attention_mask
          else:
            float_mask = torch.zeros_like(attention_mask, dtype=query.dtype)
            float_mask.masked_fill_(~mask, float("-inf"))
            mask = float_mask + attention_mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=False,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
    else:
      # Non-SWA
      if attention_mask is not None:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
      else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=module.is_causal,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )

  attn_output = attn_output.transpose(1, 2).contiguous()
  if _DEBUG_ATTENTION:
    logging.debug(
        "[DEBUG_ATTN] Layer %s | device: %s | OUT: %s",
        module.layer_idx,
        query.device,
        get_stats_str(attn_output),
    )
  return attn_output, None


# Register globally
modeling_utils.ALL_ATTENTION_FUNCTIONS.register(
    "torch_tpu_gemma4", torch_tpu_gemma4_attention_forward
)
