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

import dataclasses
from typing import Any, Mapping

import flax
import flax.nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class ModelAndInput:
  model: Any
  inputs: jax.Array


def get_model_and_input(
    model_name: str,
    batch_size: int,
    seq_len: int,
    custom_kwargs: Mapping[str, Any],
) -> ModelAndInput:
  rngs = flax.nnx.Rngs(0)
  key = jax.random.PRNGKey(0)
  dtype = jnp.bfloat16

  if model_name == "nnx.Linear":
    in_features = custom_kwargs["in_features"]
    out_features = custom_kwargs["out_features"]

    class LinearModel(flax.nnx.Module):

      def __init__(self):
        bound = 1.0 / in_features**0.5
        self.linear = flax.nnx.Linear(
            in_features,
            out_features,
            dtype=dtype,
            rngs=rngs,
            bias_init=jax.nn.initializers.uniform(scale=bound),
        )

      def __call__(self, x):
        return self.linear(x)

    model = LinearModel()
    inputs = jax.random.normal(
        key, (batch_size, seq_len, in_features), dtype=dtype
    )

  elif model_name == "nnx.BatchNorm1d":
    num_features = custom_kwargs["num_features"]

    class BatchNormModel(flax.nnx.Module):

      def __init__(self):
        self.batchnorm = flax.nnx.BatchNorm(
            num_features, dtype=dtype, rngs=rngs
        )

      def __call__(self, x):
        return self.batchnorm(x)

    model = BatchNormModel()
    inputs = jax.random.normal(
        key, (batch_size, seq_len, num_features), dtype=dtype
    )

  elif model_name == "nnx.LayerNorm":
    shape = custom_kwargs["shape"]
    normalized_shape = custom_kwargs["normalized_shape"]
    num_normalized_dims = len(normalized_shape)
    reduction_axes = range(len(shape) - num_normalized_dims, len(shape))
    num_features = shape[-1]

    class LayerNormModel(flax.nnx.Module):

      def __init__(self):
        self.layernorm = flax.nnx.LayerNorm(
            num_features=num_features,
            dtype=dtype,
            rngs=rngs,
            reduction_axes=reduction_axes,
        )

      def __call__(self, x):
        return self.layernorm(x)

    model = LayerNormModel()
    inputs = jax.random.normal(key, shape, dtype=dtype)

  elif model_name == "nnx.Conv2d":
    in_channels = custom_kwargs["in_channels"]
    out_channels = custom_kwargs["out_channels"]
    kernel_size = custom_kwargs["kernel_size"]
    stride = custom_kwargs["stride"]
    padding = custom_kwargs["padding"]
    height = custom_kwargs["height"]
    width = custom_kwargs["width"]

    class Conv2dModel(flax.nnx.Module):

      def __init__(self):
        bound = 1 / (in_channels * kernel_size**2) ** 0.5
        self.conv = flax.nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding,
            dtype=dtype,
            rngs=rngs,
            bias_init=jax.nn.initializers.uniform(scale=bound),
        )

      def __call__(self, x):
        return self.conv(x)

    model = Conv2dModel()
    # NHWC format for Conv in Flax
    inputs = jax.random.normal(
        key, (batch_size, height, width, in_channels), dtype=dtype
    )

  elif model_name == "nnx.RMSNorm":
    num_features = custom_kwargs["num_features"]

    class RMSNormModel(flax.nnx.Module):

      def __init__(self):
        self.rmsnorm = flax.nnx.RMSNorm(num_features, dtype=dtype, rngs=rngs)

      def __call__(self, x):
        return self.rmsnorm(x)

    model = RMSNormModel()
    inputs = jax.random.normal(
        key, (batch_size, seq_len, num_features), dtype=dtype
    )

  elif model_name == "nnx.sdpa":
    num_heads = custom_kwargs["num_heads"]
    head_dim = custom_kwargs["head_dim"]
    embed_dim = num_heads * head_dim

    class SDPAModel(flax.nnx.Module):

      def __init__(self):
        self.q_proj = flax.nnx.Linear(
            embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_proj = flax.nnx.Linear(
            embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.v_proj = flax.nnx.Linear(
            embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.out_proj = flax.nnx.Linear(
            embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs
        )

      def __call__(self, x):
        bsz, q_len, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape((bsz, q_len, num_heads, head_dim))
            .transpose((0, 2, 1, 3))
        )
        k = (
            self.k_proj(x)
            .reshape((bsz, q_len, num_heads, head_dim))
            .transpose((0, 2, 1, 3))
        )
        v = (
            self.v_proj(x)
            .reshape((bsz, q_len, num_heads, head_dim))
            .transpose((0, 2, 1, 3))
        )
        attn_output = jax.nn.dot_product_attention(q, k, v)
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(
            (bsz, q_len, embed_dim)
        )
        return self.out_proj(attn_output)

    model = SDPAModel()
    inputs = jax.random.normal(
        key, (batch_size, seq_len, embed_dim), dtype=dtype
    )

  elif model_name == "nnx.nonzero":
    num_features = custom_kwargs["num_features"]
    shape = (batch_size, seq_len, num_features)

    class NonZeroModel(flax.nnx.Module):

      def __init__(self):
        max_size = 1
        for s in shape:
          max_size *= s
        self.max_size = max_size

      def __call__(self, x):
        return jnp.stack(jnp.nonzero(x, size=self.max_size), axis=-1)

    model = NonZeroModel()
    x = jax.random.normal(key, shape, dtype=dtype)
    mask = (jax.random.uniform(key, shape) < 0.1).astype(dtype)
    inputs = x * mask

  elif model_name == "nnx.topk":
    num_features = custom_kwargs["num_features"]
    k = custom_kwargs["k"]
    dim = custom_kwargs["dim"]
    shape = (batch_size, seq_len, num_features)

    class TopkModel(flax.nnx.Module):

      def __init__(self):
        self.k = k
        self.axis = dim

      def __call__(self, x):
        return jax.lax.top_k(x, self.k, axis=self.axis)

    model = TopkModel()
    inputs = jax.random.normal(key, shape, dtype=dtype)

  else:
    raise ValueError(f"Unknown model_name: {model_name}")

  return ModelAndInput(model, inputs)
