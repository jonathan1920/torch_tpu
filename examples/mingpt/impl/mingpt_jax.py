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

from ast import literal_eval
import math
from typing import NamedTuple
import jax
from jax import random
import jax.numpy as jnp


class CfgNode:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __str__(self):
    return self._str_helper(0)

  def _str_helper(self, indent):
    parts = []
    for k, v in self.__dict__.items():
      if isinstance(v, CfgNode):
        parts.append(' ' * (indent * 4) + f'{k}:\n')
        parts.append(v._str_helper(indent + 1))
      else:
        parts.append(' ' * (indent * 4) + f'{k}: {v}\n')
    return ''.join(parts)

  def to_dict(self):
    return {
        k: v.to_dict() if isinstance(v, CfgNode) else v
        for k, v in self.__dict__.items()
    }

  def merge_from_dict(self, d):
    for k, v in d.items():
      if hasattr(self, k):
        if isinstance(v, dict) and isinstance(getattr(self, k), CfgNode):
          getattr(self, k).merge_from_dict(v)
        else:
          setattr(self, k, v)
      else:
        setattr(self, k, v)

  def merge_from_args(self, args):
    for arg in args:
      keyval = arg.split('=')
      assert len(keyval) == 2, f'invalid arg: {arg}'
      key, val = keyval
      assert key.startswith('--')
      key = key[2:]
      keys = key.split('.')
      try:
        val = literal_eval(val)
      except:
        pass
      obj = self
      for k in keys[:-1]:
        obj = getattr(obj, k)
      setattr(obj, keys[-1], val)


def get_default_config():
  C = CfgNode()
  C.model_type = 'gpt'
  C.n_layer = None
  C.n_head = None
  C.n_embd = None
  C.vocab_size = None
  C.block_size = None
  C.embd_pdrop = 0.1
  C.resid_pdrop = 0.1
  C.attn_pdrop = 0.1
  return C


def apply_model_type(config):
  table = {
      'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
      'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
      'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
      'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
      'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
      'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
      'tpu-optimal-xl-hd128': dict(n_layer=40, n_head=32, n_embd=4096),
      'tpu-optimal-l-hd128': dict(n_layer=8, n_head=32, n_embd=1024),
  }
  if config.model_type in table:
    config.merge_from_dict(table[config.model_type])


class GPTParams(NamedTuple):
  params: dict
  config: CfgNode


def gelu(x):
  return (
      0.5
      * x
      * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
  )


def causal_mask(T):
  return jnp.tril(jnp.ones((T, T)))


def init_linear(rng, in_dim, out_dim):
  k1, k2 = random.split(rng)
  w = random.normal(k1, (in_dim, out_dim)) * 0.02
  b = jnp.zeros((out_dim,))
  return (w, b)


def linear(params, x):
  w, b = params
  return jnp.dot(x, w) + b


def split_heads(x, n_head):
  B, T, C = x.shape
  return x.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)


def merge_heads(x):
  B, nh, T, hs = x.shape
  return x.transpose(0, 2, 1, 3).reshape(B, T, nh * hs)


def attention(params, x, mask, config):
  wqkv, wo = params
  B, T, C = x.shape
  qkv = linear(wqkv, x)
  q, k, v = jnp.split(qkv, 3, axis=-1)
  q = split_heads(q, config.n_head)
  k = split_heads(k, config.n_head)
  v = split_heads(v, config.n_head)
  scale = 1.0 / math.sqrt(k.shape[-1])
  att = jnp.matmul(q, jnp.transpose(k, axes=(0, 1, 3, 2))) * scale
  att = jnp.where(mask == 0, -1e10, att)
  att = jax.nn.softmax(att, axis=-1)
  y = jnp.matmul(att, v)
  y = merge_heads(y)
  y = linear(wo, y)
  return y


def mlp(params, x):
  w1, w2 = params
  return linear(w2, gelu(linear(w1, x)))


def layer_norm(x, weight, bias):
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  return (x - mean) / jnp.sqrt(var + 1e-5) * weight + bias


def transformer_block(params, x, mask, config):
  ln1_w, ln1_b, ln2_w, ln2_b, attn_params, mlp_params = params
  x1 = x + attention(attn_params, layer_norm(x, ln1_w, ln1_b), mask, config)
  x2 = x1 + mlp(mlp_params, layer_norm(x1, ln2_w, ln2_b))
  return x2


def init_gpt_params(rng, config):
  keys = random.split(rng, config.n_layer * 6 + 5)
  params = {}
  params['wte'] = (
      random.normal(keys[0], (config.vocab_size, config.n_embd)) * 0.02
  )
  params['wpe'] = (
      random.normal(keys[1], (config.block_size, config.n_embd)) * 0.02
  )
  params['ln_f'] = (jnp.ones((config.n_embd,)), jnp.zeros((config.n_embd,)))
  blocks = []
  for i in range(config.n_layer):
    ln1_w = jnp.ones((config.n_embd,))
    ln1_b = jnp.zeros((config.n_embd,))
    ln2_w = jnp.ones((config.n_embd,))
    ln2_b = jnp.zeros((config.n_embd,))
    wqkv = init_linear(keys[2 + i * 6 + 0], config.n_embd, 3 * config.n_embd)
    wo = init_linear(keys[2 + i * 6 + 1], config.n_embd, config.n_embd)
    w1 = init_linear(keys[2 + i * 6 + 2], config.n_embd, 4 * config.n_embd)
    w2 = init_linear(keys[2 + i * 6 + 3], 4 * config.n_embd, config.n_embd)
    block = (ln1_w, ln1_b, ln2_w, ln2_b, (wqkv, wo), (w1, w2))
    blocks.append(block)
  params['blocks'] = blocks
  params['lm_head'] = init_linear(keys[-1], config.n_embd, config.vocab_size)
  return GPTParams(params, config)


def gpt_forward(params, idx, targets=None):
  config = params.config
  p = params.params
  B, T = idx.shape
  tok_emb = p['wte'][idx]
  pos_emb = p['wpe'][:T]
  x = tok_emb + pos_emb
  mask = causal_mask(T)
  for block in p['blocks']:
    x = transformer_block(block, x, mask, config)
  ln_w, ln_b = p['ln_f']
  x = layer_norm(x, ln_w, ln_b)
  logits = linear(p['lm_head'], x)
  loss = None
  if targets is not None:
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    loss = -jax.nn.log_softmax(logits_flat)[
        jnp.arange(targets_flat.size), targets_flat
    ]
    loss = jnp.mean(loss)
  return logits, loss
