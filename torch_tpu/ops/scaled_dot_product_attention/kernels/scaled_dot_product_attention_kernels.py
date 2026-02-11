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

"""JAX/Pallas kernels scaled dot product attention."""

import functools
import jax
# import jax.experimental.pallas as pl
import jax.nn as jnn
import jax.numpy as jnp
import torch_tpu._internal.pallas.pallas_kernel_generate_utils as generate_utils

DEFAULT_MASKED_VALUE = -1e30


######################################################################
# def sdpa_kernel_reference(q, k, v):
#   seq_len, head_dim = q.shape
#   scale = 1.0 / jnp.sqrt(head_dim)
#   attn_scores = jnp.einsum("qd,kd->qk", q, k) * scale
#   i = jnp.arange(seq_len)[:, None]
#   j = jnp.arange(seq_len)[None, :]
#   mask = j <= i
#   attn_scores = jnp.where(mask, attn_scores, DEFAULT_MASKED_VALUE)
#   attn_weights = jnn.softmax(attn_scores, axis=1)
#   return jnp.einsum("qk,kd->qd", attn_weights, v)


def sdpa_forward_kernel_reference_jax(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  """Reference implementation using jax.nn.dot_product_attention."""

  del dropout_p, enable_gqa

  assert attn_mask is None

  return jnp.swapaxes(
      jnn.dot_product_attention(
          jnp.swapaxes(query, -3, -2),
          jnp.swapaxes(key, -3, -2),
          jnp.swapaxes(value, -3, -2),
          is_causal=is_causal,
          scale=scale,
      ),
      -3,
      -2,
  )


def sdpa_forward_kernel_reference_torch(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Port of reference implementation from Torch documentation page.

  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  """
  # pylint: enable=g-doc-args,g-doc-return-or-yield
  assert attn_mask is None
  assert dropout_p == 0.0

  # Hq = query.shape[-2]
  # H = key.shape[-2]
  L, S = query.shape[-2], key.shape[-2]  # pylint: disable=invalid-name
  scale_factor = 1 / jnp.sqrt(query.shape[-1]) if scale is None else scale
  attn_bias = jnp.zeros(shape=(L, S), dtype=query.dtype)
  if is_causal:
    assert attn_mask is None
    temp_mask = jnp.triu(jnp.ones(shape=(L, S), dtype=jnp.bool), k=1)
    attn_bias = temp_mask * DEFAULT_MASKED_VALUE

  # TODO(elliotenglish): Add support for attn_mask.
  # if attn_mask is not None:
  #   if attn_mask.dtype == torch.bool:
  #     attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
  #   else:
  #     attn_bias = attn_mask + attn_bias

  if enable_gqa:
    key = jnp.repeat(key, query.shape[-3] // key.shape[-3], -3)
    value = jnp.repeat(value, query.shape[-3] // value.shape[-3], -3)

  attn_weight = query @ jnp.swapaxes(key, -2, -1) * scale_factor
  attn_weight += attn_bias
  attn_weight = jnn.softmax(attn_weight, axis=-1)
  # TODO(elliotenglish): Add support for dropout.
  # attn_weight = ...dropout(attn_weight, dropout_p, train=True)
  return attn_weight @ value


#######################################################################
def forward_kernel():
  dropout_p = 0.0
  attn_mask = None
  is_causal = True
  scale = None
  enable_gqa = True

  kernel_fn = functools.partial(
      sdpa_forward_kernel_reference_torch,
      dropout_p=dropout_p,
      attn_mask=attn_mask,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
  )

  return kernel_fn


def backward_kernel():
  kernel_fn = forward_kernel()

  def backward_kernel_fn(grad_out, q, k, v):
    return jax.vjp(kernel_fn, q, k, v)[1](grad_out)

  return backward_kernel_fn


#######################################################################
def export_sdpa_forward_kernel():
  """Exports a dynamic sdpa kernel."""

  kernel_fn = forward_kernel()

  (
      # pylint: disable=invalid-name
      B_sym,
      Hq_sym,
      H_sym,
      L_sym,
      S_sym,
      E_sym,
      Ey_sym,
      # pylint: enable=invalid-name
  ) = jax.export.symbolic_shape(
      "B,Hq,H,L,S,E,Ey",
      constraints=(
          "H*floordiv(Hq, H)==Hq",
          # The following are true given the above constraint, but the symbolic
          # shape resolution logic can't handle this currently.
          "B*E*H*S*floordiv(Hq, H)==B*E*Hq*S",
          "B*Ey*H*S*floordiv(Hq, H)==B*Ey*Hq*S",
      ),
  )

  q_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, E_sym), jnp.float32)
  k_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, E_sym), jnp.float32)
  v_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, Ey_sym), jnp.float32)

  f_export = generate_utils.export(kernel_fn)

  return f_export(
      q_shape,
      k_shape,
      v_shape,
  )


def export_sdpa_backward_kernel():
  """Exports a dynamic sdpa kernel."""

  backward_kernel_fn = backward_kernel()

  (
      # pylint: disable=invalid-name
      B_sym,
      Hq_sym,
      H_sym,
      L_sym,
      S_sym,
      E_sym,
      Ey_sym,
      # pylint: enable=invalid-name
  ) = jax.export.symbolic_shape(
      "B,Hq,H,L,S,E,Ey",
      constraints=(
          "H*floordiv(Hq, H)==Hq",
          # The following are true given the above constraint, but the symbolic
          # shape resolution logic can't handle this currently.
          "B*E*H*S*floordiv(Hq, H)==B*E*Hq*S",
          "B*Ey*H*S*floordiv(Hq, H)==B*Ey*Hq*S",
      ),
  )

  q_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, E_sym), jnp.float32)
  k_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, E_sym), jnp.float32)
  v_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, Ey_sym), jnp.float32)

  out_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, Ey_sym), jnp.float32)

  f_export = generate_utils.export(backward_kernel_fn)

  return f_export(
      out_shape,
      q_shape,
      k_shape,
      v_shape,
  )


#######################################################################
def sdpa_forward_kernel_export_call(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  """Exports a dynamic kernel and calls it."""
  assert attn_mask is None
  assert dropout_p == 0.0
  assert is_causal
  assert scale is None
  del enable_gqa
  exported = export_sdpa_forward_kernel()
  return exported.call(q, k, v)
