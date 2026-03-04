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
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import torch
import torch_tpu._internal.pallas.pallas_kernel_generate_utils as generate_utils

from . import flash_attention

try:
  # Works whenever we have an updated jaxlib, i.e. g3 and when oss goes out.
  import jax.experimental.pallas  # pylint: disable=g-import-not-at-top

  pallas_export_experimental = (
      jax.experimental.pallas.pallas_export_experimental
  )
except AttributeError:
  # Works in oss because we don't dependency check
  import jax._src.pallas.core as pallas_core  # pylint: disable=g-import-not-at-top

  pallas_export_experimental = pallas_core.pallas_export_experimental

DEFAULT_MASKED_VALUE = -1e30
_BLOCK_SIZE = 512

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


def sdpa_backward_kernel_reference_jax(
    grad_out,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  """Reference backward implementation for JAX."""

  def forward_fn(q, k, v):
    return sdpa_forward_kernel_reference_jax(
        q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )

  return jax.vjp(forward_fn, query, key, value)[1](grad_out)


def sdpa_backward_kernel_reference_torch(
    grad_out,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  """Reference backward implementation for Torch."""

  # Convert JAX arrays to Torch tensors.
  def to_torch(x):
    if x is None:
      return None
    return torch.from_numpy(np.array(x)).requires_grad_(True)

  q_t = to_torch(query)
  k_t = to_torch(key)
  v_t = to_torch(value)
  mask_t = to_torch(attn_mask)
  grad_out_t = torch.from_numpy(np.array(grad_out))

  assert q_t is not None
  assert k_t is not None
  assert v_t is not None

  if enable_gqa:
    num_heads_q = q_t.shape[-3]
    num_heads_kv = k_t.shape[-3]
    repeat_factor = num_heads_q // num_heads_kv
    # We use new variables (k_rep, v_rep) so k_t and v_t remain leaf tensors.
    k_rep = k_t.repeat_interleave(repeat_factor, dim=-3)  # pylint: disable=invalid-name
    v_rep = v_t.repeat_interleave(repeat_factor, dim=-3)  # pylint: disable=invalid-name
  else:
    k_rep = k_t  # pylint: disable=invalid-name
    v_rep = v_t  # pylint: disable=invalid-name

  # Run Torch SDPA forward.
  out_t = torch.nn.functional.scaled_dot_product_attention(
      q_t,
      k_rep,
      v_rep,
      attn_mask=mask_t,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
  )

  # Run Torch SDPA backward.
  out_t.backward(grad_out_t)

  # Convert Torch gradients back to JAX arrays.
  def to_jax(x):
    if x is None or x.grad is None:
      return None
    return jnp.array(x.grad.detach().numpy())

  return to_jax(q_t), to_jax(k_t), to_jax(v_t)


#######################################################################
def forward_kernel(is_causal=True):
  """Returns a kernel function for scaled dot product attention forward pass."""
  dropout_p = 0.0
  attn_mask = None
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


def backward_kernel(is_causal=True):
  kernel_fn = forward_kernel(is_causal=is_causal)

  def backward_kernel_fn(grad_out, q, k, v):
    return jax.vjp(kernel_fn, q, k, v)[1](grad_out)

  return backward_kernel_fn


#######################################################################
def export_sdpa_forward_kernel(
    static_seq_len,
    static_head_dim,
    num_q_heads,
    batch_size,
    kernel_type,
    is_causal,
    dtype,
):
  """Exports a dynamic sdpa kernel."""
  if kernel_type == "flash":
    return export_flash_attention_forward_kernel(
        batch_size=batch_size,
        num_of_heads=num_q_heads,
        seq_len=static_seq_len,
        head_dim=static_head_dim,
        causal=is_causal,
        dtype=dtype,
    )

  kernel_fn = forward_kernel(is_causal=is_causal)

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

  q_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, E_sym), dtype)
  k_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, E_sym), dtype)
  v_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, Ey_sym), dtype)

  f_export = generate_utils.export(kernel_fn)

  return f_export(
      q_shape,
      k_shape,
      v_shape,
  )


def export_sdpa_backward_kernel(
    static_seq_len,
    static_head_dim,
    num_q_heads,
    batch_size,
    kernel_type,
    is_causal,
    dtype,
):
  """Exports a dynamic sdpa kernel."""
  if kernel_type == "flash":
    return export_flash_attention_backward_kernel(
        batch_size=batch_size,
        num_of_heads=num_q_heads,
        seq_len=static_seq_len,
        head_dim=static_head_dim,
        causal=is_causal,
        dtype=dtype,
    )

  backward_kernel_fn = backward_kernel(is_causal=is_causal)

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

  q_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, E_sym), dtype)
  k_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, E_sym), dtype)
  v_shape = jax.ShapeDtypeStruct((B_sym, H_sym, S_sym, Ey_sym), dtype)

  out_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, L_sym, Ey_sym), dtype)

  f_export = generate_utils.export(backward_kernel_fn)

  return f_export(
      out_shape,
      q_shape,
      k_shape,
      v_shape,
  )


#######################################################################
# For testing purposes only.
def sdpa_forward_kernel_export_call(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    static_seq_len=None,
    static_head_dim=None,
    num_q_heads=None,
    batch_size=None,
    kernel_type="flash",
):
  """Exports a dynamic kernel and calls it."""
  assert attn_mask is None
  assert dropout_p == 0.0
  if kernel_type != "flash":
    assert is_causal
  assert scale is None
  del enable_gqa
  exported = export_sdpa_forward_kernel(
      static_seq_len=static_seq_len,
      static_head_dim=static_head_dim,
      num_q_heads=num_q_heads,
      batch_size=batch_size,
      kernel_type=kernel_type,
      is_causal=is_causal,
      dtype=q.dtype,
  )

  return exported.call(q, k, v)


def sdpa_backward_kernel_export_call(
    grad_out,
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    static_seq_len=None,
    static_head_dim=None,
    num_q_heads=None,
    batch_size=None,
    kernel_type="flash",
):
  """Exports a dynamic backward kernel and calls it."""
  assert attn_mask is None
  assert dropout_p == 0.0
  if kernel_type != "flash":
    assert is_causal
  assert scale is None
  del enable_gqa
  exported = export_sdpa_backward_kernel(
      static_seq_len=static_seq_len,
      static_head_dim=static_head_dim,
      num_q_heads=num_q_heads,
      batch_size=batch_size,
      kernel_type=kernel_type,
      is_causal=is_causal,
      dtype=q.dtype,
  )

  gq, gk, gv = exported.call(grad_out, q, k, v)
  return gq, gk, gv


########################################################################
## Flash Attention Kernels ##
########################################################################
def flash_attention_forward_kernel_wrapper(
    q, k, v, block_size=_BLOCK_SIZE, **kwargs
):
  """Wrapper for Flash Attention kernel."""
  _, _, _, head_dim_qk = q.shape
  q = q / jnp.sqrt(head_dim_qk)
  out = flash_attention.flash_attention(
      q=q,
      k=k,
      v=v,
      block_sizes=flash_attention.BlockSizes(
          block_q=block_size,
          block_k=block_size,
          block_k_major=block_size,
          block_b=4,
          block_q_major_dkv=block_size,
          block_k_major_dkv=block_size,
          block_k_dkv=block_size,
          block_q_dkv=block_size,
          block_k_major_dq=block_size,
          block_k_dq=block_size,
          block_q_dq=block_size,
      ),
      **kwargs,
  )
  return out


def export_flash_attention_forward_kernel(
    batch_size,
    num_of_heads,
    seq_len,
    head_dim,
    dtype,
    block_size=_BLOCK_SIZE,
    **kwargs,
):
  """Exports a Flash Attention kernel as an MLIR module."""

  if batch_size is None:
    # Use symbolic shapes
    # This set of symbolic shapes set the constraint explicitly and implicitly.:
    # 1. batch_size >= 4
    # 2. q_seq_len and kv_seq_len are multiples of 512.
    # 3. head_dim is a multiple of 128.
    # 4. All qkv have the same heads, head dimension and batch size.
    (
        batch_size_sym,
        num_of_heads_sym,
        q_seq_len_sym,
        kv_seq_len_sym,
        head_dim_sym,
    ) = jax.export.symbolic_shape(
        "batch_size,num_of_heads,q_seq_len, kv_seq_len,head_dim",
        constraints=(
            "batch_size >= 4",
            f"q_seq_len >= {block_size}",
            f"kv_seq_len >= {block_size}",
            f"mod(q_seq_len, {block_size}) == 0",
            f"mod(kv_seq_len, {block_size}) == 0",
        ),
    )

    q_shape = (batch_size_sym, num_of_heads_sym, q_seq_len_sym, head_dim_sym)
    k_shape = (batch_size_sym, num_of_heads_sym, kv_seq_len_sym, head_dim_sym)
    v_shape = (batch_size_sym, num_of_heads_sym, kv_seq_len_sym, head_dim_sym)
  else:
    # Defaults if not symbolic
    q_shape = (batch_size, num_of_heads, seq_len, head_dim)
    k_shape = (batch_size, num_of_heads, seq_len, head_dim)
    v_shape = (batch_size, num_of_heads, seq_len, head_dim)

  q_dtype = jax.ShapeDtypeStruct(q_shape, dtype)
  k_dtype = jax.ShapeDtypeStruct(k_shape, dtype)
  v_dtype = jax.ShapeDtypeStruct(v_shape, dtype)

  f_p = functools.partial(flash_attention_forward_kernel_wrapper, **kwargs)
  f_j = jax.jit(f_p)
  f_e = jax.export.export(f_j, platforms=["tpu"])
  with pallas_export_experimental(dynamic_shapes=True):
    f_k = f_e(q_dtype, k_dtype, v_dtype)
  return f_k


def flash_attention_backward_kernel_wrapper(grad_out, q, k, v, **kwargs):
  """Wrapper for Flash Attention backward kernel."""

  def forward_fn(q, k, v):
    return flash_attention_forward_kernel_wrapper(q, k, v, **kwargs)

  grad_q, grad_k, grad_v = jax.vjp(forward_fn, q, k, v)[1](grad_out)
  return grad_q, grad_k, grad_v


def export_flash_attention_backward_kernel(
    batch_size,
    num_of_heads,
    seq_len,
    head_dim,
    dtype,
    block_size=_BLOCK_SIZE,
    **kwargs,
):
  """Exports a Flash Attention backward kernel as an MLIR module."""

  if batch_size is None:
    # Use symbolic shapes
    (
        batch_size_sym,
        num_of_heads_sym,
        q_seq_len_sym,
        kv_seq_len_sym,
        head_dim_sym,
    ) = jax.export.symbolic_shape(
        "batch_size,num_of_heads,q_seq_len, kv_seq_len,head_dim",
        constraints=(
            # The backward kernel ( verify_block in flash_attention.py)
            # contains python control flow that checks if block_size > seq_len.
            # Without these constraints, JAX export raises
            # InconclusiveDimensionOperation because it cannot determine the
            # result of this comparison for symbolic shapes.
            "batch_size >= 4",
            f"q_seq_len >= {block_size}",
            f"kv_seq_len >= {block_size}",
            f"mod(q_seq_len, {block_size}) == 0",
            f"mod(kv_seq_len, {block_size}) == 0",
        ),
    )

    q_shape = (batch_size_sym, num_of_heads_sym, q_seq_len_sym, head_dim_sym)
    k_shape = (batch_size_sym, num_of_heads_sym, kv_seq_len_sym, head_dim_sym)
    v_shape = (batch_size_sym, num_of_heads_sym, kv_seq_len_sym, head_dim_sym)
    out_shape = (batch_size_sym, num_of_heads_sym, q_seq_len_sym, head_dim_sym)
  else:
    # Defaults if not symbolic
    q_shape = (batch_size, num_of_heads, seq_len, head_dim)
    k_shape = (batch_size, num_of_heads, seq_len, head_dim)
    v_shape = (batch_size, num_of_heads, seq_len, head_dim)
    out_shape = (batch_size, num_of_heads, seq_len, head_dim)

  q_dtype = jax.ShapeDtypeStruct(q_shape, dtype)
  k_dtype = jax.ShapeDtypeStruct(k_shape, dtype)
  v_dtype = jax.ShapeDtypeStruct(v_shape, dtype)
  grad_out_dtype = jax.ShapeDtypeStruct(out_shape, dtype)

  f_p = functools.partial(
      flash_attention_backward_kernel_wrapper, block_size=block_size, **kwargs
  )
  f_j = jax.jit(f_p)
  f_e = jax.export.export(f_j, platforms=["tpu"])
  with pallas_export_experimental(dynamic_shapes=True):
    f_k = f_e(grad_out_dtype, q_dtype, k_dtype, v_dtype)
  return f_k
