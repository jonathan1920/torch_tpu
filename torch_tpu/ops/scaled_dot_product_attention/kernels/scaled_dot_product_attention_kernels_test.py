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

import functools
import logging

from absl.testing import absltest
import jax
import jax.numpy as jnp
from torch_tpu.ops.scaled_dot_product_attention.kernels import scaled_dot_product_attention_kernels as kernels

ALL_ONES_Q = False
ALL_ONES_K = False
ALL_ONES_V = False
KERNEL_TYPE = "flash"
USE_DYNAMIC_KERNEL = False


class ScaledDotProductAttentionGenerateTest(absltest.TestCase):
  # pylint: disable=invalid-name
  B = 16
  Hq = 8
  L = 1024
  E = 128
  H = Hq
  S = L
  Ev = E
  # pylint: enable=invalid-name

  def _test_kernel(self, base_fn, test_fn, kernel_type):
    if kernel_type == "flash":
      if ALL_ONES_Q:
        q = jnp.ones(shape=(self.B, self.Hq, self.L, self.E), dtype=jnp.float32)
      else:
        q = jax.random.normal(
            jax.random.PRNGKey(0), shape=(self.B, self.Hq, self.L, self.E)
        )
      if ALL_ONES_K:
        k = jnp.ones(shape=(self.B, self.H, self.S, self.E), dtype=jnp.float32)
      else:
        k = jax.random.normal(
            jax.random.PRNGKey(1), shape=(self.B, self.H, self.S, self.E)
        )
      if ALL_ONES_V:
        v = jnp.ones(shape=(self.B, self.H, self.S, self.Ev), dtype=jnp.float32)
      else:
        v = jax.random.normal(
            jax.random.PRNGKey(2), shape=(self.B, self.H, self.S, self.Ev)
        )
    else:
      q = jax.random.normal(
          jax.random.PRNGKey(0), shape=(self.B, self.Hq, self.L, self.E)
      )
      k = jax.random.normal(
          jax.random.PRNGKey(1), shape=(self.B, self.H, self.S, self.E)
      )
      v = jax.random.normal(
          jax.random.PRNGKey(2), shape=(self.B, self.H, self.S, self.Ev)
      )

    out_base = base_fn(q, k, v)
    out_test = test_fn(q, k, v)
    logging.info("out_base.shape=%s", out_base.shape)
    logging.info("out_base=%s", out_base)
    logging.info("out_test.shape=%s", out_test.shape)
    logging.info("out_test=%s", out_test)

    rtol = 1e-2
    atol = 1e-2
    # abs(out_test - out_base) < atol + rtol * abs(out_base)
    error = jnp.max(jnp.abs(out_test - out_base) - rtol * jnp.abs(out_base))
    assert error < atol, f"{error=} {atol=}"

  def test_forward_torch_ref(self):
    self._test_kernel(
        functools.partial(
            kernels.sdpa_forward_kernel_reference_jax,
            is_causal=True,
        ),
        functools.partial(
            kernels.sdpa_forward_kernel_reference_torch,
            is_causal=True,
        ),
        kernel_type="ref",
    )

  def test_forward_export(self):
    if USE_DYNAMIC_KERNEL:
      static_seq_len = None
      static_head_dim = None
      num_q_heads = None
      batch_size = None
    else:
      static_seq_len = self.L
      static_head_dim = self.E
      num_q_heads = self.Hq
      batch_size = self.B

    is_causal = True
    if KERNEL_TYPE == "flash":
      is_causal = False

    self._test_kernel(
        functools.partial(
            kernels.sdpa_forward_kernel_reference_jax,
            is_causal=is_causal,
        ),
        functools.partial(
            kernels.sdpa_forward_kernel_export_call,
            is_causal=is_causal,
            static_seq_len=static_seq_len,
            static_head_dim=static_head_dim,
            num_q_heads=num_q_heads,
            batch_size=batch_size,
            kernel_type=KERNEL_TYPE,
        ),
        kernel_type=KERNEL_TYPE,
    )


if __name__ == "__main__":
  absltest.main()
