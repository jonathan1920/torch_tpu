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
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torch_tpu.ops.scaled_dot_product_attention.kernels import scaled_dot_product_attention_kernels as kernels

ALL_ONES_Q = False
ALL_ONES_K = False
ALL_ONES_V = False
KERNEL_TYPE = "flash"
USE_DYNAMIC_KERNEL = True


class ScaledDotProductAttentionGenerateTest(parameterized.TestCase):
  # pylint: disable=invalid-name
  B = 16
  Hq = 16
  L = 1024
  E = 128
  H = 2
  S = L
  Ev = E
  # pylint: enable=invalid-name

  def _test_kernel(self, base_fn, test_fn, kernel_type, dtype=jnp.float32):
    if kernel_type in ["flash", "splash"]:
      if ALL_ONES_Q:
        q = jnp.ones(shape=(self.B, self.Hq, self.L, self.E), dtype=dtype)
      else:
        q = jax.random.normal(
            jax.random.PRNGKey(0), shape=(self.B, self.Hq, self.L, self.E)
        ).astype(dtype)
      if ALL_ONES_K:
        k = jnp.ones(shape=(self.B, self.H, self.S, self.E), dtype=dtype)
      else:
        k = jax.random.normal(
            jax.random.PRNGKey(1), shape=(self.B, self.H, self.S, self.E)
        ).astype(dtype)
      if ALL_ONES_V:
        v = jnp.ones(shape=(self.B, self.H, self.S, self.Ev), dtype=dtype)
      else:
        v = jax.random.normal(
            jax.random.PRNGKey(2), shape=(self.B, self.H, self.S, self.Ev)
        ).astype(dtype)
    else:
      q = jax.random.normal(
          jax.random.PRNGKey(0), shape=(self.B, self.Hq, self.L, self.E)
      ).astype(dtype)
      k = jax.random.normal(
          jax.random.PRNGKey(1), shape=(self.B, self.H, self.S, self.E)
      ).astype(dtype)
      v = jax.random.normal(
          jax.random.PRNGKey(2), shape=(self.B, self.H, self.S, self.Ev)
      ).astype(dtype)

    out_base = base_fn(q, k, v)
    out_test = test_fn(q, k, v)
    logging.info("out_base.shape=%s", jax.tree.map(jnp.shape, out_base))
    logging.info("out_test.shape=%s", jax.tree.map(jnp.shape, out_test))

    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            a,
            b,
            atol=5e-2,
            rtol=5e-2,
        ),
        out_test,
        out_base,
    )

  def test_forward_export_jax(self):
    self._test_kernel(
        functools.partial(
            kernels.SDPAKernelReferenceTorch.forward,
            is_causal=True,
        ),
        kernels.SDPAKernelReferenceJax.export_forward(
            dtype=jnp.float32, is_causal=True
        ).call,
        kernel_type="ref",
    )

  def get_input_sizes(self):
    if USE_DYNAMIC_KERNEL:
      input_sizes = None
    else:
      input_sizes = kernels.InputSizes(
          batch_size=self.B,
          q_num_heads=self.Hq,
          kv_num_heads=self.H,
          q_seq_len=self.L,
          kv_seq_len=self.S,
          qk_head_dim=self.E,
          v_head_dim=self.Ev,
      )
    return input_sizes

  def test_forward_export_flash(self):
    input_sizes = self.get_input_sizes()

    is_causal_list = [True, False]

    for is_causal in is_causal_list:
      logging.info("Running test_forward_export with is_causal=%s", is_causal)
      self._test_kernel(
          functools.partial(
              kernels.SDPAKernelReferenceTorch.forward,
              is_causal=is_causal,
          ),
          kernels.SDPAKernelFlashAttention.export_forward(
              dtype=jnp.float32,
              is_causal=is_causal,
              input_sizes=input_sizes,
          ).call,
          kernel_type=KERNEL_TYPE,
      )

  def test_backward_torch_ref(self):
    grad_out = jax.random.normal(
        jax.random.PRNGKey(42), shape=(self.B, self.Hq, self.L, self.Ev)
    )
    self._test_kernel(
        functools.partial(
            kernels.SDPAKernelReferenceJax.backward,
            grad_out,
            is_causal=True,
        ),
        functools.partial(
            kernels.SDPAKernelReferenceTorch.backward,
            grad_out,
            is_causal=True,
        ),
        kernel_type="ref",
    )

  @parameterized.parameters(
      (True, jnp.float32),
      (False, jnp.float32),
      (True, jnp.bfloat16),
      (False, jnp.bfloat16),
  )
  def test_forward_export_splash(self, is_causal, dtype):
    # pylint: disable-next=invalid-name
    self.H = self.Hq  # Splash only supports MHA
    input_sizes = self.get_input_sizes()

    logging.info(
        "Running test_forward_export_splash with dtype=%s is_causal=%s",
        dtype,
        is_causal,
    )
    self._test_kernel(
        functools.partial(
            kernels.SDPAKernelReferenceTorch.forward,
            is_causal=is_causal,
        ),
        kernels.SDPAKernelSplashAttention.export_forward(
            dtype=dtype,
            is_causal=is_causal,
            input_sizes=input_sizes,
        ).call,
        kernel_type="splash",
        dtype=dtype,
    )

  def test_backward_export(self):
    input_sizes = self.get_input_sizes()

    is_causal_list = [True, False]

    grad_out = jax.random.normal(
        jax.random.PRNGKey(42), shape=(self.B, self.Hq, self.L, self.Ev)
    )

    for is_causal in is_causal_list:
      logging.info("Running test_backward_export with is_causal=%s", is_causal)
      self._test_kernel(
          functools.partial(
              kernels.SDPAKernelReferenceTorch.backward,
              grad_out,
              is_causal=is_causal,
          ),
          functools.partial(
              kernels.SDPAKernelFlashAttention.export_backward(
                  dtype=jnp.float32,
                  is_causal=is_causal,
                  input_sizes=input_sizes,
              ).call,
              grad_out,
          ),
          kernel_type=KERNEL_TYPE,
      )


if __name__ == "__main__":
  absltest.main()
