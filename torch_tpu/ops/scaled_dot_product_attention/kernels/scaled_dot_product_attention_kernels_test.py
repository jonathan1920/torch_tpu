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


class ScaledDotProductAttentionGenerateTest(absltest.TestCase):

  def _test_kernel(self, base_fn, test_fn):
    # pylint: disable=invalid-name
    B = 1
    Hq = 2
    H = 2
    L = 2
    S = 4
    E = 2
    Ev = 2
    # pylint: enable=invalid-name

    q = jax.random.normal(jax.random.PRNGKey(0), shape=(B, Hq, L, E))
    k = jax.random.normal(jax.random.PRNGKey(1), shape=(B, H, S, E))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(B, H, S, Ev))

    out_base = base_fn(q, k, v)
    out_test = test_fn(q, k, v)
    logging.info("out_base.shape=%s", out_base.shape)
    logging.info("out_base=%s", out_base)
    logging.info("out_test.shape=%s", out_test.shape)
    logging.info("out_test=%s", out_test)

    tol = 1e-5
    error = jnp.max(jnp.abs(out_test - out_base))
    assert error < tol, f"{error=} {tol=}"

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
    )

  def test_forward_export(self):
    self._test_kernel(
        functools.partial(
            kernels.sdpa_forward_kernel_reference_jax,
            is_causal=True,
        ),
        functools.partial(
            kernels.sdpa_forward_kernel_export_call,
            is_causal=True,
        ),
    )


if __name__ == "__main__":
  absltest.main()
