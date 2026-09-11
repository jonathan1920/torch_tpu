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

"""Unit tests for detect_causal_mask pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import detect_causal_mask
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class DetectCausalMaskTest(seed_test_utils.RepeatableTest):

  def test_is_causal_mask_bool(self):
    seq_len = 8
    causal_mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool)
    )
    self.assertTrue(detect_causal_mask._is_causal_mask(causal_mask))

    # Invert to make non-causal
    non_causal = ~causal_mask
    self.assertFalse(detect_causal_mask._is_causal_mask(non_causal))

    # Mask with a padded token
    padded_mask = causal_mask.clone()
    padded_mask[0, 0, :, 0] = False
    self.assertFalse(detect_causal_mask._is_causal_mask(padded_mask))

  def test_is_causal_mask_float(self):
    seq_len = 8
    lower_tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

    float_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    float_mask[0, 0, ~lower_tri] = float("-inf")
    self.assertTrue(detect_causal_mask._is_causal_mask(float_mask))

    # With finite negative value
    finite_neg_mask = torch.zeros(
        (1, 1, seq_len, seq_len), dtype=torch.bfloat16
    )
    finite_neg_mask[0, 0, ~lower_tri] = -65504.0
    self.assertTrue(detect_causal_mask._is_causal_mask(finite_neg_mask))

    # All zeros (non-causal, bidirectional)
    zero_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    self.assertFalse(detect_causal_mask._is_causal_mask(zero_mask))

  def test_detect_and_eliminate_in_native_sdpa(self):
    seq_len = 16
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    # Slice the mask as Hugging Face transformers does
    sliced_m = graph.call_function(
        torch.ops.aten.slice.Tensor, (m, 3, 0, seq_len)
    )
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, sliced_m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        mask,
    ]

    detect_causal_mask.apply(gm, example_inputs)

    # Check that SDPA args were updated: attn_mask=None, is_causal=True
    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIsNone(node.args[3])
    self.assertTrue(node.args[5])

    # Check that slice node was eliminated
    slice_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor
    ]
    self.assertEmpty(slice_nodes)

  def test_detect_and_eliminate_in_overrideable_fwd_and_bwd(self):
    seq_len = 16
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")
    grad_out = graph.placeholder("grad_out")

    # Forward
    fwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        (q, k, v, m, 0.0, False, False),
    )
    out = graph.call_function(torch.ops.aten.select.int, (fwd, 0, 0))
    lse = graph.call_function(torch.ops.aten.select.int, (fwd, 0, 1))

    # Backward
    cum_seq = graph.placeholder("cum_seq")
    seed = graph.placeholder("seed")
    offset = graph.placeholder("offset")
    bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (
            grad_out,
            q,
            k,
            v,
            m,
            [True, True, True, False],
            out,
            lse,
            cum_seq,
            cum_seq,
            0,
            0,
            0.0,
            False,
            seed,
            offset,
        ),
    )
    graph.output(bwd)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        mask,
        torch.randn(1, 4, seq_len, 32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    ]

    detect_causal_mask.apply(gm, example_inputs)

    # Verify forward node
    fwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ]
    self.assertLen(fwd_nodes, 1)
    fwd_node = fwd_nodes[0]
    self.assertIsNone(fwd_node.args[3])
    self.assertTrue(fwd_node.args[5])

    # Verify backward node
    bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default
    ]
    self.assertLen(bwd_nodes, 1)
    bwd_node = bwd_nodes[0]
    # args[4] should be an empty tensor call, args[13] should be True
    self.assertTrue(bwd_node.args[13])
    self.assertEqual(
        bwd_node.args[4].target, torch.ops.aten.empty.memory_format
    )

  def test_non_causal_mask_preserved(self):
    seq_len = 16
    # An arbitrary attention mask (e.g. diagonal only)
    mask = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        mask,
    ]

    detect_causal_mask.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    # Mask should NOT be eliminated
    self.assertIsNotNone(node.args[3])
    self.assertFalse(node.args[5])

  def test_detect_and_eliminate_in_efficient_fwd_and_bwd(self):
    seq_len = 16
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")
    grad_out = graph.placeholder("grad_out")

    # Forward Efficient: (query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal)
    fwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        (q, k, v, m, True, 0.0, False),
    )
    out = graph.call_function(torch.ops.aten.select.int, (fwd, 0, 0))
    lse = graph.call_function(torch.ops.aten.select.int, (fwd, 0, 1))

    # Backward Efficient: (grad_out, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal)
    seed = graph.placeholder("seed")
    offset = graph.placeholder("offset")
    bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
        (
            grad_out,
            q,
            k,
            v,
            m,
            out,
            lse,
            seed,
            offset,
            0.0,
            [True, True, True, False],
            False,
        ),
    )
    graph.output(bwd)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        mask,
        torch.randn(1, 4, seq_len, 32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    ]

    detect_causal_mask.apply(gm, example_inputs)

    # Verify forward node
    fwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_efficient_attention.default
    ]
    self.assertLen(fwd_nodes, 1)
    fwd_node = fwd_nodes[0]
    self.assertIsNone(fwd_node.args[3])
    self.assertTrue(fwd_node.args[6])

    # Verify backward node
    bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_efficient_attention_backward.default
    ]
    self.assertLen(bwd_nodes, 1)
    bwd_node = bwd_nodes[0]
    self.assertTrue(bwd_node.args[11])
    self.assertEqual(
        bwd_node.args[4].target, torch.ops.aten.empty.memory_format
    )

  def test_meta_tensor_handled_gracefully(self):
    seq_len = 16
    meta_mask = torch.empty(
        (1, 1, seq_len, seq_len), device="meta", dtype=torch.bool
    )
    self.assertFalse(detect_causal_mask._is_causal_mask(meta_mask))

  def test_detect_and_eliminate_with_kwargs(self):
    seq_len = 16
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    # Pass attn_mask in kwargs and omit is_causal
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v),
        {"attn_mask": m},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        torch.randn(1, 4, seq_len, 32),
        mask,
    ]

    detect_causal_mask.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIsNone(node.kwargs.get("attn_mask"))
    self.assertTrue(node.kwargs.get("is_causal"))

  def test_numerical_correctness_sdpa_bool_mask(self):
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    # 1. Forward and backward on original module (with explicit mask)
    out_orig = gm(q_val, k_val, v_val, mask)
    out_orig.sum().backward()
    grad_q_orig = q_val.grad.clone()
    grad_k_orig = k_val.grad.clone()
    grad_v_orig = v_val.grad.clone()

    # Reset gradients
    q_val.grad.zero_()
    k_val.grad.zero_()
    v_val.grad.zero_()

    # 2. Apply detect_causal_mask pass
    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), mask]
    detect_causal_mask.apply(gm, example_inputs)

    # 3. Forward and backward on transformed module (is_causal=True, mask=None)
    out_transformed = gm(q_val, k_val, v_val, mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()
    grad_k_transformed = k_val.grad.clone()
    grad_v_transformed = v_val.grad.clone()

    # 4. Verify numerical equivalence
    utils.assert_close(out_transformed, out_orig)
    utils.assert_close(grad_q_transformed, grad_q_orig)
    utils.assert_close(grad_k_transformed, grad_k_orig)
    utils.assert_close(grad_v_transformed, grad_v_orig)

  def test_numerical_correctness_sdpa_float_neginf_mask(self):
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    lower_tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    mask[0, 0, ~lower_tri] = float("-inf")

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    out_orig = gm(q_val, k_val, v_val, mask)
    out_orig.sum().backward()
    grad_q_orig = q_val.grad.clone()
    grad_k_orig = k_val.grad.clone()
    grad_v_orig = v_val.grad.clone()

    q_val.grad.zero_()
    k_val.grad.zero_()
    v_val.grad.zero_()

    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), mask]
    detect_causal_mask.apply(gm, example_inputs)

    out_transformed = gm(q_val, k_val, v_val, mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()
    grad_k_transformed = k_val.grad.clone()
    grad_v_transformed = v_val.grad.clone()

    utils.assert_close(out_transformed, out_orig)
    utils.assert_close(grad_q_transformed, grad_q_orig)
    utils.assert_close(grad_k_transformed, grad_k_orig)
    utils.assert_close(grad_v_transformed, grad_v_orig)

  def test_numerical_correctness_sdpa_float_finite_neg_mask(self):
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    lower_tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    mask[0, 0, ~lower_tri] = -65504.0

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    out_orig = gm(q_val, k_val, v_val, mask)
    out_orig.sum().backward()
    grad_q_orig = q_val.grad.clone()
    grad_k_orig = k_val.grad.clone()
    grad_v_orig = v_val.grad.clone()

    q_val.grad.zero_()
    k_val.grad.zero_()
    v_val.grad.zero_()

    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), mask]
    detect_causal_mask.apply(gm, example_inputs)

    out_transformed = gm(q_val, k_val, v_val, mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()
    grad_k_transformed = k_val.grad.clone()
    grad_v_transformed = v_val.grad.clone()

    # exp(-65504.0) is effectively 0 in float32, so outputs match closely
    utils.assert_close(out_transformed, out_orig, atol=1e-4, rtol=1e-4)
    utils.assert_close(grad_q_transformed, grad_q_orig, atol=1e-4, rtol=1e-4)
    utils.assert_close(grad_k_transformed, grad_k_orig, atol=1e-4, rtol=1e-4)
    utils.assert_close(grad_v_transformed, grad_v_orig, atol=1e-4, rtol=1e-4)

  def test_numerical_correctness_sliced_causal_mask(self):
    full_seq_len = 32
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    full_mask = torch.tril(
        torch.ones((1, 1, full_seq_len, full_seq_len), dtype=torch.bool)
    )

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sliced_m = graph.call_function(
        torch.ops.aten.slice.Tensor, (m, 2, 0, seq_len)
    )
    sliced_m2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (sliced_m, 3, 0, seq_len)
    )
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, sliced_m2, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    out_orig = gm(q_val, k_val, v_val, full_mask)
    out_orig.sum().backward()
    grad_q_orig = q_val.grad.clone()
    grad_k_orig = k_val.grad.clone()
    grad_v_orig = v_val.grad.clone()

    q_val.grad.zero_()
    k_val.grad.zero_()
    v_val.grad.zero_()

    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), full_mask]
    detect_causal_mask.apply(gm, example_inputs)

    out_transformed = gm(q_val, k_val, v_val, full_mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()
    grad_k_transformed = k_val.grad.clone()
    grad_v_transformed = v_val.grad.clone()

    utils.assert_close(out_transformed, out_orig)
    utils.assert_close(grad_q_transformed, grad_q_orig)
    utils.assert_close(grad_k_transformed, grad_k_orig)
    utils.assert_close(grad_v_transformed, grad_v_orig)

  def test_numerical_correctness_across_dtypes(self):
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
      with self.subTest(dtype=dtype):
        graph = fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        m = graph.placeholder("mask")

        sdpa = graph.call_function(
            torch.ops.aten.scaled_dot_product_attention.default,
            (q, k, v, m, 0.0, False),
        )
        graph.output(sdpa)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        q_val = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype
        )
        k_val = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype
        )
        v_val = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype
        )

        out_orig = gm(q_val, k_val, v_val, mask)

        example_inputs = [q_val, k_val, v_val, mask]
        detect_causal_mask.apply(gm, example_inputs)

        out_transformed = gm(q_val, k_val, v_val, mask)
        utils.assert_close(out_transformed, out_orig)

  def test_numerical_correctness_non_causal_preservation(self):
    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    # Non-causal diagonal identity mask
    mask = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("mask")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    out_orig = gm(q_val, k_val, v_val, mask)
    out_orig.sum().backward()
    grad_q_orig = q_val.grad.clone()

    q_val.grad.zero_()
    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), mask]
    detect_causal_mask.apply(gm, example_inputs)

    out_transformed = gm(q_val, k_val, v_val, mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()

    utils.assert_close(out_transformed, out_orig)
    utils.assert_close(grad_q_transformed, grad_q_orig)

  def test_numerical_correctness_nn_module_with_fx_trace(self):
    class CausalAttention(torch.nn.Module):

      def forward(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False
        )

    seq_len = 16
    batch_size = 2
    num_heads = 4
    head_dim = 32

    mod = CausalAttention()
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))

    q_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    k_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )
    v_val = torch.randn(
        batch_size, num_heads, seq_len, head_dim, requires_grad=True
    )

    out_ref = mod(q_val, k_val, v_val, mask)
    out_ref.sum().backward()
    grad_q_ref = q_val.grad.clone()
    grad_k_ref = k_val.grad.clone()
    grad_v_ref = v_val.grad.clone()

    q_val.grad.zero_()
    k_val.grad.zero_()
    v_val.grad.zero_()

    gm = fx.symbolic_trace(mod)
    example_inputs = [q_val.detach(), k_val.detach(), v_val.detach(), mask]
    detect_causal_mask.apply(gm, example_inputs)

    out_transformed = gm(q_val, k_val, v_val, mask)
    out_transformed.sum().backward()
    grad_q_transformed = q_val.grad.clone()
    grad_k_transformed = k_val.grad.clone()
    grad_v_transformed = v_val.grad.clone()

    utils.assert_close(out_transformed, out_ref)
    utils.assert_close(grad_q_transformed, grad_q_ref)
    utils.assert_close(grad_k_transformed, grad_k_ref)
    utils.assert_close(grad_v_transformed, grad_v_ref)


if __name__ == "__main__":
  absltest.main()
