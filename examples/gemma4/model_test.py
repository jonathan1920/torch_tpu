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

"""Tests for Gemma4 model implementation."""

from absl.testing import absltest
import torch
from examples.gemma4 import model


class Gemma4ModelTest(absltest.TestCase):

  def test_dense_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_moe_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        enable_moe=True,
        num_experts=4,
        top_k_experts=2,
        expert_dim=64,
        moe_dense_hidden_dim=64,
    )

    m = model.Gemma4Model(config).to('tpu')
    input_ids = torch.randint(0, config.vocab_size, (2, 10), device='tpu')

    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_multimodal_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vision_proj_dim=64,
        audio_proj_dim=64,
    )
    m = model.Gemma4Model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    pixel_values = torch.randn(2, 5, 64)
    audio_features = torch.randn(2, 5, 64)

    output = m(
        input_ids, pixel_values=pixel_values, audio_features=audio_features
    )

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_sliding_attention_cpu_fallback(self):
    config = model.Gemma4Config(
        num_hidden_layers=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )
    attn = model.Gemma4Attention(config, layer_idx=0)
    hidden_states = torch.randn(2, 32, config.hidden_size)
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)

    # On CPU, query_states.device.type is 'cpu'. use_splash must be False.
    output, _ = attn(hidden_states, position_ids)
    self.assertEqual(output.shape, (2, 32, config.hidden_size))

  def test_sliding_attention_with_custom_attention_mask(self):
    config = model.Gemma4Config(
        num_hidden_layers=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )
    attn = model.Gemma4Attention(config, layer_idx=0)
    hidden_states = torch.randn(2, 32, config.hidden_size)
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
    attention_mask = torch.zeros(2, 1, 32, 32)

    # When attention_mask is passed, use_splash must be False.
    output, _ = attn(hidden_states, position_ids, attention_mask=attention_mask)
    self.assertEqual(output.shape, (2, 32, config.hidden_size))


if __name__ == '__main__':
  absltest.main()
