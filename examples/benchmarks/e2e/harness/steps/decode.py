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

import torch
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness.steps import common
from transformers import cache_utils


class DecodeStepper(common.BaseStepper):

  def __init__(self, output_tokens: int = 16):
    super().__init__()
    self.output_tokens = output_tokens
    self.input_ids = None
    self.batch_size = None
    self.prompt_len = None
    self.head_dim = None
    self.num_heads = None
    self.device = None
    self.dtype = None
    self.cache = None
    self.next_token = None
    self.prefill_next_token = None
    self.current_pos = None
    self._decode_model = None

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, *args, **kwargs
  ) -> None:
    del input_args
    super().init_with_benchmark_args(model, *args, **kwargs)
    assert self._model is not None
    self.input_ids = input_kwargs["input_ids"]
    self.batch_size = self.input_ids.shape[0]
    self.prompt_len = self.input_ids.shape[1]

    model_config = getattr(self._model, "config")  # pytype: disable=attribute-error
    self.head_dim = model_config.head_dim
    self.num_heads = model_config.num_key_value_heads
    self.device = self.input_ids.device
    self.dtype = (
        self._model.dtype if hasattr(self._model, "dtype") else torch.bfloat16
    )

    self.cache = cache_utils.StaticCache(
        config=model_config,
        max_cache_len=self.prompt_len + self.output_tokens,
        batch_size=self.batch_size,
        dtype=self.dtype,
        device=self.device,
    )
    self.cache.early_initialization(
        batch_size=self.batch_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dtype=self.dtype,
        device=self.device,
    )
    self.next_token = None
    self.prefill_next_token = None
    self.current_pos = None
    self._decode_model = self._model

  # Compiles only the inner model used for decode.
  def compile(
      self,
      compile_config: compile_lib.CompileConfig,
      target: target_lib.Target,
  ) -> None:
    # We ignore compile_config.scope and always compile the model for decoding
    self._decode_model = compile_config.apply(self._model, target)

  @staticmethod
  def _step_function(
      m,
      next_token,
      step_kwargs,
      cache,
  ):
    return m(
        input_ids=next_token,
        past_key_values=cache,
        **step_kwargs,
    )

  # Always eager prefill and save state for future decode.
  def pre_warmup_init(self):
    assert self._model is not None
    with torch.no_grad():
      prefill_output = self._model(
          input_ids=self.input_ids,
          past_key_values=self.cache,
          use_cache=True,
      )
    self.cache = prefill_output.past_key_values
    prefill_logits = prefill_output.logits
    self.next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).unsqueeze(
        -1
    )
    self.prefill_next_token = self.next_token.clone()
    self.current_pos = self.prompt_len

  # In-place reset of cache to return it to its post-prefill state.
  # This is used to facilitate warmup decode runs to warm the compilation cache w/o post warmup recompilation.
  def post_warmup_hook(self):
    assert self.prefill_next_token is not None
    assert self.cache is not None
    assert self.prompt_len is not None
    self.next_token = self.prefill_next_token.clone()

    def _reset_internal(cache, prompt_len):
      for layer in cache.layers:
        if hasattr(layer, "keys"):
          layer.keys[:, :, prompt_len:, :].zero_()
        if hasattr(layer, "values"):
          layer.values[:, :, prompt_len:, :].zero_()
        if hasattr(layer, "cumulative_length"):
          if isinstance(layer.cumulative_length, int):
            layer.cumulative_length = prompt_len
          else:
            layer.cumulative_length.fill_(prompt_len)

    _reset_internal(self.cache, self.prompt_len)
    self.current_pos = self.prompt_len

  def get_cache(self):
    return self.cache

  def get_step_fn(self) -> step_lib.StepFn:

    def step_fn():
      with torch.no_grad():
        decode_output = None
        for _ in range(self.output_tokens):
          cache_position = torch.tensor([self.current_pos], device=self.device)
          decode_output = self._inner_stepper(
              self._decode_model,
              self.next_token,
              {"use_cache": True, "cache_position": cache_position},
              self.cache,
          )
          logits = decode_output.logits
          self.next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
          assert self.current_pos is not None
          self.current_pos += 1
        return decode_output

    return step_fn


@step_lib.register_stepper(step_lib.StepperType.DECODER_ONLY_DECODE)
def decode(
    output_tokens: int = 16,
) -> step_lib.Stepper:
  """Stateful decode step: prefill, reset, and decode iterations."""
  return DecodeStepper(output_tokens=output_tokens)
