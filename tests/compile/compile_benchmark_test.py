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

"""Benchmark tests for TorchTPU compile mode."""

import concurrent.futures
import functools
import json
import time
from urllib import parse

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu import api
from torch_tpu._internal.utils import log_utils
from transformers.models.qwen3 import configuration_qwen3
from transformers.models.qwen3 import modeling_qwen3

from torch_tpu._internal.shims.xprof import traceme
from torch_tpu._internal.shims.xprof import xprof_analysis_client
from torch_tpu._internal.shims.xprof import xprof_session

log_utils.log_to_stderr()


_NUM_WARMUP_STEPS = flags.DEFINE_integer(
    "num_warmup_steps", 5, "Numer of warm-up steps."
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 20, "Numer of post-warm up steps."
)


def _get_base_qwen3_config():
  # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  config = configuration_qwen3.Qwen3Config.from_dict({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 12288,
      "max_position_embeddings": 40960,
      "max_window_layers": 36,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 4,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_scaling": None,
      "rope_theta": 1000000,
      "sliding_window": None,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.51.0",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 151936,
  })
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  return config


@functools.cache
def _get_tpu_duty_cycle(xprof_url: str) -> float:
  parsed_url = parse.urlparse(xprof_url)
  session_id = parse.parse_qs(parsed_url.query).get("session_id", [None])[0]
  if not session_id:
    session_id = parsed_url.path.strip("/").split("/")[-1]
  if not session_id:
    raise RuntimeError(
        f"Could not parse session_id from xprof_url: {xprof_url}"
    )
  xprof = xprof_analysis_client.XprofAnalysisClient()
  result = xprof.get_profile_data(
      "overview_page.json", session_id=session_id, format="json"
  )
  content = (
      result[1] if (isinstance(result, tuple) and len(result) > 1) else result
  )
  if not content:
    raise RuntimeError(
        f"Failed to retrieve overview_page.json data from {xprof_url}"
    )
  overview_data = json.loads(content.decode("utf-8"))
  for section in overview_data:
    p_dict = section.get("p", {})
    if "device_duty_cycle_percent" in p_dict:
      return float(str(p_dict["device_duty_cycle_percent"]).rstrip("%"))
  raise RuntimeError(
      "device_duty_cycle_percent not found in overview sections."
  )


class CompileBenchmarkTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logs = []
    torch.manual_seed(42)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    for log in sorted(cls.logs):
      logging.info(log)

  def _append_log(self, prefix, batch_size, seq_len, time_per_step, xprof_url):
    try:
      duty_cycle = _get_tpu_duty_cycle(xprof_url)
      duty_str = f"{duty_cycle:.2f}%"
    except Exception:  # pylint: disable=broad-except
      duty_str = "N/A"
    self.logs.append(
        f"{prefix}: Run for bs={batch_size}, seq={seq_len}: took"
        f" {time_per_step * 1000:.2f} ms per step, TPU Duty Cycle:"
        f" {duty_str}, XProf URL: {xprof_url}"
    )

  @parameterized.parameters(
      (1, 256),
  )
  def test_eager_forward(self, batch_size, seq_len):
    device = api.tpu_device()
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Model(config).to(device)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    for _ in range(_NUM_WARMUP_STEPS.value):
      out = model(input_ids)
      out = out.last_hidden_state.cpu()
      del out

    session = xprof_session.XprofSession()
    session.start_session(host_trace_level=3, enable_python_tracer=True)

    loop_start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval"):
        out = model(input_ids)
        out = out.last_hidden_state.cpu()
        del out
    time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
    xprof_url = session.end_session_and_get_url()

    self._append_log("Eager", batch_size, seq_len, time_per_step, xprof_url)

  @parameterized.parameters(
      (1, 256),
  )
  def test_compile_forward(self, batch_size, seq_len):
    device = api.tpu_device()
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Model(config).to(device)
    model.eval()

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    for _ in range(_NUM_WARMUP_STEPS.value):
      out = compiled_model(input_ids)
      out = out.last_hidden_state.cpu()
      del out

    session = xprof_session.XprofSession()
    session.start_session(host_trace_level=3, enable_python_tracer=True)

    loop_start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval"):
        out = compiled_model(input_ids)
        out = out.last_hidden_state.cpu()
        del out
    time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
    xprof_url = session.end_session_and_get_url()

    self._append_log("Compile", batch_size, seq_len, time_per_step, xprof_url)

  @parameterized.parameters(
      (1, 256),
  )
  def test_per_layer_compile_forward(self, batch_size, seq_len):
    device = api.tpu_device()
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Model(config).to(device)
    model.eval()

    for i in range(len(model.layers)):
      model.layers[i] = torch.compile(
          model.layers[i],
          backend="tpu",
          dynamic=False,
      )

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.inference_mode():
      for _ in range(_NUM_WARMUP_STEPS.value):
        out = model(input_ids)
        out = out.last_hidden_state.cpu()
        del out

      session = xprof_session.XprofSession()
      session.start_session(host_trace_level=3, enable_python_tracer=True)

      loop_start_time = time.time()
      for _ in range(_NUM_STEPS.value):
        with traceme.TraceMe("Eval"):
          out = model(input_ids)
          out = out.last_hidden_state.cpu()
          del out
      time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
      xprof_url = session.end_session_and_get_url()

    self._append_log(
        "Per-layer Compile",
        batch_size,
        seq_len,
        time_per_step,
        xprof_url,
    )

  @parameterized.parameters(
      (1, 256),
  )
  def test_compile_with_async_copy(self, batch_size, seq_len):
    device = api.tpu_device()
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Model(config).to(device)
    model.eval()

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    all_inputs = [
        torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        for _ in range(_NUM_STEPS.value)
    ]
    pinned_out = torch.empty(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )

    def copy_fn(out):
      pinned_out.copy_(out.last_hidden_state, non_blocking=False)

    # Async Copy Strategy:
    # 1. The main thread exclusively handles dispatching the TPU operations
    #     to the device.
    # 2. The background thread exclusively handles the Device-to-Host (D2H)
    #     transfer via `.copy_()`.
    # 3. Using pinned memory avoids an extra host-side buffer copy, enabling
    #    direct DMA offload.
    with torch.inference_mode():
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(_NUM_WARMUP_STEPS.value):
          out = compiled_model(all_inputs[0])
          executor.submit(copy_fn, out)
        torch.tpu.synchronize()

        session = xprof_session.XprofSession()
        session.start_session(host_trace_level=3, enable_python_tracer=True)

        loop_start_time = time.time()
        for i in range(_NUM_STEPS.value):
          with traceme.TraceMe("Eval"):
            out = compiled_model(all_inputs[i])
            executor.submit(copy_fn, out)
        torch.tpu.synchronize()

      time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
      xprof_url = session.end_session_and_get_url()

    self._append_log(
        "Compile with Async Copy",
        batch_size,
        seq_len,
        time_per_step,
        xprof_url,
    )

    duty_cycle = _get_tpu_duty_cycle(xprof_url)
    self.assertGreater(
        duty_cycle,
        90.0,
        f"TPU Duty Cycle should be > 90%, got {duty_cycle:.2f}% (xprof_url:"
        f" {xprof_url})",
    )

  @parameterized.parameters(
      (1, 256),
  )
  def test_compile_with_synced_copy(self, batch_size, seq_len):
    device = api.tpu_device()
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Model(config).to(device)
    model.eval()

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    all_inputs = [
        torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        for _ in range(_NUM_STEPS.value)
    ]
    pinned_out = torch.empty(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )

    with torch.inference_mode():
      for _ in range(_NUM_WARMUP_STEPS.value):
        out = compiled_model(all_inputs[0])
        pinned_out.copy_(out.last_hidden_state)
        del out
      torch.tpu.synchronize()

      session = xprof_session.XprofSession()
      session.start_session(host_trace_level=3, enable_python_tracer=True)

      loop_start_time = time.time()
      for i in range(_NUM_STEPS.value):
        with traceme.TraceMe("Eval"):
          out = compiled_model(all_inputs[i])
          pinned_out.copy_(out.last_hidden_state)
          del out
      torch.tpu.synchronize()
      time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
      xprof_url = session.end_session_and_get_url()

    self._append_log(
        "Compile with Synced Copy",
        batch_size,
        seq_len,
        time_per_step,
        xprof_url,
    )


if __name__ == "__main__":
  absltest.main()
