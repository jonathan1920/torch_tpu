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
# pylint: skip-file

"""Tests for performance_profiling.py — verifies each executable cell runs without error."""

import os
import torch


def test_profiler_init(device):
  """Cell: import PyTorch profiler and TpuProfilerConfig."""
  from torch.tpu.profiler import TpuProfilerConfig

  assert torch.profiler.profile is not None
  assert TpuProfilerConfig is not None


def test_capture_profile(device):
  """Cell: run training loop inside native profiler context (reduced to 10 steps)."""
  from torch.tpu.profiler import TpuProfilerConfig

  model = (
      torch.nn.Sequential(
          torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
      )
      .to(device)
      .to(torch.bfloat16)
  )

  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

  log_dir = "/tmp/test_profiler_output"

  config = TpuProfilerConfig(
      host_tracer_level=2,
      device_tracer_level=1,
  )

  with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.TPU,
      ],
      experimental_config=config,
      on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
  ) as prof:
    for step in range(10):  # Reduced from 100 for test speed
      optimizer.zero_grad()
      loss = model(data).sum()
      loss.backward()
      optimizer.step()
      torch.accelerator.synchronize()
      prof.step()

  assert os.path.exists(log_dir)


def test_verify_traces():
  """Cell: check that trace files exist in the output directory."""
  log_dir = "/tmp/test_profiler_output"

  if not os.path.exists(log_dir):
    # If previous test didn't create the dir, skip
    import pytest

    pytest.skip("Profiler output directory not found")

  trace_files = os.listdir(log_dir)
  assert len(trace_files) > 0, "No trace files found in profiler output"


def test_kernel_profiling_config():
  """Cell: configure TpuProfilerConfig with experimental kernel profiling options."""
  from torch.tpu.profiler import TpuProfilerConfig

  config = TpuProfilerConfig(
      experimental_options={
          "tpu_enable_periodic_counter_sampling": True,
          "tpu_tc_perf_counter_sampling_options": (
              "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
              " indices:11 indices:56 indices:57 indices:58"
          ),
      }
  )
  assert config is not None
