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

import time
from typing import Any

from absl import logging
import flax
import flax.nnx
import jax
import jax.numpy as jnp
from examples.benchmarks.e2e import benchmark_utils as pt_benchmark_utils
from examples.benchmarks.e2e import performance_utils as pt_performance_utils
from examples.benchmarks.e2e.jax import model_utils

from torch_tpu._internal.shims.xprof import traceme


def _run_jax_forward_pass(
    model: flax.nnx.Module,
    inputs: Any,
    run_mode: pt_benchmark_utils.RunMode,
) -> tuple[float, float]:
  model.eval()
  if pt_benchmark_utils.is_torch_compile(run_mode):
    runnable_model = flax.nnx.jit(model)
  else:
    runnable_model = model

  def sync_jax_device(x):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), x)

  # Warmup
  warmup_start = time.time()
  for i in range(pt_benchmark_utils.MIN_WARMUP_STEPS):
    with traceme.TraceMe("Warmup", step_num=i):
      out = runnable_model(inputs)
      sync_jax_device(out)
  warmup_end = time.time()
  warmup_time = (
      warmup_end - warmup_start
  ) / pt_benchmark_utils.MIN_WARMUP_STEPS

  # Eval
  eval_start = time.time()
  for i in range(pt_benchmark_utils.POST_WARMUP_STEPS):
    with traceme.TraceMe("Eval", step_num=i):
      out = runnable_model(inputs)
      sync_jax_device(out)
  eval_end = time.time()
  eval_time = (eval_end - eval_start) / pt_benchmark_utils.POST_WARMUP_STEPS

  return warmup_time, eval_time


def _run_jax_backward_pass(
    model: flax.nnx.Module,
    inputs: Any,
    run_mode: pt_benchmark_utils.RunMode,
) -> tuple[float, float]:
  model.train()

  def grad_step(model, x):
    def loss_fn(model, x):
      y_pred = model(x)
      if isinstance(y_pred, tuple):
        return jnp.mean(y_pred[0])
      return jnp.mean(y_pred)

    loss_val, grads = flax.nnx.value_and_grad(loss_fn)(model, x)
    return loss_val, grads

  if pt_benchmark_utils.is_torch_compile(run_mode):
    runnable_step = flax.nnx.jit(grad_step)
  else:
    runnable_step = grad_step

  runnable_model = lambda x: runnable_step(model, x)

  def sync_jax_device(x):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), x)

  # Warmup
  warmup_start = time.time()
  for i in range(pt_benchmark_utils.MIN_WARMUP_STEPS):
    with traceme.TraceMe("Warmup", step_num=i):
      loss_val, grads = runnable_model(inputs)
      sync_jax_device(loss_val)
      sync_jax_device(grads)
  warmup_end = time.time()
  warmup_time = (
      warmup_end - warmup_start
  ) / pt_benchmark_utils.MIN_WARMUP_STEPS

  # Eval
  eval_start = time.time()
  for i in range(pt_benchmark_utils.POST_WARMUP_STEPS):
    with traceme.TraceMe("Train", step_num=i):
      loss_val, grads = runnable_model(inputs)
      sync_jax_device(loss_val)
      sync_jax_device(grads)
  eval_end = time.time()
  eval_time = (eval_end - eval_start) / pt_benchmark_utils.POST_WARMUP_STEPS

  return warmup_time, eval_time


def run_benchmark(
    config: pt_performance_utils.PerformanceBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  jax.clear_caches()

  device = jax.devices()[0]

  args = config.model_and_input_args
  model_and_input = model_utils.get_model_and_input(
      args.model_name, args.batch_size, args.sequence_length, args.custom_kwargs
  )

  model = model_and_input.model
  inputs = jax.device_put(model_and_input.inputs, device)

  if config.is_training and args.model_name not in ["nonzero", "topk"]:
    # BW pass
    warmup_time, eval_time = _run_jax_backward_pass(
        model, inputs, config.run_mode
    )
  else:
    # FW pass
    warmup_time, eval_time = _run_jax_forward_pass(
        model, inputs, config.run_mode
    )

  logging.info(
      "Test: %s, benchmark: %s, microbenchmark: %s, run_mode: %s,"
      " is_training: %s, warmup_time (seconds): %s, eval_time (seconds): %s",
      test_method_name,
      benchmark_name,
      microbenchmark_name,
      config.run_mode.value,
      config.is_training,
      warmup_time,
      eval_time,
  )
