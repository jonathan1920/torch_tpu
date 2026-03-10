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

"""Utilities for running end-to-end performance benchmarks."""

import contextlib
import dataclasses
import functools
import os
from typing import Any, Callable, Mapping, Optional, Sequence

from absl import flags
from absl import logging
import torch
from torch import distributed as dist
# from torch.google import distributed as g3_distributed
from torch_tpu import api
from torch_tpu._internal.distributed import gpu_env
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import device_utils
from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import model_utils

from torch_tpu._internal.shims.xprof import xprof_analysis_client


WEIGHTS_DTYPE = flags.DEFINE_string(
    "weights_dtype",
    "bfloat16",
    "The dtype to use for model weights.",
)

ENABLE_XPROF = flags.DEFINE_bool(
    "enable_xprof",
    False,
    "Whether to enable xprof profiling for post warmup run.",
)

DISTRIBUTED_PLATFORMS = (
    benchmark_utils.Platform.GFC_2X2X1,
    benchmark_utils.Platform.B200_4,
)


@functools.lru_cache(maxsize=None)
def get_xprof_client() -> xprof_analysis_client.XprofAnalysisClient | None:
  """Returns the xprof analysis client.

  This function caches the client to avoid creating multiple connections.
  """
  return xprof_analysis_client.XprofAnalysisClient()


@dataclasses.dataclass
class ModelAndInputArgs:
  """Args to get the model and example inputs.

  Attributes:
    model_name: The name of the model to benchmark.
    sequence_length: The sequence length of the input.
    batch_size: The batch size of the input.
    custom_kwargs: Custom kwargs that might be needed to get the model and
      example inputs.
  """

  model_name: Optional[str] = None
  sequence_length: Optional[int] = None
  batch_size: Optional[int] = None
  custom_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


# LINT.IfChange
@dataclasses.dataclass
class PerformanceBenchmarkConfig:
  """The config for a performance benchmark.

  Attributes:
    supported_platforms: The platforms the benchmark supports. If the current
      platform is not in this sequence, the test will be skipped.
    benchmark_category: The category of the benchmark. This defines how to get
      the model and example inputs and the benchmark function to run.
    run_mode: The mode to run the benchmark in. This is used to set environment
      variables like `TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE` or use torch
      compile.
    is_training: Whether the benchmark is for training. If True, the benchmark
      will use the training mode of the model and will run the optimizer.
    model_and_input_args: The args to get the model and example inputs.
    grad_accumulation_steps: The number of steps to accumulate gradients for.
  """

  supported_platforms: Sequence[benchmark_utils.Platform]
  benchmark_category: benchmark_utils.BenchmarkCategory
  run_mode: benchmark_utils.RunMode
  is_training: bool
  model_and_input_args: ModelAndInputArgs
  grad_accumulation_steps: int = 4
# LINT.ThenChange(../../../g3doc/benchmarking.md)

@contextlib.contextmanager
def _run_mode_context(run_mode: benchmark_utils.RunMode, device: torch.device):
  """Context manager to configure the environment for different run modes.

  This includes adjusting environment variables like
  `TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE` and performing necessary cleanups
  like clearing caches and resetting torch.dynamo.

  Args:
    run_mode: The benchmark_utils.RunMode to configure the context for.
    device: The torch device being used.

  Yields:
    None
  """
  original_env_var_value = os.environ.get(
      "TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE", ""
  )
  if run_mode == benchmark_utils.RunMode.OPTIMIZED_EAGER:
    os.environ["TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE"] = "optimized"
  try:
    yield
  finally:
    # Change back to the original value.
    os.environ["TORCH_TPU_INTERNAL_EAGER_COMPILATION_MODE"] = (
        original_env_var_value
    )
    # pylint: disable=protected-access
    device_utils.clear_cache(device.type)
    if benchmark_utils.is_torch_compile(run_mode):
      torch._dynamo.reset()


# Not using keyword arguments here because the google version of torchrun
# doesn't support passing keyword arguments to the worker function.
def _run_single_process_benchmark(
    rank: int,
    world_size: int,
    config: PerformanceBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config.

  Args:
    rank: The rank of the process. This is used to run the benchmark in
      distributed mode.
    world_size: The size of the world. This is used to run the benchmark in
      distributed mode.
    config: The benchmark config. Contains the arguments to run the benchmark.
    test_method_name: The name of the test method. This is used for logging the
      benchmark results.
    benchmark_name: The name of the benchmark. This combined with
      test_method_name is used to uniquely identify the benchmark in MLCompass.
  """
  platform = benchmark_utils.PLATFORM.value
  device = benchmark_utils.get_torch_device(platform)
  # Seed random number generators for reproducibility. This should be done after
  # initializing the device.
  benchmark_utils.seed_rngs()
  weights_dtype = _get_torch_dtype(WEIGHTS_DTYPE.value)
  func = _get_benchmark_function(config, device)
  model_and_input = _get_model_and_input(
      config.model_and_input_args,
      config.benchmark_category,
      device,
      weights_dtype,
      config.is_training,
      benchmark_utils.is_torch_compile(config.run_mode),
  )
  optimizer = _get_optimizer(
      model_and_input.model,
      is_training=config.is_training,
      use_torch_compile=benchmark_utils.is_torch_compile(config.run_mode),
  )
  # Only enable xprof for rank 0 process.
  enable_xprof = ENABLE_XPROF.value and rank == 0

  xprof_client = None
  if enable_xprof:
    xprof_client = get_xprof_client()

  with _run_mode_context(config.run_mode, device):
    result = benchmark_utils.run_performance_benchmark(
        func,
        model_and_input.model,
        model_and_input.example_inputs,
        device,
        enable_xprof=enable_xprof,
        optimizer=optimizer,
        xprof_client=xprof_client,
    )
    logging.info(
        "Test: %s, benchmark: %s, microbenchmark: %s, platform: %s,"
        " weights_dtype: %s, rank(0-indexed): %s, world_size: %s,"
        " num_warmup_steps: %s, first_step_time (seconds): %s, warmup_overhead"
        " (seconds): %s, average_step_time (seconds): %s, peak_device_memory"
        " (MB): %s, e2e_wall_time (seconds): %s",
        test_method_name,
        benchmark_name,
        microbenchmark_name,
        benchmark_utils.PLATFORM.value,
        WEIGHTS_DTYPE.value,
        rank,
        world_size,
        result.num_warmup_steps,
        result.first_step_time_seconds,
        result.warmup_overhead_seconds,
        result.post_warmup_step_time_seconds,
        result.peak_device_memory_mb,
        result.e2e_wall_time_seconds,
    )
  # Only export results from the rank 0 process to avoid duplicate entries in
  # MLCompass.
  if benchmark_utils.MLCOMPASS_TRACKING_ID.value and rank == 0:
    mlcompass_utils.export_to_mlcompass(
        platform,
        result,
        benchmark_utils.BASE_CL.value,
        benchmark_utils.MLCOMPASS_TRACKING_ID.value,
        benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )


def _run_torch_tpu_worker(
    config: PerformanceBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
):
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  _run_single_process_benchmark(
      rank,
      world_size,
      config,
      test_method_name,
      benchmark_name,
      microbenchmark_name,
  )
  dist.destroy_process_group()


def _run_distributed_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config."""
  if benchmark_utils.PLATFORM.value == benchmark_utils.Platform.GFC_2X2X1:
    singlehost_wrapper.prepare_tpu_environment()
    run_worker = singlehost_wrapper.tpu_env_wrapper(
        _run_torch_tpu_worker,
        world_size=8,
    )
    dist.torchrun(
        run_worker,
        nproc_per_node=8,
    )(
        config,
        test_method_name,
        benchmark_name,
        microbenchmark_name,
    )
  elif benchmark_utils.PLATFORM.value == benchmark_utils.Platform.B200_4:
    # A single B200 device is roughly equivalent to two GFC devices,
    # so double the batch size on B200 to make a fairer comparison.
    config.model_and_input_args.batch_size = (
        config.model_and_input_args.batch_size * 2
    )
    dist.torchrun(
        gpu_env.run_in_workers,
        nproc_per_node=4,
    )(
        _run_single_process_benchmark,
        config,
        test_method_name,
        benchmark_name,
        microbenchmark_name,
    )


def run_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config.

  Args:
    config: The benchmark config. Contains the arguments to run the benchmark.
    test_method_name: The name of the test method. This is used for logging the
      benchmark results.
    benchmark_name: The name of the benchmark. This combined with
      test_method_name is used to uniquely identify the benchmark in MLCompass.
    microbenchmark_name: This is used to export microbenchmark results to
      MLCompass when a benchmark test method is composed of multiple
      microbenchmarks. See go/mlcompass-microbenchmark-guide for more details.
  """

  platform = benchmark_utils.PLATFORM.value
  if platform in DISTRIBUTED_PLATFORMS:
    _run_distributed_benchmark(
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )
  else:
    _run_single_process_benchmark(
        0,
        1,
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )


def _get_optimizer(
    model: torch.nn.Module, *, is_training: bool, use_torch_compile: bool
) -> torch.optim.Optimizer | None:
  """Returns optimizer for training, or None for inference."""
  if not is_training:
    return None
  return torch.optim.AdamW(
      model.parameters(),
      lr=0.1,  # Gigantic LR for testing.
      capturable=use_torch_compile,
      # The non-fused version of adam will expand foreach into loops and
      # increase the graph size. The compile time increase is especially
      # obvious when compiling AdamW with torch.compile.
      # We should consider implementing 'aten::_fused_adamw_' to both
      # improve compile time and increase performance.
      fused=False,
  )


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
  if dtype_str == "bfloat16":
    return torch.bfloat16
  elif dtype_str == "float16":
    return torch.float16
  elif dtype_str == "float32":
    return torch.float32
  else:
    raise ValueError(f"Unknown model dtype: {dtype_str}")


def _get_benchmark_function(
    config: PerformanceBenchmarkConfig,
    device: torch.device,
) -> Callable[
    [torch.nn.Module, Any, torch.optim.Optimizer | None],
    Any,
]:
  """Returns callable for the given benchmark category and is_training.

  Args:
    config: The benchmark config.
    device: The torch device being used.

  Returns:
    The benchmark function. The returned function takes in the model, example
    inputs, and optimizer (if is_training is True) and returns output of the
    model.
  """
  if (
      config.benchmark_category
      == benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM
  ):
    if not config.is_training:
      return benchmark_function_db.huggingface_llm_forward_pass
    else:
      return benchmark_function_db.get_huggingface_llm_training_function(
          device,
          benchmark_utils.is_torch_compile(config.run_mode),
          config.grad_accumulation_steps,
      )
  elif (
      config.benchmark_category == benchmark_utils.BenchmarkCategory.META_LLAMA
      and not config.is_training
  ):
    return benchmark_function_db.meta_llama_forward_pass
  elif config.benchmark_category == benchmark_utils.BenchmarkCategory.ML_LAYER:
    if not config.is_training:
      return benchmark_function_db.ml_layer_forward_pass
    else:
      return benchmark_function_db.get_ml_layer_train_step_function(
          device, benchmark_utils.is_torch_compile(config.run_mode)
      )
  else:
    raise ValueError(
        "No benchmark function found for category:"
        f" {config.benchmark_category}, is_training: {config.is_training}"
    )


def _get_model_and_input(
    model_and_input_args: ModelAndInputArgs,
    benchmark_category: benchmark_utils.BenchmarkCategory,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
    use_torch_compile: bool,
) -> model_utils.ModelAndInput:
  """Returns the benchmark model and example input for the given args.

  Args:
    model_and_input_args: The model and input args.
    benchmark_category: The benchmark category. Defines how to get the model and
      example inputs.
    device: The torch device. The model and example inputs will be loaded on
      this device.
    weights_dtype: The model dtype. The model weights will be loaded with this
      dtype.
    is_training: Whether the model is in training mode or eval mode.
    use_torch_compile: Whether to wrap the model with torch.compile.

  Returns:
    ModelAndInput dataclass containing the model and example inputs.
  """
  if benchmark_category == benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM:
    dist_strat_str = model_and_input_args.custom_kwargs.get(
        "dist_strat", "none"
    )
    dist_strat = model_utils.DistStrat(dist_strat_str)
    return model_utils.get_huggingface_llm_model(
        model_and_input_args.model_name,
        device=device,
        weights_dtype=weights_dtype,
        is_training=is_training,
        use_torch_compile=use_torch_compile,
        sequence_length=model_and_input_args.sequence_length,
        batch_size=model_and_input_args.batch_size,
        dist_strat=dist_strat,
    )
  elif benchmark_category == benchmark_utils.BenchmarkCategory.META_LLAMA:
    return model_utils.get_meta_llama_model(
        model_and_input_args.model_name,
        device,
        weights_dtype,
        use_torch_compile,
        model_and_input_args.sequence_length,
        model_and_input_args.batch_size,
    )
  elif benchmark_category == benchmark_utils.BenchmarkCategory.ML_LAYER:
    return model_utils.get_ml_layer_model(
        model_name=model_and_input_args.model_name,
        device=device,
        weights_dtype=weights_dtype,
        is_training=is_training,
        use_torch_compile=use_torch_compile,
        batch_size=model_and_input_args.batch_size,
        sequence_length=model_and_input_args.sequence_length,
        **model_and_input_args.custom_kwargs,
    )
  else:
    raise ValueError(f"Unknown benchmark category: {benchmark_category}")
