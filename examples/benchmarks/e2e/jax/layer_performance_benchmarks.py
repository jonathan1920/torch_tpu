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

import dataclasses
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from examples.benchmarks.e2e import benchmark_utils as pt_benchmark_utils
from examples.benchmarks.e2e import layer_configs
from examples.benchmarks.e2e import performance_utils as pt_performance_utils
from examples.benchmarks.e2e import test_utils
from examples.benchmarks.e2e.jax import performance_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_ALL_RUN_MODES = (
    pt_benchmark_utils.RunMode.EAGER,
    pt_benchmark_utils.RunMode.COMPILED,
)

_LINEAR_LAYER_BENCHMARK_NAME = "linear_jax"
_BATCHNORM1D_LAYER_BENCHMARK_NAME = "batchnorm1d_jax"
_LAYERNORM_LAYER_BENCHMARK_NAME = "layernorm_jax"
_CONV2D_LAYER_BENCHMARK_NAME = "conv2d_jax"
_RMSNORM_LAYER_BENCHMARK_NAME = "rmsnorm_jax"
_SDPA_BENCHMARK_NAME = "sdpa_jax"


class LayerPerformanceBenchmarks(parameterized.TestCase):
  """Tests for end-to-end performance benchmarks for JAX layers."""

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_linear_layer(self, run_mode, is_training):
    for layer_config in layer_configs.LINEAR_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.Linear",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "in_features": layer_config.in_features,
                  "out_features": layer_config.out_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _LINEAR_LAYER_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_batchnorm1d(self, run_mode, is_training):
    # TODO(b/486204316): Fix batchnorm1d training with compiled mode on TPU.
    if run_mode == pt_benchmark_utils.RunMode.COMPILED and is_training:
      self.skipTest(
          "Batchnorm1d in compiled mode with training doesn't stablize in cache"
          " misses.."
      )
    for layer_config in layer_configs.BATCH_NORM_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.BatchNorm1d",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _BATCHNORM1D_LAYER_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_layernorm(self, run_mode, is_training):
    for layer_config in layer_configs.LAYER_NORM_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.LayerNorm",
              custom_kwargs={
                  "normalized_shape": layer_config.normalized_shape,
                  "shape": layer_config.shape,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _LAYERNORM_LAYER_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_conv2d(self, run_mode, is_training):
    for layer_config in layer_configs.CONV2D_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.Conv2d",
              batch_size=layer_config.batch_size,
              custom_kwargs={
                  "in_channels": layer_config.in_channels,
                  "out_channels": layer_config.out_channels,
                  "kernel_size": layer_config.kernel_size,
                  "stride": layer_config.stride,
                  "padding": layer_config.padding,
                  "height": layer_config.height,
                  "width": layer_config.width,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _CONV2D_LAYER_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_rmsnorm(self, run_mode, is_training):
    for layer_config in layer_configs.RMS_NORM_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.RMSNorm",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _RMSNORM_LAYER_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_sdpa(self, run_mode, is_training):
    for layer_config in layer_configs.SDPA_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.sdpa",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_heads": layer_config.num_heads,
                  "head_dim": layer_config.head_dim,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          _SDPA_BENCHMARK_NAME,
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_topk(self, run_mode, is_training):
    for layer_config in layer_configs.TOPK_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.topk",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
                  "k": layer_config.k,
                  "dim": layer_config.dim,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          "topk",
          microbenchmark_name,
      )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_nonzero(self, run_mode, is_training):
    for layer_config in layer_configs.NONZERO_CONFIGS:
      config = pt_performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              pt_benchmark_utils.Platform.GFC_1X1X1,
              pt_benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=pt_benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=pt_performance_utils.ModelAndInputArgs(
              model_name="nnx.nonzero",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [f"{k}_{v}" for k, v in config_dict.items()]
      microbenchmark_name = "_".join(name_parts)
      performance_utils.run_benchmark(
          config,
          self._testMethodName,
          "nonzero",
          microbenchmark_name,
      )


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
