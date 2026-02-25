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

"""Benchmarks for model performance."""

import dataclasses
from typing import Any, Sequence
from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torch_tpu.examples.benchmarks.e2e import benchmark_utils
from torch_tpu.examples.benchmarks.e2e import performance_utils
from torch_tpu.examples.benchmarks.e2e import test_utils
from torch_tpu.shims.g3_multiprocessing import g3_multiprocessing

_ALL_RUN_MODES = (
    benchmark_utils.RunMode.EAGER,
    benchmark_utils.RunMode.OPTIMIZED_EAGER,
    benchmark_utils.RunMode.COMPILED,
)

_LINEAR_LAYER_BENCHMARK_NAME = "linear"
_BATCHNORM1D_LAYER_BENCHMARK_NAME = "batchnorm1d"
_LAYERNORM_LAYER_BENCHMARK_NAME = "layernorm"
_CONV2D_LAYER_BENCHMARK_NAME = "conv2d"
_RMSNORM_LAYER_BENCHMARK_NAME = "rmsnorm"


def generate_run_mode_and_train_configs(
    run_modes: Sequence[Any],
    is_training: Sequence[Any],
):
  """Generates test parameters from a list of run modes and training modes."""
  for training_mode in is_training:
    for run_mode in run_modes:
      name_parts = []
      name_parts.append(f"{run_mode.value}")
      name_parts.append("train" if training_mode else "eval")
      testcase_name = "_".join(name_parts)
      yield dict(
          testcase_name=testcase_name,
          run_mode=run_mode,
          is_training=training_mode,
      )


class LayerPerformanceBenchmarks(test_utils.BenchmarkTest):
  """Tests for end-to-end performance benchmarks."""

  @dataclasses.dataclass
  class _LinearConfig:
    batch_size: int
    seq_len: int
    in_features: int
    out_features: int

  _linear_configs = (
      # Default config for smoke test.
      _LinearConfig(
          batch_size=1,
          seq_len=128,
          in_features=128,
          out_features=128,
      ),
      # Configs for Llama3 70B for MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=8192,
          out_features=28672,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=28672,
          out_features=8192,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=8192,
          out_features=28672,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=28672,
          out_features=8192,
      ),
      # Configs for Qwen3 480B MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=6144,
          out_features=2560,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=2560,
          out_features=6144,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=6144,
          out_features=2560,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=2560,
          out_features=6144,
      ),
      # Configs for Gemma3 27B MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=4608,
          out_features=36864,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=36864,
          out_features=4608,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=4608,
          out_features=36864,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=36864,
          out_features=4608,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_linear_layer(self, run_mode, is_training):
    for layer_config in self._linear_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.Linear",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "in_features": layer_config.in_features,
                  "out_features": layer_config.out_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      microbenchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(
          config, _LINEAR_LAYER_BENCHMARK_NAME, microbenchmark_name
      )

  @dataclasses.dataclass
  class _BatchNormConfig:
    batch_size: int
    seq_len: int
    num_features: int

  _batch_norm_configs = (
      # Default config for smoke test.
      _BatchNormConfig(
          batch_size=1,
          seq_len=128,
          num_features=128,
      ),
      # Larger configs.
      _BatchNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=8192,
      ),
      # High-batch, more TPU friendly shape.
      _BatchNormConfig(
          batch_size=2056,
          seq_len=512,
          num_features=1024,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_batchnorm1d(self, run_mode, is_training):
    # TODO(b/486204316): Fix batchnorm1d training with compiled mode on TPU.
    if run_mode == benchmark_utils.RunMode.COMPILED and is_training:
      self.skipTest(
          "Batchnorm1d in compiled mode with training doesn't stablize in cache"
          " misses.."
      )
    for layer_config in self._batch_norm_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.BatchNorm1d",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      microbenchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(
          config, _BATCHNORM1D_LAYER_BENCHMARK_NAME, microbenchmark_name
      )

  @dataclasses.dataclass
  class _LayerNormConfig:
    batch_size: int
    seq_len: int
    num_features: int
    num_normalized_dims: int = 1

    @property
    def shape(self):
      return (self.batch_size, self.seq_len, self.num_features)

    @property
    def normalized_shape(self):
      return self.shape[-self.num_normalized_dims :]

  _layer_norm_configs = (
      # Default config for smoke test.
      _LayerNormConfig(batch_size=1, seq_len=128, num_features=128),
      # Larger configs.
      _LayerNormConfig(batch_size=32, seq_len=8192, num_features=8192),
      _LayerNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=8192,
          num_normalized_dims=2,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_layernorm(self, run_mode, is_training):
    for layer_config in self._layer_norm_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.LayerNorm",
              custom_kwargs={
                  "normalized_shape": layer_config.normalized_shape,
                  "shape": layer_config.shape,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      microbenchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(
          config, _LAYERNORM_LAYER_BENCHMARK_NAME, microbenchmark_name
      )

  @dataclasses.dataclass
  class _Conv2dConfig:
    batch_size: int
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    height: int
    width: int

  _conv2d_configs = (
      # Default config for smoke test.
      _Conv2dConfig(
          batch_size=1,
          in_channels=2,
          out_channels=4,
          kernel_size=3,
          stride=1,
          padding=1,
          height=128,
          width=128,
      ),
      # Larger configs.
      _Conv2dConfig(
          batch_size=128,
          in_channels=32,
          out_channels=64,
          kernel_size=3,
          stride=1,
          padding=1,
          height=256,
          width=256,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_conv2d(self, run_mode, is_training):
    for layer_config in self._conv2d_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.Conv2d",
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
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      microbenchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(
          config, _CONV2D_LAYER_BENCHMARK_NAME, microbenchmark_name
      )

  @dataclasses.dataclass
  class _RmsNormConfig:
    batch_size: int
    seq_len: int
    num_features: int

  _rms_norm_configs = (
      # Default config for smoke test.
      _RmsNormConfig(
          batch_size=1,
          seq_len=128,
          num_features=128,
      ),
      # Configs for Llama3 70B for MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=8192,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=8192,
      ),
      # Configs for Qwen3 480B MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=6144,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=6144,
      ),
      # Configs for Gemma3 27B MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=4608,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=4608,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_rmsnorm(self, run_mode, is_training):
    for layer_config in self._rms_norm_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.RMSNorm",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_features": layer_config.num_features,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      microbenchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(
          config, _RMSNORM_LAYER_BENCHMARK_NAME, microbenchmark_name
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
