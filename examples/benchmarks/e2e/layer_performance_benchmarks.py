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
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils
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
      # Configs for BERT
      _LinearConfig(
          batch_size=32,
          seq_len=128,
          in_features=768,
          out_features=768,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=128,
          in_features=768,
          out_features=3072,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=128,
          in_features=3072,
          out_features=768,
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
      # BERT configs
      _LayerNormConfig(
          batch_size=32,
          seq_len=128,
          num_features=768,
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
  class _EmbeddingConfig:
    batch_size: int
    seq_len: int
    num_embeddings: int
    embedding_dim: int

  _embedding_configs = (
      # Default config for smoke test.
      _EmbeddingConfig(
          batch_size=1,
          seq_len=128,
          num_embeddings=128,
          embedding_dim=128,
      ),
      # Configs for BERT
      _EmbeddingConfig(
          batch_size=32,
          seq_len=128,
          num_embeddings=30522,
          embedding_dim=768,
      ),
      _EmbeddingConfig(
          batch_size=1,
          seq_len=128,
          num_embeddings=512,
          embedding_dim=768,
      ),
      _EmbeddingConfig(
          batch_size=32,
          seq_len=128,
          num_embeddings=2,
          embedding_dim=768,
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_embedding(self, run_mode, is_training):
    for layer_config in self._embedding_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.Embedding",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
              custom_kwargs={
                  "num_embeddings": layer_config.num_embeddings,
                  "embedding_dim": layer_config.embedding_dim,
              },
          ),
      )
      config_dict = dataclasses.asdict(layer_config)
      name_parts = [
          f"{k}_{str(v).replace('torch.', '')}" if k == "dtype" else f"{k}_{v}"
          for k, v in config_dict.items()
      ]
      benchmark_name = "_".join(name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @dataclasses.dataclass
  class _DropoutConfig:
    p: float
    shape: tuple[int, ...]

  _dropout_configs = (
      # BERT configs
      _DropoutConfig(
          p=0.1,
          shape=(32, 128, 768),
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_dropout(self, run_mode, is_training):
    for layer_config in self._dropout_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.Dropout",
              custom_kwargs={
                  "p": layer_config.p,
                  "shape": layer_config.shape,
              },
          ),
      )
      # Extracting details from shape config for name building
      custom_name_parts = []
      custom_name_parts.append(f"p_{layer_config.p}")
      custom_name_parts.append(
          f"shape_{'x'.join((str(x) for x in layer_config.shape))}"
      )
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @dataclasses.dataclass
  class _TanhConfig:
    shape: tuple[int, ...]

  _tanh_configs = (
      # BERT configs
      _TanhConfig(
          shape=(32, 768),
      ),
  )

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_tanh(self, run_mode, is_training):
    for layer_config in self._tanh_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="nn.Tanh",
              custom_kwargs={
                  "shape": layer_config.shape,
              },
          ),
      )
      # Extracting details from shape config for name building
      custom_name_parts = []
      custom_name_parts.append(
          f"shape_{'x'.join((str(x) for x in layer_config.shape))}"
      )
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @dataclasses.dataclass
  class _BertLayerConfig:
    batch_size: int
    seq_len: int

  _bert_layer_configs = (_BertLayerConfig(batch_size=32, seq_len=128),)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(False, True)
      )
  )
  def test_gelu_activation(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="GELUActivation",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  # TODO(b/484415655): Known bert training issue.
  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_layer(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertLayer",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_self_output(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertSelfOutput",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_intermediate(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertIntermediate",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_output(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertOutput",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_pooler(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertPooler",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_bert_embeddings(self, run_mode, is_training):
    for layer_config in self._bert_layer_configs:
      config = performance_utils.PerformanceBenchmarkConfig(
          supported_platforms=[
              benchmark_utils.Platform.GFC_1X1X1,
              benchmark_utils.Platform.B200_1,
          ],
          benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
          run_mode=run_mode,
          is_training=is_training,
          model_and_input_args=performance_utils.ModelAndInputArgs(
              model_name="BertEmbeddings",
              batch_size=layer_config.batch_size,
              sequence_length=layer_config.seq_len,
          ),
      )
      custom_name_parts = [
          f"batch_size_{layer_config.batch_size}",
          f"seq_len_{layer_config.seq_len}",
      ]
      benchmark_name = "_".join(custom_name_parts)
      self.run_performance_benchmark_test(config, benchmark_name)

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

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_qwen3_attention(self, run_mode, is_training):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3Attention",
            batch_size=1,
            sequence_length=128,
            custom_kwargs={},
        ),
    )
    self.run_performance_benchmark_test(config, f"Qwen3Attention")

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_qwen3_rms_norm(self, run_mode, is_training):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3RMSNorm",
            batch_size=1,
            sequence_length=128,
            custom_kwargs={"hidden_size": 128},
        ),
    )
    self.run_performance_benchmark_test(config, f"Qwen3RMSNorm")

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(
          _ALL_RUN_MODES, is_training=(True, False)
      )
  )
  def test_qwen3_mlp(self, run_mode, is_training):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3MLP",
            batch_size=1,
            sequence_length=128,
            custom_kwargs={"hidden_size": 128, "intermediate_size": 512},
        ),
    )
    self.run_performance_benchmark_test(config, f"Qwen3MLP")

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_silu_activation(self, run_mode, is_training):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="SiLUActivation",
            batch_size=1,
            sequence_length=128,
            custom_kwargs={"shape": (1, 128, 512)},
        ),
    )
    self.run_performance_benchmark_test(config, f"SiLUActivation")

  @parameterized.named_parameters(
      generate_run_mode_and_train_configs(_ALL_RUN_MODES, is_training=(False,))
  )
  def test_qwen3_rotary_embedding(self, run_mode, is_training):
    self.skipTest("TODO(b/484415655): Investigate cache miss.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3RotaryEmbedding",
            batch_size=1,
            sequence_length=128,
            custom_kwargs={"head_dim": 128},
        ),
    )
    self.run_performance_benchmark_test(config, f"Qwen3RotaryEmbedding")


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
