# Copyright 2025 Google LLC
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

# TODO(b/474644492): generate a metric into a class, and add this as a metric
# for quality benchmark.

"""Perplexity score computation for quality benchmarks."""

import torch
import torch.nn.functional as F
from examples.benchmarks.quality_utils import quality_benchmark_model


def _get_text_chunks(text, max_text_chunk_size: int) -> list[str]:
  """Splits the text into chunks of the given length."""
  return [
      text[i : i + max_text_chunk_size]
      for i in range(0, len(text), max_text_chunk_size)
  ]


def perplexity_score(
    pred_logits: torch.Tensor, target: torch.Tensor, target_len: int
) -> torch.Tensor:
  """Computes perplexity score for the given sample output and data.

  Simplified implementation of
  https://lightning.ai/docs/torchmetrics/stable/gallery/text/perplexity.html.
  TODO(b/462811452): Currently there is an issue with the torchmetrics library
   where it generates a segmentation fault during the test. Once this
  is fixed, we should use the torchmetrics library instead of this custom
  implementation.

  Args:
    pred_logits: Sample output from the model
    target: Target data
    target_len: The length of the original target.

  Returns:
    Perplexity score.
  """
  ce = F.cross_entropy(pred_logits, target, reduction="none")
  loss = ce[:target_len].sum() / target_len
  return torch.exp(loss)


class PerplexityMetric(quality_benchmark_model.MetricProducer):
  """Perplexity metric for quality benchmarks."""

  def __init__(self, max_text_chunk_size: int):
    self._max_text_chunk_size = max_text_chunk_size

  def get_name(self) -> str:
    return "perplexity"

  def assess(
      self,
      model_input: str,
      benchmark_model: quality_benchmark_model.QualityBenchmarkModel,
  ) -> torch.Tensor:
    """Runs perplexity score on a single sample.

    This function is used to run perplexity score on a single sample. It will
    split the input text into chunks of the given length and run perplexity
    score on each chunk. The perplexity scores are then averaged to get the
    final score.

    Args:
      model_input: The text to run perplexity score on.
      benchmark_model: The benchmark model to use for encoding the text.

    Returns:
      The perplexity score for the given text.
    """
    ppl_avg = []
    for text_chunk in _get_text_chunks(model_input, self._max_text_chunk_size):
      # Encode arbitrary text.
      formatted_input = benchmark_model.format(text_chunk)

      if formatted_input.unpadded_length < 2:
        continue

      pred_logits, target = benchmark_model.get_logits_and_targets(
          formatted_input
      )

      # Compute perplexity score.
      ppl_avg.append(
          perplexity_score(
              pred_logits, target, formatted_input.unpadded_length - 1
          )
      )

    if not ppl_avg:
      raise ValueError("No valid text chunks to compute perplexity.")
    return torch.stack(ppl_avg).mean()
