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

"""Provides base classes and utilities for controlling RNG seeds in tests."""

import random
from typing import Final

from absl.testing import absltest
import torch
from torch.testing._internal import common_utils

DEFAULT_RANDOM_SEED: Final[int] = 1234


def seed_rngs(seed: int) -> None:
  """Seeds Python and PyTorch RNGs with the given seed."""
  # TODO: b/542689633 - Seed NumPy RNG here as well.
  random.seed(seed)
  torch.manual_seed(seed)


class RepeatableTest(common_utils.TestCase):
  """Base class that fixes RNG seeds so tests are reproducible.

  This base class uses a constant RNG seed or from the test_random_seed absl
  flag if provided. It resets the same RNG seed before each test method for
  reproducibility.
  """

  _test_random_seed: int = DEFAULT_RANDOM_SEED

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls._test_random_seed = cls._choose_seed()
    print(f"Repro with --test_random_seed={cls._test_random_seed}", flush=True)

  @classmethod
  def _choose_seed(cls) -> int:
    if absltest.FLAGS["test_random_seed"].present:
      # The user explicitly passed --test_random_seed=N, so we use that value.
      return absltest.FLAGS.test_random_seed
    return DEFAULT_RANDOM_SEED

  def setUp(self) -> None:
    super().setUp()
    # Set the random seed for Python and Torch.
    seed_rngs(self._test_random_seed)
