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

"""Tests for mingpt_threshold."""

from absl.testing import absltest
from examples.mingpt import mingpt_runner_lib
from torch_tpu._internal.shims.pyglib import build_data

# Mingpt latency with async compile is ~0.005 seconds on VLP 1x1 at the time of
# writing this test. We set the threshold to 0.006 seconds to leave a buffer of
# 20% for potential latency increase and avoid flakiness. While this won't be
# able to catch gradual latency regressions, it will catch anything major.
#
# We only test the functionality for debug builds, which can be much slower.
THRESHOLD_SEC = 30 if build_data.BuildDebugMode() else 0.006


class MingptThresholdTest(absltest.TestCase):

  def test_mingpt_threshold(self):
    _, latency = mingpt_runner_lib.run_mingpt_with_default_params()
    self.assertLess(latency, THRESHOLD_SEC)


if __name__ == "__main__":
  absltest.main()
