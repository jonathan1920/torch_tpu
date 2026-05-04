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

"""Stress test for RNG state and random operations."""

import concurrent.futures
from absl.testing import absltest
from absl.testing import parameterized
import torch


class RngStressTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rand", lambda device: torch.rand(100, device=device)),
      ("randn", lambda device: torch.randn(100, device=device)),
      (
          "bernoulli",
          lambda device: torch.full((100,), 0.5, device=device).bernoulli_(),
      ),
      (
          "exponential",
          lambda device: torch.ones(100, device=device).exponential_(),
      ),
      (
          "dropout",
          lambda device: torch.nn.functional.dropout(
              torch.ones(100, device=device), p=0.5
          ),
      ),
      ("random", lambda device: torch.zeros(100, device=device).random_()),
      ("randperm", lambda device: torch.randperm(100, device=device)),
      ("uniform", lambda device: torch.zeros(100, device=device).uniform_()),
  )
  def test_no_rng_state_collision(self, op_fn):
    device = torch.accelerator.current_accelerator()
    num_threads = 10
    num_iterations = 50

    def run_op():
      out = op_fn(device)
      return out.cpu()

    for _ in range(num_iterations):
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_threads
      ) as executor:
        futures = [executor.submit(run_op) for _ in range(num_threads)]
        results = [f.result() for f in futures]

      # Check for collisions (identical results between different threads).
      for i in range(len(results)):
        for j in range(i + 1, len(results)):
          if torch.equal(results[i], results[j]):
            self.fail(
                f"Collision detected between thread {i} and {j} for op {op_fn}"
            )


if __name__ == "__main__":
  absltest.main()
