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

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal.utils import utils


class AllocatorReproTest(absltest.TestCase):

  def test_parallel_async_transfers(self):
    device = api.tpu_device()
    num_tensors = 10
    size = 2048  # 2048x2048 int32 = 16MB per tensor, 160MB total

    tpu_tensors = []
    expected_cpus = []

    # 1. Setup Phase
    for _ in range(num_tensors):
      # Generate on CPU first to avoid lazy evaluation differences
      a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
      b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)

      a = a_cpu.to(device)
      b = b_cpu.to(device)
      c = a * 2 + b
      tpu_tensors.append(c)
      expected_cpus.append(a_cpu * 2 + b_cpu)

    # Force materialization (compilation and execution) of the graph
    _ = [t.to('cpu') for t in tpu_tensors]
    torch.tpu.synchronize()

    # Pre-allocate CPU tensors so we can test the direct copy path
    cpu_tensors_async = [
        torch.empty_like(t, device='cpu', pin_memory=True)
        for t in expected_cpus
    ]
    cpu_tensors_sync = [
        torch.empty_like(t, device='cpu', pin_memory=True)
        for t in expected_cpus
    ]

    # Create a dummy CPU workload that takes roughly ~20ms to execute.
    dummy_cpu_tensor_a = torch.randn(2000, 2000, device='cpu')
    dummy_cpu_tensor_b = torch.randn(2000, 2000, device='cpu')

    def do_cpu_work():
      _ = dummy_cpu_tensor_a @ dummy_cpu_tensor_b

    # 2. Async Transfers WITH Overlapped CPU Work
    t0 = time.time()
    for t_src, t_dst in zip(tpu_tensors, cpu_tensors_async):
      t_dst.copy_(t_src, non_blocking=True)

    # CPU does heavy work WHILE the TPU is DMA transferring data in the
    # background
    do_cpu_work()

    torch.tpu.synchronize()
    overlapped_async_time = time.time() - t0

    # 3. Synchronous (Blocking) Transfers WITH Sequential CPU Work
    t2 = time.time()
    for t_src, t_dst in zip(tpu_tensors, cpu_tensors_sync):
      t_dst.copy_(t_src, non_blocking=False)

    # CPU is blocked until transfers finish, so this work happens sequentially
    do_cpu_work()

    sequential_sync_time = time.time() - t2

    print(f'\n--- Overlap Performance Results ({num_tensors} x 16MB) ---')
    print(f'Async (Overlapped Work) Total Time : {overlapped_async_time:.5f}s')
    print(f'Sync (Sequential Work) Total Time  : {sequential_sync_time:.5f}s')
    print('-----------------------------------------\n')

    # If non_blocking=True allows true overlapping, the overlapped total time
    # should be significantly faster than doing the transfer and CPU work
    # sequentially.
    self.assertLess(overlapped_async_time, sequential_sync_time)

    # Verify data correctness
    for cpu_t, expected in zip(cpu_tensors_async, expected_cpus):
      utils.assert_close(cpu_t, expected)

    for cpu_t, expected in zip(cpu_tensors_sync, expected_cpus):
      utils.assert_close(cpu_t, expected)

  def test_to_cpu_non_blocking(self):
    device = api.tpu_device()

    # Generate on CPU first to avoid lazy evaluation generating different
    # numbers!
    size = 2048
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)

    a = a_cpu.to(device)
    b = b_cpu.to(device)

    # Do some compute
    c = a * 2 + b

    # Non-blocking transfer to CPU. With our custom _to_copy implementation,
    # this will now use pinned memory and the fast async DMA path.
    cpu_tensor = c.to('cpu', non_blocking=True)

    self.assertEqual(cpu_tensor.device.type, 'cpu')

    torch.tpu.synchronize()

    # Now verify the values.
    expected_cpu_tensor = a_cpu * 2 + b_cpu

    utils.assert_close(cpu_tensor, expected_cpu_tensor)

  def test_pin_memory(self):
    device = api.tpu_device()
    # Initialize the TPU device by creating a small tensor on it
    _ = torch.zeros(1, device=device)

    t = torch.randn(100)

    # The default pin_memory() will use the current device's pinned
    # allocator if a custom backend like TPU is registered as the accelerator.
    t_pinned = t.pin_memory()

    self.assertTrue(t_pinned.is_pinned())
    utils.assert_close(t, t_pinned)


if __name__ == '__main__':
  absltest.main()
