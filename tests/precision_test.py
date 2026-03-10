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

import threading

from absl.testing import absltest
import torch
from torch_tpu import api
import torch_tpu._internal.precision as p

precision = p.precision
Precision = p.Precision
p_impl = p.precision_impl


class PrecisionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    api.tpu_device()

  def test_precision_exports_to_torch_module(self):
    self.assertTrue(torch.tpu.precision)
    self.assertTrue(torch.tpu.Precision)

    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.highest)

  def test_context_manager(self):
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)
    with precision(Precision.HIGH):
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.high)
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)

    with precision(Precision.HIGHEST):
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.highest)
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)

  def test_nested_context_manager(self):
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)
    with precision(Precision.HIGH):
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.high)
      with precision(Precision.HIGHEST):
        self.assertEqual(p_impl._get_precision(), p_impl.Precision.highest)
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.high)
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)

  def test_exception_handling(self):
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)
    try:
      with precision(Precision.HIGH):
        self.assertEqual(p_impl._get_precision(), p_impl.Precision.high)
        raise ValueError("Test Exception")
    except ValueError:
      pass
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)

  def test_matmul_runs(self):
    a = torch.randn(10, 10, device="tpu")
    b = torch.randn(10, 10, device="tpu")

    with precision(Precision.DEFAULT):
      c = torch.matmul(a, b)
      self.assertEqual(c.shape, (10, 10))

    with precision(Precision.HIGH):
      d = torch.matmul(a, b)
      self.assertEqual(d.shape, (10, 10))

    with precision(Precision.HIGHEST):
      e = torch.matmul(a, b)
      self.assertEqual(e.shape, (10, 10))

  def test_thread_local(self):
    # Set precision in main thread
    self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)

    def worker_high():
      with precision(Precision.HIGH):
        self.assertEqual(p_impl._get_precision(), p_impl.Precision.high)

    def worker_highest():
      # Start without any context, should default to default
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)
      with precision(Precision.HIGHEST):
        self.assertEqual(p_impl._get_precision(), p_impl.Precision.highest)

    with precision(Precision.DEFAULT):
      t1 = threading.Thread(target=worker_high)
      t2 = threading.Thread(target=worker_highest)

      t1.start()
      t2.start()

      t1.join()
      t2.join()

      # The main thread should still be default
      self.assertEqual(p_impl._get_precision(), p_impl.Precision.default)


if __name__ == "__main__":
  absltest.main()
