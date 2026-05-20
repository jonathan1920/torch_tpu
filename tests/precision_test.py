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

import concurrent.futures
import threading
import unittest

from absl.testing import absltest
import torch
import torch_tpu._internal.precision as p
from torch_tpu._internal.utils import utils


precision = p.precision
Precision = p.Precision
p_impl = p.precision_impl


class PrecisionTest(absltest.TestCase):

  def test_precision_exports_to_torch_module(self):
    self.assertTrue(torch.tpu.precision)
    self.assertTrue(torch.tpu.Precision)

    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      self.assertEqual(p_impl._get_precision(), Precision.HIGHEST)

  def test_context_manager(self):
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)
    with precision(Precision.HIGH):
      self.assertEqual(p_impl._get_precision(), Precision.HIGH)
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

    with precision(Precision.HIGHEST):
      self.assertEqual(p_impl._get_precision(), Precision.HIGHEST)
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

  def test_nested_context_manager(self):
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)
    with precision(Precision.HIGH):
      self.assertEqual(p_impl._get_precision(), Precision.HIGH)
      with precision(Precision.HIGHEST):
        self.assertEqual(p_impl._get_precision(), Precision.HIGHEST)
      self.assertEqual(p_impl._get_precision(), Precision.HIGH)
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

  def test_exception_handling(self):
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)
    try:
      with precision(Precision.HIGH):
        self.assertEqual(p_impl._get_precision(), Precision.HIGH)
        raise ValueError("Test Exception")
    except ValueError:
      pass
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

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

  def test_precision_in_shlo(self):
    device = torch.device("tpu")
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)

    def model(a, b):
      return torch.matmul(a, b)

    with precision(Precision.DEFAULT):
      self.assertIn(
          "precision = [DEFAULT, DEFAULT]",
          utils.format_model(model, a, b, shlo=True),
      )
      self.assertEqual(model(a, b).shape, (10, 10))

    with precision(Precision.HIGH):
      self.assertIn(
          "precision = [HIGH, HIGH]",
          utils.format_model(model, a, b, shlo=True),
      )
      self.assertEqual(model(a, b).shape, (10, 10))

    with precision(Precision.HIGHEST):
      self.assertIn(
          "precision = [HIGHEST, HIGHEST]",
          utils.format_model(model, a, b, shlo=True),
      )
      self.assertEqual(model(a, b).shape, (10, 10))

  def test_nested_precision_in_shlo(self):
    device = torch.device("tpu")
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)

    def model(a, b):
      return torch.matmul(a, b)

    self.assertIn(
        "precision = [DEFAULT, DEFAULT]",
        utils.format_model(model, a, b, shlo=True),
    )

    with precision(Precision.HIGH):
      d = torch.matmul(a, b)
      self.assertIn(
          "precision = [HIGH, HIGH]",
          utils.format_model(model, a, b, shlo=True),
      )
      self.assertEqual(d.shape, (10, 10))
      with precision(Precision.HIGHEST):
        e = torch.matmul(a, b)
        self.assertIn(
            "precision = [HIGHEST, HIGHEST]",
            utils.format_model(model, a, b, shlo=True),
        )
        self.assertEqual(e.shape, (10, 10))
      f = torch.matmul(a, b)
      self.assertIn(
          "precision = [HIGH, HIGH]",
          utils.format_model(model, a, b, shlo=True),
      )
      self.assertEqual(f.shape, (10, 10))

  def test_nested_precision_layer_in_shlo(self):
    device = torch.device("tpu")
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)

    class Model(torch.nn.Module):

      def forward(self, a, b):
        self.matmul_default = torch.matmul(a, b)
        with precision(Precision.HIGH):
          self.matmul_high = torch.matmul(self.matmul_default, b)
        with precision(Precision.HIGHEST):
          self.matmul_highest = torch.matmul(self.matmul_high, b)
        self.matmul_default_2 = torch.matmul(self.matmul_highest, b)
        return [
            self.matmul_default,
            self.matmul_high,
            self.matmul_highest,
            self.matmul_default_2,
        ]

    model = Model()
    model_str = utils.format_model(model, a, b, shlo=True)
    self.assertRegex(
        model_str,
        r"(?s).*precision = \[DEFAULT, DEFAULT\]"
        r".*precision = \[HIGH, HIGH\]"
        r".*precision = \[HIGHEST, HIGHEST\]"
        r".*precision = \[DEFAULT, DEFAULT\]",
    )

  def test_thread_local(self):
    # Set precision in main thread
    self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

    def worker_high():
      with precision(Precision.HIGH):
        self.assertEqual(p_impl._get_precision(), Precision.HIGH)

    def worker_highest():
      # Start without any context, should default to default
      self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)
      with precision(Precision.HIGHEST):
        self.assertEqual(p_impl._get_precision(), Precision.HIGHEST)

    with precision(Precision.DEFAULT):
      t1 = threading.Thread(target=worker_high)
      t2 = threading.Thread(target=worker_highest)

      t1.start()
      t2.start()

      t1.join()
      t2.join()

      # The main thread should still be default
      self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

  def test_thread_local_stress(self):

    def worker(target_precision):
      for _ in range(100):
        with precision(target_precision):
          self.assertEqual(p_impl._get_precision(), target_precision)
        self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

    threads = []
    configs = [
        Precision.HIGH,
        Precision.HIGHEST,
        Precision.DEFAULT,
    ]
    for i in range(10):
      config = configs[i % len(configs)]
      t = threading.Thread(target=worker, args=(config,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

  def test_thread_local_inheritance(self):
    def child_worker():
      # Child thread should NOT inherit the parent's precision context.
      self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

    with precision(Precision.HIGH):
      self.assertEqual(p_impl._get_precision(), Precision.HIGH)
      t = threading.Thread(target=child_worker)
      t.start()
      t.join()
      self.assertEqual(p_impl._get_precision(), Precision.HIGH)

  def test_thread_pool_executor(self):
    def worker(target_precision, target_impl_precision):
      with precision(target_precision):
        self.assertEqual(p_impl._get_precision(), target_impl_precision)
      return p_impl._get_precision()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = []
      futures.append(executor.submit(worker, Precision.HIGH, Precision.HIGH))
      futures.append(
          executor.submit(worker, Precision.HIGHEST, Precision.HIGHEST)
      )

      for future in concurrent.futures.as_completed(futures):
        self.assertEqual(future.result(), Precision.DEFAULT)

  def test_nested_thread_local(self):
    def worker():
      with precision(Precision.HIGH):
        self.assertEqual(p_impl._get_precision(), Precision.HIGH)
        with precision(Precision.HIGHEST):
          self.assertEqual(p_impl._get_precision(), Precision.HIGHEST)
        self.assertEqual(p_impl._get_precision(), Precision.HIGH)
      self.assertEqual(p_impl._get_precision(), Precision.DEFAULT)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

  @unittest.skip("b/514829191")
  def test_torch_compile_precision_caching_bug(self):
    # Arrange
    n = 256
    device = torch.device("tpu")
    eye = torch.eye(n, dtype=torch.float32, device=device)

    one_plus_beyond_bf16_epsilon = 1 + 2**-8
    # Under DEFAULT (bf16 math), one_plus_beyond_bf16_epsilon rounds to 1.0.
    expected_default = 1.0
    # Under HIGHEST (f32 math), one_plus_beyond_bf16_epsilon is preserved.
    expected_highest = 1.0 + 2**-7 + 2**-16

    a = eye.clone() * one_plus_beyond_bf16_epsilon

    def tensor_square(a):
      return a @ a

    # Verify eager semantics.
    with precision(Precision.DEFAULT):
      actual = tensor_square(a)
      self.assertEqual(actual[0, 0].item(), expected_default)
    with precision(Precision.HIGHEST):
      actual = tensor_square(a)
      self.assertEqual(actual[0, 0].item(), expected_highest)

    # Populate torch.compile cache.
    with precision(Precision.HIGHEST):
      actual = torch.compile(tensor_square)(a)
      self.assertEqual(actual[0, 0].item(), expected_highest)

    # Act
    with precision(Precision.DEFAULT):
      actual = torch.compile(tensor_square)(a)

      # Assert
      # This assert fails because tensor_square is not freshly compiled.
      # Instead, torch.compile gives a false positive cache hit.
      self.assertEqual(actual[0, 0].item(), expected_default)


if __name__ == "__main__":
  absltest.main()
