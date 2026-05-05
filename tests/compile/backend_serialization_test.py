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

import pickle
import random

from absl.testing import absltest
import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch._aot_autograd.aot_autograd_result import (
    deserialize_bundled_cache_entry,
)
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


class BackendSerializationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Set a fixed random seed to avoid flakes.
    random.seed(42)
    torch.manual_seed(42)

  def check_serialization(self, f, inputs):
    inputs_tpu = _backend.to_device(inputs, torch.device("tpu"))
    gm = dynamo_graph_capture_for_export(f)(*inputs_tpu)

    backend = _backend.TpuBackend(
        debug=True,
        enable_serialization=True,
    )
    compiled_fn = backend(gm, inputs_tpu)

    result = compiled_fn(*inputs_tpu)
    result = _backend.to_device(result, "cpu")
    utils.assert_close(result[0], inputs[0] + inputs[1])
    utils.assert_close(result[1], inputs[0] * inputs[1])

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)
    pickled = pickle.dumps(entry)
    restored_entry = pickle.loads(pickled)

    restored_fn = deserialize_bundled_cache_entry(restored_entry)
    self.assertTrue(callable(restored_fn))

    restored_result = restored_fn(*inputs_tpu)
    restored_result = _backend.to_device(restored_result, "cpu")
    utils.assert_close(restored_result[0], inputs[0] + inputs[1])
    utils.assert_close(restored_result[1], inputs[0] * inputs[1])

  def test_simple(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), torch.randn(4, 4)]
    self.check_serialization(simple, inputs)

  def test_int_scalar_input(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), 4]
    self.check_serialization(simple, inputs)

  def test_input_mutation(self):
    def inplace_update(x, cache):
      cache.add_(x.sum())
      return (cache + 0,)

    inputs_tpu = _backend.to_device(
        [torch.randn(4, 4), torch.zeros(8, 8)], torch.device("tpu")
    )
    gm = dynamo_graph_capture_for_export(inplace_update)(*inputs_tpu)

    backend = _backend.TpuBackend(
        debug=True,
        enable_serialization=True,
    )
    compiled_fn = backend(gm, inputs_tpu)

    x = torch.randn(4, 4).to(torch.device("tpu"))
    cache = torch.zeros(8, 8).to(torch.device("tpu"))
    result = compiled_fn(x, cache)
    result_cpu = _backend.to_device(result, "cpu")
    expected = x.cpu().sum().item()
    utils.assert_close(result_cpu[0], torch.full((8, 8), expected))

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)
    restored_fn = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry))
    )

    cache2 = torch.zeros(8, 8).to(torch.device("tpu"))
    restored_result = restored_fn(x, cache2)
    restored_result_cpu = _backend.to_device(restored_result, "cpu")
    utils.assert_close(restored_result_cpu[0], torch.full((8, 8), expected))

  def test_different_shapes_produce_different_cache_keys(self):
    def simple(x):
      return (x * 2,)

    backend = _backend.TpuBackend(
        debug=True,
        enable_serialization=True,
    )

    x_small = torch.randn(4, 4).to(torch.device("tpu"))
    gm_small = dynamo_graph_capture_for_export(simple)(x_small)
    compiled_small = backend(gm_small, [x_small])
    result_small = compiled_small(x_small)
    result_small_cpu = _backend.to_device(result_small, "cpu")
    utils.assert_close(result_small_cpu[0], x_small.cpu() * 2)

    x_large = torch.randn(8, 8).to(torch.device("tpu"))
    gm_large = dynamo_graph_capture_for_export(simple)(x_large)
    compiled_large = backend(gm_large, [x_large])
    result_large = compiled_large(x_large)
    result_large_cpu = _backend.to_device(result_large, "cpu")
    utils.assert_close(result_large_cpu[0], x_large.cpu() * 2)

    entry_small = compiled_small.serialize()
    entry_large = compiled_large.serialize()
    self.assertIsNotNone(entry_small)
    self.assertIsNotNone(entry_large)

    restored_small = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_small))
    )
    restored_large = deserialize_bundled_cache_entry(
        pickle.loads(pickle.dumps(entry_large))
    )

    r_small = _backend.to_device(restored_small(x_small), "cpu")
    r_large = _backend.to_device(restored_large(x_large), "cpu")
    utils.assert_close(r_small[0], x_small.cpu() * 2)
    utils.assert_close(r_large[0], x_large.cpu() * 2)

  def test_multiple_serialization_roundtrips(self):
    def simple(x):
      return (x + 1,)

    x = torch.randn(4, 4).to(torch.device("tpu"))
    gm = dynamo_graph_capture_for_export(simple)(x)
    backend = _backend.TpuBackend(
        debug=True,
        enable_serialization=True,
    )
    compiled_fn = backend(gm, [x])

    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)

    for _ in range(3):
      pickled = pickle.dumps(entry)
      entry = pickle.loads(pickled)
      restored_fn = deserialize_bundled_cache_entry(entry)
      result = _backend.to_device(restored_fn(x), "cpu")
      utils.assert_close(result[0], x.cpu() + 1)


if __name__ == "__main__":
  absltest.main()
