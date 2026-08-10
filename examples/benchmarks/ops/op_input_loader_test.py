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

from absl.testing import absltest
import torch
from examples.benchmarks.ops.op_input_loader import deserialize_args
from examples.benchmarks.ops.op_input_loader import format_shape_signature


class OpInputLoaderTest(absltest.TestCase):

  def test_format_shape_signature(self):
    # Matrix multiplication with bfloat16
    inputs_mm = (
        "([T([4096, 2048], torch.bfloat16), T([2048, 2048], torch.bfloat16)],"
        " {})"
    )
    self.assertEqual(
        format_shape_signature(inputs_mm),
        "<4096x2048xbf16>, <2048x2048xbf16>",
    )

    # Pointwise with float32 and scalar
    inputs_add = "([T([1, 4096, 1], torch.float32), 1e-05], {})"
    self.assertEqual(
        format_shape_signature(inputs_add),
        "<1x4096x1xf32>, 1e-05",
    )

    # Reduction with list dimension and keepdim kwargs
    inputs_mean = "([T([1, 4096, 2048], torch.float32), [-1], True], {})"
    self.assertEqual(
        format_shape_signature(inputs_mean),
        "<1x4096x2048xf32>, [-1], True",
    )

    # Fused Attention / SDPA with kwargs
    inputs_sdpa = (
        "([T([1, 32, 4096, 64], torch.bfloat16), T([1, 8, 4096, 64],"
        " torch.bfloat16), T([1, 8, 4096, 64], torch.bfloat16), None, 0.0,"
        " True], {'scale': 0.125})"
    )
    self.assertEqual(
        format_shape_signature(inputs_sdpa),
        "<1x32x4096x64xbf16>, <1x8x4096x64xbf16>, <1x8x4096x64xbf16>, None,"
        " 0.0, True, scale=0.125",
    )

  def test_deserialize_args(self):
    inputs_str = (
        "([T([2, 3], torch.float32), 5], {'alpha': T([4], torch.int64)})"
    )

    # Use 'cpu' for unit testing to avoid requiring TPU hardware
    args, kwargs = deserialize_args(inputs_str, device="cpu")

    self.assertEqual(len(args), 2)
    self.assertTrue(isinstance(args[0], torch.Tensor))
    self.assertEqual(args[0].shape, torch.Size([2, 3]))
    self.assertEqual(args[0].dtype, torch.float32)
    self.assertEqual(args[0].device.type, "cpu")
    self.assertEqual(args[1], 5)

    self.assertEqual(len(kwargs), 1)
    self.assertTrue(isinstance(kwargs["alpha"], torch.Tensor))
    self.assertEqual(kwargs["alpha"].shape, torch.Size([4]))
    self.assertEqual(kwargs["alpha"].dtype, torch.int64)
    self.assertEqual(kwargs["alpha"].device.type, "cpu")

  def test_random_data(self):
    inputs_str = "([T([100], torch.float32)], {})"
    args, _ = deserialize_args(inputs_str, device="cpu")
    t = args[0]
    # Verify it's not all zeros or ones (implying random data was generated)
    self.assertFalse(torch.all(t == 0))
    self.assertFalse(torch.all(t == 1))

  def test_different_dtypes(self):
    candidate_dtypes = [
        # Floating point
        "float32",
        "float64",
        "float16",
        "bfloat16",
        # Signed integers
        "int64",
        "int32",
        "int16",
        "int8",
        # Unsigned integers
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        # Boolean
        "bool",
        # Complex types
        "complex64",
        "complex128",
        "complex32",
        # Float8 types
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        # Quantized types
        "qint8",
        "quint8",
        "qint32",
        "quint4x2",
        "quint2x4",
    ]
    tested_count = 0
    for dtype_name in candidate_dtypes:
      dtype = getattr(torch, dtype_name, None)
      if dtype is None or not isinstance(dtype, torch.dtype):
        continue
      inputs_str = f"([T([2, 3], torch.{dtype_name})], {{}})"
      args, _ = deserialize_args(inputs_str, device="cpu")
      self.assertEqual(args[0].dtype, dtype)
      self.assertEqual(args[0].shape, torch.Size([2, 3]))
      tested_count += 1

    # Verify at least all 12 baseline standard dtypes were tested
    self.assertGreaterEqual(tested_count, 12)

  def test_format_shape_signature_extended_dtypes(self):
    test_cases = [
        ("([T([2, 4], torch.complex64)], {})", "<2x4xcomplex<f32>>"),
        ("([T([8, 16], torch.complex128)], {})", "<8x16xcomplex<f64>>"),
        ("([T([4], torch.complex32)], {})", "<4xcomplex<f16>>"),
        ("([T([32], torch.uint16)], {})", "<32xui16>"),
        ("([T([32], torch.uint32)], {})", "<32xui32>"),
        ("([T([32], torch.uint64)], {})", "<32xui64>"),
        ("([T([16], torch.float8_e4m3fn)], {})", "<16xf8E4M3FN>"),
        ("([T([16], torch.float8_e5m2)], {})", "<16xf8E5M2>"),
        ("([T([64], torch.qint8)], {})", "<64xqint8>"),
        ("([T([64], torch.quint8)], {})", "<64xquint8>"),
        ("([T([64], torch.qint32)], {})", "<64xqint32>"),
    ]
    for inputs_str, expected in test_cases:
      self.assertEqual(format_shape_signature(inputs_str), expected)

  def test_invalid_input_not_expression(self):
    with self.assertRaises(ValueError):
      deserialize_args("import os", device="cpu")

  def test_invalid_input_not_tuple(self):
    with self.assertRaises(ValueError):
      deserialize_args("[1, 2]", device="cpu")

  def test_invalid_input_wrong_tuple_len(self):
    with self.assertRaises(ValueError):
      deserialize_args("([1], {}, 3)", device="cpu")

  def test_invalid_t_args_count(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2])], {})", device="cpu")

  def test_invalid_t_shape_not_list(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T(2, torch.float32)], {})", device="cpu")

  def test_invalid_t_shape_elements_not_int(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T(['2'], torch.float32)], {})", device="cpu")

  def test_invalid_t_dtype_not_attribute(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], 'float32')], {})", device="cpu")

  def test_invalid_t_dtype_not_torch(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], np.float32)], {})", device="cpu")

  def test_invalid_t_dtype_unknown(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], torch.invalid_dtype)], {})", device="cpu")

  def test_unsupported_function(self):
    with self.assertRaises(ValueError):
      deserialize_args("([some_func()], {})", device="cpu")

  def test_unsupported_ast_node(self):
    with self.assertRaises(ValueError):
      deserialize_args("({1, 2}, {})", device="cpu")


if __name__ == "__main__":
  absltest.main()
