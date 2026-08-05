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

"""Unit tests for TorchTPU API lifecycle stage annotations."""

import warnings

from absl.testing import absltest
from torch_tpu._internal.device._device_module import _DeviceModule
from torch_tpu._internal.utils import annotations

experimental = annotations.experimental
stable = annotations.stable
deprecated = annotations.deprecated


class AnnotationsTest(absltest.TestCase):

  def test_annotated_real_api_get_amp_supported_dtype(self):
    """Verifies that real API get_amp_supported_dtype is annotated."""
    fn = getattr(_DeviceModule, "get_amp_supported_dtype", None)
    self.assertIsNotNone(fn)
    self.assertEqual(getattr(fn, "__tt_api_stage__", None), "Experimental")
    self.assertIn(
        "get_amp_supported_dtype", getattr(fn, "__tt_stage_reason__", "")
    )

  def test_experimental_first_call_triggers_warning(self):
    @experimental("Testing experimental feature.")
    def sample_func(a: int, b: int) -> int:
      return a + b

    self.assertEqual(sample_func.__tt_api_stage__, "Experimental")
    self.assertEqual(
        sample_func.__tt_stage_reason__, "Testing experimental feature."
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func(1, 2)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("sample_func is experimental", str(w[0].message))

  def test_experimental_subsequent_calls_suppress_warning(self):
    @experimental("Testing experimental feature.")
    def sample_func(a: int, b: int) -> int:
      return a + b

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      sample_func(1, 2)
      self.assertLen(w, 1)

      # Second and third calls must NOT trigger additional warnings
      sample_func(2, 3)
      sample_func(3, 4)
      self.assertLen(w, 1)

  def test_experimental_multiple_functions_independent_warnings(self):
    @experimental("First experimental feature.")
    def func_one(x: int) -> int:
      return x + 1

    @experimental("Second experimental feature.")
    def func_two(x: int) -> int:
      return x + 2

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      func_one(1)
      self.assertLen(w, 1)
      self.assertIn("func_one is experimental", str(w[0].message))

      # Calling second experimental function must trigger its own warning
      func_two(1)
      self.assertLen(w, 2)
      self.assertIn("func_two is experimental", str(w[1].message))

  def test_experimental_invalid_reason_raises(self):
    with self.assertRaises(ValueError):

      @experimental("")
      def dummy_one():
        pass

    with self.assertRaises(ValueError):

      @experimental("   ")
      def dummy_two():
        pass

  def test_stable_metadata_no_warning(self):
    @stable("Production ready.")
    def stable_func(x: int) -> int:
      return x * 2

    self.assertEqual(stable_func.__tt_api_stage__, "Stable")
    self.assertEqual(stable_func.__tt_stage_reason__, "Production ready.")

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      stable_func(5)
      self.assertEmpty(w)

  def test_deprecated_metadata_and_warning(self):
    @deprecated(version="2.13", reason="Use new_api() instead.")
    def legacy_func(x: int) -> int:
      return x + 10

    self.assertEqual(legacy_func.__tt_api_stage__, "Deprecated")
    self.assertEqual(legacy_func.__tt_deprecated_version__, "2.13")
    self.assertEqual(legacy_func.__tt_stage_reason__, "Use new_api() instead.")

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      legacy_func(5)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "legacy_func is deprecated as of TorchTPU 2.13", str(w[0].message)
      )

      # Second invocation must NOT trigger warning
      legacy_func(6)
      self.assertLen(w, 1)

  def test_deprecated_invalid_args_raises(self):
    with self.assertRaises(ValueError):

      @deprecated(version="", reason="Valid reason")
      def dummy_one():
        pass

    with self.assertRaises(ValueError):

      @deprecated(version="2.13", reason="")
      def dummy_two():
        pass


if __name__ == "__main__":
  absltest.main()
