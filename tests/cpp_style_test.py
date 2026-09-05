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
from tests import cpp_style_lib as style


class CppStyleTest(absltest.TestCase):  # ABSLTEST_OK=avoid heavy dependencies

  def test_find_test_data(self):
    cpp_code = style.CppCode("tests")

    self.assertIsNotNone(cpp_code.find_file("cpp_style_test_data.cc"))

  def test_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (["torch_tpu"], "void Foo() { torch_tpu::TorchTpuClass(); }"), output
    )

  def test_torch_tpu_path(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")
    self.assertIsNotNone(file, "Could not find cpp_style_test_data.cc")
    self.assertEqual(file.torch_tpu_path, "tests/cpp_style_test_data.cc")

  def test_nested_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (
            ["outer", "torch_tpu"],
            "void Foo3() { ::torch_tpu::TorchTpuClass(); }",
        ),
        output,
    )

  def test_unnamed_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(([""], "void Foo4() { torch_tpu::TorchTpuClass(); }"), output)

  def test_multi_segment_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (["foo", "bar"], "void Foo5() { torch_tpu::TorchTpuClass(); }"), output
    )

  def test_unnapproved_nested_namespace_in_test_data(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")
    self.assertIsNotNone(file, "Could not find cpp_style_test_data.cc")

    errors = style.check_unnapproved_namespace(file)

    expected_error = (
        "}  // namespace torch_tpu::bar: Only approved named namespaces allowed"
        " under torch_tpu: ['error', 'internal'], got ['bar']"
    )
    expected_error_2 = (
        "void Baz2() { torch_tpu::TorchTpuClass(); }: Only approved named"
        " namespaces allowed under torch_tpu: ['error', 'internal'], got ['',"
        " 'internal']"
    )
    self.assertIn(expected_error, errors)
    self.assertIn(expected_error_2, errors)
    self.assertFalse(any("Baz3()" in e for e in errors))


if __name__ == "__main__":
  absltest.main()
