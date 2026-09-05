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

"""Tests for unnapproved nested namespace violations."""

from typing import Generator

from absl.testing import absltest
from absl.testing import parameterized
from tests import cpp_style_lib as style


def _get_files_to_lint() -> Generator[style.CppCode, None, None]:
  """Yields all files subject to linting."""
  cpp_code = style.CppCode()
  for file in cpp_code.files():
    if file.torch_tpu_path != 'tests/cpp_style_test_data.cc':
      yield file


class UnnapprovedNestedNamespaceTest(
    parameterized.TestCase,  # ABSLTEST_OK=linter tool test
):

  @parameterized.parameters(_get_files_to_lint())
  def test_unnapproved_nested_namespace(self, file: style.CppFile):
    errors = style.check_unnapproved_namespace(file)
    self.assertEmpty(errors, '\n'.join(errors))


if __name__ == '__main__':
  absltest.main()
