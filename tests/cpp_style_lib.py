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

"""Helper library for C++ style linter."""

import os
import pathlib
import re
from typing import Generator


def _find_repo_root() -> pathlib.Path:
  """Returns the root directory of the torch_tpu repository."""
  rf_dir = os.environ.get("TEST_SRCDIR") or os.environ.get("RUNFILES_DIR")
  if rf_dir:
    for candidate in [
        # Blaze (internal Google3).
        pathlib.Path(rf_dir) / "google3" / "third_party" / "py" / "torch_tpu",
        # Bazel (OSS with Bzlmod names the root workspace '_main').
        pathlib.Path(rf_dir) / "_main",
        # Bazel (classic WORKSPACE or when torch_tpu is used as an external dependency).
        pathlib.Path(rf_dir) / "torch_tpu",
    ]:
      if (candidate / "tests" / "cpp_style_test_data.cc").exists():
        return candidate

  raise FileNotFoundError(
      "Could not find torch_tpu repository root in runfiles."
  )


_RUNFILES_DIR = _find_repo_root()

_NAMESPACE_START_RE = re.compile(r"^namespace\s*([\w_:]*)\s*{")
_NAMESPACE_END_RE = re.compile(r"^}  // namespace\s*([\w_:]*)\s*")


class CppFile:
  """Represents a C++ file with its path."""

  def __init__(self, path: pathlib.Path):
    self._path = path

  @property
  def path(self):
    return self._path

  @property
  def torch_tpu_path(self) -> str:
    """Returns the torch_tpu relative path as a string."""
    return str(self._path.relative_to(pathlib.Path(_RUNFILES_DIR)))

  def __repr__(self):
    return self.torch_tpu_path

  def get_namespace(self) -> Generator[tuple[list[str], str], None, None]:
    """Yields ((active_namespace...), line) for each line in the file.

    Example:
    c++```
    namespace foo {
    namespace bar{
    namespace {
    void Baz() {}
    }  // namespace
    }  // namespace bar
    }  // namespace foo
    ```

    Will generate result containing (["foo", "bar", ""], "void Baz() {}").
    """

    namespace_stack = []

    for line in self.path.read_text().splitlines():
      if not line:
        continue

      match = _NAMESPACE_START_RE.match(line)
      if match:
        name = match.group(1)
        segments = name.split("::")
        yield (list(namespace_stack), line)
        namespace_stack.extend(segments)
        continue

      match = _NAMESPACE_END_RE.match(line)
      if match:
        name = match.group(1)
        segments = name.split("::")
        yield (list(namespace_stack), line)
        for _ in reversed(segments):
          namespace_stack.pop()
        continue

      yield (list(namespace_stack), line)


class CppCode:
  """Provides access to C++ files in torch_tpu."""

  def __init__(self, torch_tpu_path: str = ""):
    self._torch_tpu_path = torch_tpu_path
    assert self.path.exists()

  @property
  def path(self):
    return pathlib.Path(_RUNFILES_DIR) / self._torch_tpu_path

  def files(self) -> Generator[CppFile, None, None]:
    """Yields CppFile objects for the selected files."""
    for path in self.path.rglob("*"):
      if path.is_file() and path.suffix in [".cc", ".h"]:
        yield CppFile(path)

  def find_file(self, filename: str) -> CppFile | None:
    """Finds a file by its name in the selected files."""
    for f in self.files():
      if f.path.name == filename:
        return f
    return None


# TODO(lukeboyer): Consider more general checks that monitor first-level
# namespaces other than torch_tpu.
def check_unnapproved_namespace(file: CppFile) -> list[str]:
  """Checks if a CppFile uses any unapproved nested namespaces under torch_tpu.

  Args:
    file: The CppFile to check.

  Returns:
    A list of strings, where each string is an error message. Empty results mean
    no improper usage.
  """
  approved = {"error", "internal"}
  approved_msg = str(sorted(approved))
  errors = []
  for active_namespaces, line in file.get_namespace():
    if len(active_namespaces) <= 1 or active_namespaces[0] != "torch_tpu":
      continue
    # We only permit the inner-most namespace to be anonymous.
    if not active_namespaces[-1]:
      active_namespaces.pop(-1)
    namespaces_to_check = active_namespaces[1:]
    if not all(n in approved for n in namespaces_to_check):
      errors.append(
          f"{line}: Only approved named namespaces allowed under torch_tpu:"
          f" {approved_msg}, got {namespaces_to_check}"
      )

  return errors
