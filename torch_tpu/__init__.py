# Copyright 2025 Google LLC
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

"""TorchTPU library."""

try:
  import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
