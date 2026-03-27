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

"""Shim for build_defs from pynball."""

def ipynb_deps(
        name,
        pytype = False,
        srcs = None,
        deps = None,
        **kwargs):
    """OSS shim for ipynb_deps rule.

    Internally, this rule converts notebooks to Python scripts, injects Colab
    dependencies, and generates internal build_tests. In OSS, we don't execute
    notebooks via Bazel, so this shim simply registers the files in a filegroup
    to satisfy BUILD targets without crashing.

    Args:
        name: The name of the target.
        pytype: Ignored in OSS.
        srcs: The list of .ipynb files.
        deps: Ignored in OSS (prevents missing dependency errors).
        **kwargs: Catches and swallows any other internal args (tags, data, etc.).
    """

    _ = pytype  # @unused
    _ = deps  # @unused
    _ = kwargs  # @unused

    # Use a filegroup to ensure the target name exists and the files are tracked
    # by Bazel, but do not attempt to compile or test them.
    native.filegroup(
        name = name,
        srcs = srcs,
    )
