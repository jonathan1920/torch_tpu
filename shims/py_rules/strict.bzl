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

"""Stubs for py-strict rules."""

load("@rules_python//python:py_test.bzl", "py_test")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "use_pywrap_rules")
load("//:visibility.bzl", "BZL_VISIBILITY")

visibility(BZL_VISIBILITY)

def py_strict_test(
        extra_pywrap_deps = ["//torch_tpu/common:pywrap_torch_tpu"],
        **kwargs):
    """
    wrapper around py_test that adds pywrap_torch_tpu to the deps.

    Args:
        extra_pywrap_deps: Additional pywrap dependencies to add to the test.
        **kwargs: Any additional arguments.
    """
    deps_to_add = []
    if use_pywrap_rules():
        deps_to_add = extra_pywrap_deps

    current_deps = kwargs.pop("deps", [])
    all_deps = current_deps + deps_to_add

    py_test(
        # PY_TEST_OK=no changes in tests except for the wrapper and add pywrap_torch_tpu
        deps = all_deps,
        **kwargs
    )
