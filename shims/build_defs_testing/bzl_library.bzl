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

"""Shim for the Google version of bzl_library.

The Google version of bzl_library supports additional functionality not
present in bazel_skylib. These shims allow exported code to support the
same interface and simplify the copybara import/export rewriting logic.
"""

load("@bazel_skylib//:bzl_library.bzl", _bzl_library = "bzl_library")
load("//:visibility.bzl", "BZL_VISIBILITY")

visibility(BZL_VISIBILITY)

def bzl_library(**kwargs):
    internal_args = [
        "parse_tests",
        "test_tags",
        "test_size",
        "test_timeout",
        "selected_build_configurations",
    ]
    for arg_name in internal_args:
        kwargs.pop(arg_name, None)

    _bzl_library(**kwargs)
