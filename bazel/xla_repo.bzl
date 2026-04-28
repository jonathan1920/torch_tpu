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

"""Defines the repo that provides the xla codebase."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(":xla_revision.bzl", "XLA_COMMIT", "XLA_SHA256")

def xla_repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = [
            "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT),
        ],
        patches = ["//bazel/patches/xla:xla_bazel8.patch"],
        patch_args = ["-p1"],
    )
