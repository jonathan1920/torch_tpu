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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "hello_world_distributed_test",
    srcs = ["hello_world_distributed_test.py"],
    tags = [
        "notap",  # NOTAP_OK=this is a manual test
    ],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

build_test(
    name = "build_test",
    targets = [
        ":hello_world_distributed_test",
        ":run_script",
    ],
)

torch_tpu_py_test(
    name = "run_script",
    srcs = ["run_script.py"],
    env = {
        "WORLD_SIZE": "8",
    },
    tags = [
        "notap",  # NOTAP_OK=this is a manual test
        "requires-net:external",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":hello_world_distributed_test",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/shims/g3_multiprocessing",
    ],
)

torch_tpu_package_end()
