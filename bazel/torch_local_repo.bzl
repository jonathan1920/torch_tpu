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

"""Repository rule for local PyTorch development."""

def _torch_local_repo_impl(ctx):
    # We use TORCH_SOURCE to specify the path (e.g., /home/user/pytorch)
    torch_path = ctx.os.environ.get("TORCH_SOURCE", "")

    if not torch_path:
        # Fallback for when the env var isn't set (e.g. non-local analysis)
        ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_library(name = "torch_headers")
cc_library(name = "libc10")
cc_library(name = "libtorch_cpu")
cc_library(name = "libtorch_python")
cc_library(name = "torch_libs")
""")
        return

    # 1. Create a "Shadow Tree" using symbolic links
    # OLD WAY: ctx.symlink("lib", "lib") -> Only saw lib/include
    # NEW WAY: cp -sr -> Symlinks EVERYTHING (c10, aten, torch/csrc, third_party...)
    # This allows Bazel to see the files needed for Source Layout builds.
    ctx.execute(["bash", "-c", "cp -sr {}/* .".format(torch_path)])

    # 2. Surgically remove the conflicting BUILD files
    # Since we are using our own root BUILD file, we must hide PyTorch's internal
    # BUILD files so Bazel doesn't treat subdirectories as separate packages.
    # Since these are symlinks, this deletes the link in the sandbox, not the source file.
    ctx.execute(["bash", "-c", """
        find . -name "BUILD" -delete
        find . -name "BUILD.bazel" -delete
    """])
    ctx.execute(["bash", "-c", """
        if [ -d torch/include ]; then
            rm -rf c10 aten torch/headeronly torch/csrc
            find torch -maxdepth 1 -name '*.h' -delete
        fi
        if [ -d include ] && [ -d torch/include ]; then
            rm -rf include
        fi
    """])

    # 3. Link our custom BUILD file to the root
    ctx.symlink(ctx.attr.build_file_content, "BUILD.bazel")

torch_local_repo = repository_rule(
    implementation = _torch_local_repo_impl,
    environ = ["TORCH_SOURCE"],  # Triggers rebuild if this var changes
    attrs = {
        "build_file_content": attr.label(
            default = "//shims/torch:torch_local.BUILD",
            doc = "The BUILD file to use for the local torch repo",
            allow_single_file = True,
        ),
    },
    local = True,  # Critical: Re-run this rule if local files change
)
