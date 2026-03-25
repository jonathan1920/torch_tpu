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

"""Module for single-sourcing wheel dependencies with version pins."""

def _normalize_pkg_name(req_str):
    """Extracts the base package name (e.g. 'absl-py' from 'absl-py==2.3.1')"""
    name = req_str

    # Split by common version/extra specifiers to isolate the base name
    for char in ["=", ">", "<", "~", "[", ";", " "]:
        if char in name:
            name = name.split(char)[0]
    return name.strip().lower().replace("_", "-")

def _torch_tpu_deps_repo_impl(ctx):
    # 1. Build a dictionary of pinned versions from requirements.txt
    reqs_path = ctx.path(ctx.attr.requirements_txt)
    reqs_content = ctx.read(reqs_path)

    pinned_deps = {}
    for line in reqs_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Remove inline pip-compile comments (e.g. "numpy==1.0  # via...")
        dep_full = line.split(" #")[0].split(";")[0].strip()
        dep_full = dep_full.split("\\")[0].strip()

        if dep_full:
            pkg_name = _normalize_pkg_name(dep_full)
            pinned_deps[pkg_name] = dep_full

    # 2. Parse pyproject.toml groupings and apply the pins
    pyproject_path = ctx.path(ctx.attr.pyproject_toml)
    content = ctx.read(pyproject_path)

    dependencies = []
    optional_deps = {}
    current_section = None
    current_list_name = None

    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Keep track of the current TOML table/section
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            current_list_name = None
            continue

        # Parse [project] -> dependencies
        if current_section == "project":
            if line.startswith("dependencies = ["):
                current_list_name = "dependencies"
                continue
            elif current_list_name == "dependencies":
                if line == "]":
                    current_list_name = None
                else:
                    raw_dep = line.strip(",").strip('"').strip("'").strip()
                    if raw_dep:
                        pkg_name = _normalize_pkg_name(raw_dep)

                        # Swap in the pinned version if we found it in requirements.txt!
                        final_dep = pinned_deps.get(pkg_name, raw_dep)
                        dependencies.append(final_dep)

            # Parse [project.optional-dependencies] -> dev, benchmark, etc.
        elif current_section == "project.optional-dependencies":
            if "=" in line and line.endswith("["):
                current_list_name = line.split("=")[0].strip()
                optional_deps[current_list_name] = []
                continue
            elif current_list_name:
                if line == "]":
                    current_list_name = None
                else:
                    raw_dep = line.strip(",").strip('"').strip("'").strip()
                    if raw_dep:
                        pkg_name = _normalize_pkg_name(raw_dep)
                        final_dep = pinned_deps.get(pkg_name, raw_dep)
                        optional_deps[current_list_name].append(final_dep)

    if not dependencies:
        fail("Dependencies not found in pyproject.toml [project] section")

    # 3. Export the variables
    ctx.file("BUILD.bazel", "")

    bzl_content = "WHEEL_REQUIRES = {}\n".format(dependencies)
    for opt_name, opt_deps in optional_deps.items():
        bzl_content += "WHEEL_{}_REQUIRES = {}\n".format(opt_name.upper().replace("-", "_"), opt_deps)

    ctx.file("dependencies.bzl", bzl_content)

torch_tpu_deps_repo = repository_rule(
    implementation = _torch_tpu_deps_repo_impl,
    attrs = {
        "pyproject_toml": attr.label(mandatory = True, allow_single_file = True),
        "requirements_txt": attr.label(mandatory = True, allow_single_file = True),
    },
)
