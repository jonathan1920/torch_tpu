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

"""Common build definitions for torch_tpu."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_test.bzl", "py_test")
load(
    "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    "use_pywrap_rules",
)
load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")
load("//shims/py_rules:pytype.bzl", "pytype_strict_contrib_test")

# This file is torch_tpu implementation details and should not be imported by
# other projects.
visibility(
    [
        "//...",
        # Visibility is not ignored for .bzl loads in experimental
        # unlike regular targets.
        # copybara:uncomment "//experimental/users/...",
    ],
)

# Default C/C++ build flags for torch_tpu.
_TORCH_TPU_COPTS = [
    "-std=c++17",  # Align with pytorch and openxla.
    # Enable exceptions for reporting errors to pytorch. Without this, any
    # C++ exception thrown will immediately crash the process instead of being
    # converted to a Python exception.
    "-fexceptions",
]

def _get_relative_torch_tpu_root(package_name):
    """Gets the location of "torch_tpu"""
    components = package_name.split("/")
    for i, component in enumerate(reversed(components)):
        if component == "torch_tpu":
            # It's an error for the includes attribute to refer to the workspace
            # root. See http://github.com/bazelbuild/bazel/issues/27390
            # Luckily, the workspace root is already on the include search
            # path, so we can omit it entirely in such a case.
            if i + 1 == len(components):
                return None
            return "/".join([".."] * (i + 1))
    fail("Package is not relative to torch_tpu")

def adjust_cc_options(copts, features):
    """Adjusts the C/C++ build options for torch_tpu.

    Args:
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.

    Returns:
        A tuple of the adjusted C/C++ compiler options and blaze features.
    """

    copts = copts or []
    copts += _TORCH_TPU_COPTS
    features = features or []

    # use_header_modules is enabled by default but incompatible with -fexceptions.
    # Therefore we must explicitly disable it here.
    features.append("-use_header_modules")
    return copts, features

def torch_tpu_cc_library(name, copts = None, features = None, **kwargs):
    """Creates a C++ library for torch_tpu.

    Args:
        name: The name of the library.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        **kwargs: Any additional arguments.
    """

    copts, features = adjust_cc_options(copts, features)

    if "includes" not in kwargs:
        include_dir = _get_relative_torch_tpu_root(native.package_name())
        if include_dir:
            kwargs["includes"] = [include_dir]

    cc_library(
        name = name,
        copts = copts,
        features = features,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_cc_library.
register_extension_info(
    extension = torch_tpu_cc_library,
    label_regex_for_dep = "{extension_name}",
)

def _validate_test_tags(tags):
    """Validates the test tags."""

    pass

def torch_tpu_cc_test(
        name,
        copts = None,
        features = None,
        args = None,
        linkstatic = True,
        shuffle_tests = True,
        fail_if_no_test_linked = True,
        tags = None,
        **kwargs):
    """Creates a cc_test for torch_tpu.

    Compared to cc_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        args: The arguments to pass to the test.
        linkstatic: Whether to link statically. We link statically by default to catch duplicate
            definitions and increase accelerator utilization by reducing test run time.
        shuffle_tests: Whether to shuffle the test cases.
        fail_if_no_test_linked: Whether to fail if no tests are linked.
        tags: The tags to add to the test.
        **kwargs: Any additional arguments.
    """

    copts, features = adjust_cc_options(copts, features)

    args = args or []
    if shuffle_tests:
        # Shuffle tests to avoid test ordering dependencies.
        args = args + ["--gunit_shuffle"]
    if fail_if_no_test_linked:
        # Fail if no tests are linked. This is to avoid having a test target that does not run any
        # tests. This can happen if the test's link options are not set correctly.
        args = args + ["--gunit_fail_if_no_test_linked"]
    tags = tags or []
    _validate_test_tags(tags)
    cc_test(
        name = name,
        copts = copts,
        args = args,
        features = features,
        linkstatic = linkstatic,
        tags = tags,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_cc_test.
register_extension_info(
    extension = torch_tpu_cc_test,
    label_regex_for_dep = "{extension_name}",
)

def torch_tpu_py_test(
        name,
        args = None,
        shuffle_tests = True,
        extra_pywrap_deps = ["//torch_tpu/common:pywrap_torch_tpu"],
        strict = False,
        tags = None,
        **kwargs):
    """Creates a py_test for torch_tpu.

    Compared to py_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        args: The arguments to pass to the test.
        shuffle_tests: Whether to shuffle the test cases.
        extra_pywrap_deps: Additional pywrap dependencies to add to the test.
        strict: Whether to use pytype.
        tags: The tags to add to the test.
        **kwargs: Any additional arguments.
    """

    args = args or []
    if shuffle_tests:
        # Shuffle test cases to avoid test ordering dependencies.
        args = args + ["--test_randomize_ordering_seed=random"]

    tags = tags or []
    _validate_test_tags(tags)

    # Remove internal-only attributes
    kwargs.pop("linking_mode", None)

    # Add env and LD_LIBRARY_PATH for wheel based testing.
    # Define the necessary library paths

    # 1. Define all paths needed for the dynamic linker
    torch_lib = "../pypi_torch/site-packages/torch/lib"
    libtpu_lib = "../pypi_libtpu/site-packages/libtpu"
    wheel_lib = "../torch_tpu_py_import_unpacked_wheel/torch_tpu/_internal"
    solib_path = "../_solib_x86_64"

    # 2. Join them into a single string
    all_paths = [torch_lib, libtpu_lib, wheel_lib, solib_path]
    new_paths_str = ":".join(all_paths)

    # 3. Create the specialized environment for wheel testing
    existing_env = kwargs.pop("env", {})
    env_with_wheel = dict(existing_env)
    if "LD_LIBRARY_PATH" in existing_env:
        env_with_wheel["LD_LIBRARY_PATH"] = new_paths_str + ":" + existing_env["LD_LIBRARY_PATH"]
    else:
        env_with_wheel["LD_LIBRARY_PATH"] = new_paths_str

    # 4. Use select to swap between the wheel and non-wheel envs
    test_env = select({
        "//:wheel_test_enabled": env_with_wheel,
        "//conditions:default": existing_env,
    })

    deps_to_add = []
    if use_pywrap_rules():
        deps_to_add = extra_pywrap_deps

    current_deps = kwargs.pop("deps", [])

    all_deps = select({
        "//:wheel_test_enabled": ["//:torch_tpu_py_import"],
        "//conditions:default": current_deps + deps_to_add,
    })

    if strict:
        rule = pytype_strict_contrib_test
    else:
        rule = py_test
    rule(
        name = name,
        args = args,
        deps = all_deps,
        env = test_env,
        tags = tags,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_py_test.
register_extension_info(
    extension = torch_tpu_py_test,
    label_regex_for_dep = "{extension_name}",
)

def get_subpackage_targets_named(name):
    """Gets targets with the given name from all direct subpackages.

    Example:
        If the current package is `foo` and it has subpackages `foo/bar`,
        `foo/baz`, and `foo/bar/qux`, calling
        `get_subpackage_targets_named("my_target")` from the BUILD file in `foo`
        will return:
        `["//foo/bar:my_target", "//foo/baz:my_target"]`, but not
        `//foo/bar/qux:my_target` since it is not a direct subpackage.

    Args:
        name: The name of the target to collect from each subpackage.

    Returns:
        A list of full target labels.
    """
    current_package = native.package_name()
    targets = []
    for subpackage in native.subpackages(
        # Include all direct subpackages.
        include = ["**"],
        # Allow the include matching results to be empty.
        allow_empty = True,
    ):
        target_path = "//{}/{}:{}".format(current_package, subpackage, name)
        targets.append(target_path)
    return targets

def define_cpp_filegroup(name):
    """Defines filegroups for all C++ files in the current package and subpackages.

    The filegroups created are:
    - "all_cpp_files": All C++ files in the current package.
    - "all_cpp_files_recursive": All C++ files in the current package and
    subpackages recursively.

    Args:
        name: The name of the filegroup. It has to be "all_cpp_files_recursive"
            to collect files from subpackages recursively.
    """
    if name != "all_cpp_files_recursive":
        fail("The name must be 'all_cpp_files_recursive' to collect files recursively.")
    native.filegroup(
        name = "all_cpp_files",
        srcs = native.glob([
            "**/*.cc",
            "**/*.h",
        ]),
        visibility = ["//:__subpackages__"],
    )
    native.filegroup(
        name = name,
        srcs = [":all_cpp_files"] + get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//:__subpackages__"],
    )
