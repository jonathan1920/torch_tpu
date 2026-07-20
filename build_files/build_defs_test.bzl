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

"""Unit tests for build_defs.bzl."""

load("@rules_testing//lib:test_suite.bzl", "test_suite")
load("//build_files:build_defs.bzl", "check_and_adjust_test_tags_for_testing", "is_oss", "tpu_gen")

def _test_nobuild(env):
    """Tests the nobuild parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nobuild"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_notap(env):
    """Tests the notap parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "notap", "notest"])  # NOTAP_OK=Testing notap tagging
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly([])

def _test_nopresubmit(env):
    """Tests the nopresubmit parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "nopresubmit"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_nolocal(env):
    """Tests the nolocal parameter."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nolocal = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_nobuild_oss(env):
    """Tests the nobuild_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild_oss = "reason",
        tags = tags,
    )

    # In an internal build, nobuild_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nobuild_oss(env):
    """Tests the nobuild_oss parameter in OSS."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nobuild_oss = "reason",
        tags = tags,
    )

    # In an OSS build, nobuild_oss should disable generating the build_test.
    env.expect.that_collection(tags).contains_exactly(["nobuild", "notest"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_notest_oss(env):
    """Tests the notest_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notest_oss = "reason",
        tags = tags,
    )

    # In an internal build, notest_oss has no effect and thus does not make
    # it necessary to generate a build_test.
    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_notest_oss(env):
    """Tests the notest_oss parameter in OSS."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason",
        tags = tags,
    )

    # In an OSS build, notest_oss should enable generating the build_test.
    env.expect.that_collection(tags).contains_exactly(["notest"])
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly([])

def _test_cuda_build_test(env):
    """Tests the notap parameter for CUDA tests."""
    tags = ["requires-gpu-a100"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    expected_tags = ["manual", "notap", "notest", "requires-gpu", "requires-gpu-a100"] if is_oss() else ["manual", "notap", "notest", "requires-gpu-a100"]  # NOTAP_OK=Testing notap tagging
    env.expect.that_collection(tags).contains_exactly(expected_tags)
    env.expect.that_bool(result.create_build_test).equals(True)
    env.expect.that_collection(result.build_test_tags).contains_exactly(["requires-gpu-nvidia"])

def _test_internal_notap_nobuild(env):
    """Tests using both notap and nobuild in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        nobuild = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["manual", "nobuild", "notap", "notest"])  # NOTAP_OK=Testing notap tagging
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_notest_oss_nobuild_oss(env):
    """Tests using both notest_oss and nobuild_oss in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        notest_oss = "reason1",
        nobuild_oss = "reason2",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nobuild", "notest"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_internal_nopresubmit_oss(env):
    """Tests the nopresubmit_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nopresubmit_oss(env):
    """Tests the nopresubmit_oss parameter in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_manual_nopresubmit_oss_tag(env):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in internal builds."""
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit_oss"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_manual_nopresubmit_oss_tag(env):
    """Tests that the nopresubmit_oss param appends correctly even if 'nopresubmit_oss' is in tags in OSS builds."""
    tags = ["nopresubmit_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nopresubmit_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nopresubmit", "nopresubmit_oss"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_nonightly_oss(env):
    """Tests the nonightly_oss parameter in internal builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly([])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_nonightly_oss(env):
    """Tests the nonightly_oss parameter in OSS builds."""
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_internal_manual_nonightly_oss_tag(env):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in internal builds."""
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly_oss"])
    env.expect.that_bool(result.create_build_test).equals(False)

def _test_oss_manual_nonightly_oss_tag(env):
    """Tests that the nonightly_oss param appends correctly even if 'nonightly_oss' is in tags in OSS builds."""
    tags = ["nonightly_oss"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = True,
        nonightly_oss = "reason",
        tags = tags,
    )

    env.expect.that_collection(tags).contains_exactly(["nonightly", "nonightly_oss"])
    env.expect.that_bool(result.create_build_test).equals(True)

def _test_oss_presubmit_tpu_generation_explicit(env):
    tags = []
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        oss_presubmit_tpu_generation = tpu_gen("v7", reason = "Requires specific hardware for feature testing"),
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")

def _test_oss_presubmit_tpu_generation_implicit(env):
    tags = ["requires-tpu", "fails-on-tpu-v5"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v7")

def _test_oss_presubmit_tpu_generation_implicit_v6(env):
    tags = ["requires-tpu", "fails-on-tpu-v5", "fails-on-tpu-v7"]
    check_and_adjust_test_tags_for_testing(
        is_oss = True,
        tags = tags,
    )
    env.expect.that_collection(tags).contains("presubmit-v6")

def build_defs_test_suite(name):
    """Creates a test suite for build_defs.bzl, which will run all tests in this file.

    Args:
        name: The name of the test suite. All tests in this suite will be prefixed with this name.
    """
    test_suite(
        name = name,
        basic_tests = [
            # go/keep-sorted start
            _test_cuda_build_test,
            _test_internal_manual_nonightly_oss_tag,
            _test_internal_manual_nopresubmit_oss_tag,
            _test_internal_nobuild_oss,
            _test_internal_nonightly_oss,
            _test_internal_nopresubmit_oss,
            _test_internal_notap_nobuild,
            _test_internal_notest_oss,
            _test_nobuild,
            _test_nolocal,
            _test_nopresubmit,
            _test_notap,
            _test_oss_manual_nonightly_oss_tag,
            _test_oss_manual_nopresubmit_oss_tag,
            _test_oss_nobuild_oss,
            _test_oss_nonightly_oss,
            _test_oss_nopresubmit_oss,
            _test_oss_notest_oss_nobuild_oss,
            _test_oss_notest_oss,
            _test_oss_presubmit_tpu_generation_explicit,
            _test_oss_presubmit_tpu_generation_implicit,
            _test_oss_presubmit_tpu_generation_implicit_v6,
            # go/keep-sorted end
        ],
    )
