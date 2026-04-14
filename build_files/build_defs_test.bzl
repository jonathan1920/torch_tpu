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

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//build_files:build_defs.bzl", "check_and_adjust_test_tags_for_testing")

def _test_nobuild(ctx):
    """Tests the nobuild parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nobuild"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_notap(ctx):
    """Tests the notap parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["manual", "notap"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == [],
        "build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_nopresubmit(ctx):
    """Tests the nopresubmit parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nopresubmit = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["manual", "nofastbuild"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_nolocal(ctx):
    """Tests the nolocal parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nolocal = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["manual"], "tags: %s" % tags)
    asserts.false(env, result.create_build_test)
    return unittest.end(env)

def _test_nobuild_oss(ctx):
    """Tests the nobuild_oss parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        nobuild_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["nobuild_oss", "notest_oss"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == ["nobuild_oss"],
        "result.build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_notest_oss(ctx):
    """Tests the notest_oss parameter."""

    env = unittest.begin(ctx)
    tags = []
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notest_oss = "reason",
        tags = tags,
    )

    asserts.true(env, tags == ["notest_oss"], "tags: %s" % tags)
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == [],
        "result.build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

def _test_cuda_build_test(ctx):
    """Tests the notap parameter for CUDA tests."""

    env = unittest.begin(ctx)
    tags = ["requires-gpu-a100"]
    result = check_and_adjust_test_tags_for_testing(
        is_oss = False,
        notap = "reason",
        tags = tags,
    )

    asserts.true(
        env,
        tags == ["manual", "notap", "requires-gpu-a100"],  # NOTAP_OK=tests
        "tags: %s" % tags,
    )
    asserts.true(env, result.create_build_test)
    asserts.true(
        env,
        result.build_test_tags == ["requires-gpu-nvidia"],
        "result.build_test_tags: %s" % result.build_test_tags,
    )
    return unittest.end(env)

nobuild_test = unittest.make(_test_nobuild)
notap_test = unittest.make(_test_notap)
nopresubmit_test = unittest.make(_test_nopresubmit)
nolocal_test = unittest.make(_test_nolocal)
nobuild_oss_test = unittest.make(_test_nobuild_oss)
notest_oss_test = unittest.make(_test_notest_oss)
cuda_build_test = unittest.make(_test_cuda_build_test)

def build_defs_test_suite(name):
    """Creates a test suite for build_defs.bzl, which will run all tests in this file.

    Args:
        name: The name of the test suite. All tests in this suite will be prefixed with this name.
    """

    def add_test(tests, rule_func, name):
        rule_func(name = name)
        tests.append(":" + name)

    tests = []
    add_test(tests, nobuild_test, name + "_nobuild")
    add_test(tests, notap_test, name + "_notap")
    add_test(tests, nopresubmit_test, name + "_nopresubmit")
    add_test(tests, nolocal_test, name + "_nolocal")
    add_test(tests, nobuild_oss_test, name + "_nobuild_oss")
    add_test(tests, notest_oss_test, name + "_notest_oss")
    add_test(tests, cuda_build_test, name + "_cuda_build_test")

    native.test_suite(
        name = name,
        tests = tests,
    )
