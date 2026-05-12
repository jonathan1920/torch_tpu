#!/bin/bash
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

# Increase the stack size to 64 MiB. TSan uses significantly more stack memory.
ulimit -s 65536

# Find the TSan runtime library file mounted into the runfiles by Bazel.
TSAN_LIB=$(find "${TEST_SRCDIR}" -name "libclang_rt.tsan.so" -print -quit 2>/dev/null)

if [ -n "$TSAN_LIB" ]; then
    # Preload the TSan runtime library before any other shared objects.
    # This forces the OS to load the TSan symbols first and ensures that the
    # TSan runtime is initialized before the Python interpreter is imported.
    export LD_PRELOAD="${TSAN_LIB}:${LD_PRELOAD:-}"
else
    echo "FATAL: Bazel failed to mount libclang_rt.tsan.so into the runfiles!"
    exit 1
fi

# Resolve the absolute path to the suppressions file.
SUPPRESSIONS_FILE=$(find "${TEST_SRCDIR}" -name "tsan_suppressions.txt" -print -quit)
export TSAN_OPTIONS="halt_on_error=1,history_size=7,suppressions=${SUPPRESSIONS_FILE}"

# Run the actual test command passed by Bazel.
exec "$@"
