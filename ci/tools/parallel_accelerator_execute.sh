#!/usr/bin/env bash
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
# ==============================================================================
#
# Coordinates access to accelerators TPUs between concurrent Bazel tests.
#
# Assigns each test an accelerator and ensures that tests are evenly distributed
# between accelerators.
#
# Example usage:
# bazel test --run_under=/path/to/parallel_accelerator_execute.sh //tests/...
#
#
# Environment variables:
#     TORCH_TPU_ACCELERATOR_COUNT =
#         Number of accelerators TPUs available.
#     TORCH_TPU_TESTS_PER_ACCELERATOR =
#         Number of test runs per accelerators TPUs available.

TORCH_TPU_ACCELERATOR_COUNT=${TORCH_TPU_ACCELERATOR_COUNT:-8}
TORCH_TPU_TESTS_PER_ACCELERATOR=${TORCH_TPU_TESTS_PER_ACCELERATOR:-1}
LOCK_DIR="${TORCH_TPU_LOCK_DIR:-/tmp/torch_tpu_locks}"
MAX_WAIT_SECONDS=${TORCH_TPU_LOCK_TIMEOUT_SECONDS:-600}

mkdir -p -m 777 "$LOCK_DIR" 2>/dev/null || true

TEST_BINARY="$1"
if [ ! -f "$TEST_BINARY" ]; then
  f=bazel_tools/tools/bash/runfiles/runfiles.bash
  source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
    source "$(grep -m1 "$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
    source "$0.runfiles/$f" 2>/dev/null || true

  if command -v rlocation >/dev/null 2>&1; then
    resolved="$(rlocation "${TEST_WORKSPACE:-_main}/${1#./}" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -f "$resolved" ] && TEST_BINARY="$resolved"
  fi
fi
shift

# *******************************************************************

# Multi-accelerator/distributed tests require exclusive access across all slots.
if [ "${TORCH_TPU_EXCLUSIVE_TEST:-0}" = "1" ]; then
  echo "Acquiring ALL accelerator locks for exclusive test $TEST_BINARY..."
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      exec {fd}>"${LOCK_DIR}/lock_${i}_${j}" || exit 1
      flock "$fd" || exit 1
    done
  done
  (
    export TPU_VISIBLE_CHIPS="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
    export TPU_VISIBLE_DEVICES="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
    unset TPU_ACCELERATOR_TYPE TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS TPU_TOPOLOGY CHIPS_PER_HOST_BOUNDS HOST_BOUNDS
    "$TEST_BINARY" "$@"
  )
  return_code=$?
  exit $return_code
fi

# Try to acquire any of the
# TORCH_TPU_ACCELERATOR_COUNT * TORCH_TPU_TESTS_PER_ACCELERATOR
# slots to run a test at.
#
# Prefer to allocate 1 test per accelerator over 4 tests on 1 accelerator
# So, we iterate over TORCH_TPU_TESTS_PER_ACCELERATOR
# first (j) then accelerators (i).
#
# If all slots are busy, we loop and retry until one is free.
start_time=$(date +%s)
while true; do
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      exec {lock_fd}>"${LOCK_DIR}/lock_${i}_${j}" || exit 1
      if flock -n "$lock_fd"; then
        (
          export TPU_VISIBLE_CHIPS=$i
          export TPU_VISIBLE_DEVICES=$i
          unset TPU_ACCELERATOR_TYPE TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS TPU_TOPOLOGY CHIPS_PER_HOST_BOUNDS HOST_BOUNDS
          echo "Running test $TEST_BINARY $* on accelerator $i"
          "$TEST_BINARY" "$@"
        )
        return_code=$?
        # flock locks are automatically released when the FD is closed.
        exec {lock_fd}>&-
        exit $return_code
      else
        exec {lock_fd}>&-
      fi
    done
  done

  current_time=$(date +%s)
  if [ $((current_time - start_time)) -ge "$MAX_WAIT_SECONDS" ]; then
    echo "ERROR: Timed out waiting for an accelerator slot after ${MAX_WAIT_SECONDS}s." >&2
    exit 1
  fi
  echo "All accelerator slots are busy, retrying in 1s..."
  sleep 1
done
