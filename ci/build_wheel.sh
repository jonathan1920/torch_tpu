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
#
# Kokoro orchestration for the TorchTPU wheel build, located in the OSS repo.
#
# This script is orchestration ONLY: it provisions the build container and
# collects artifacts for Kokoro. Everything about how the wheel is actually
# built -- the bazel target, flags, and version-suffix handling -- lives in
# the shared ci/build_wheels.sh, which the GitHub CI uses too, so the wheels
# published from here are built exactly like the wheels CI smoke-tests. The
# only Kokoro-specific bazel flag is --config=no_rbe: this environment has no
# RBE credentials.

set -exu -o history -o allexport

echo "===> Starting Python wheel build in Kokoro..."

# Prepare wheel version string with metadata date suffix (similar to GitHub)
WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS:-.dev$(date +%Y%m%d%H%M%S)}"
export WHEEL_VERSION_EXTRAS
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"
export TORCH_TPU_SRC="$(pwd)"

# Define target wheel dir inside Kokoro artifacts folder
KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
export KOKORO_ARTIFACTS_DIR
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"
mkdir -p "${WHEEL_DIR}"

# We use the standard ml-build container toolchain to build the wheel hermetically
CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

echo "===> Pulling ml-build container: ${CONTAINER_IMAGE}..."
docker pull "${CONTAINER_IMAGE}"

echo "===> Compiling wheel via Bazel inside ml-build container..."
# We map the checked out directory into /workspace and run the shared build
# script for every supported Python version.
mkdir -p dist
docker run --rm \
  -v "$(pwd):/workspace" \
  -w "/workspace" \
  "${CONTAINER_IMAGE}" \
  ci/build_wheels.sh --output-dir /workspace/dist 3.11 3.12 3.13 3.14 -- --config=no_rbe

# Move the built wheels from local dist back to Kokoro artifacts directory
if [[ -d dist && -n "$(ls -A dist/*.whl 2>/dev/null)" ]]; then
  cp dist/*.whl "${WHEEL_DIR}/"
  echo "===> Wheels created and copied to Kokoro artifacts successfully:"
  ls -lh "${WHEEL_DIR}"
else
  echo "ERROR: Python wheel build failed - No .whl files found in dist/"
  exit 1
fi

# Perform fatal inline Twine checks over primary torch_tpu wheels
echo "===> Running Twine check over torch_tpu wheels..."
docker run --rm \
  -v "${WHEEL_DIR}:/dist" \
  "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/torch_tpu-*.whl"

echo "===> torch_tpu wheel build and verification successful!"

# ==============================================================================
# NON-FATAL SECONDARY STAGE: tpu_raiden Python Wheel Build & Verification
# ==============================================================================
# All raiden wheel generation and metadata checks are grouped below and run in a
# non-fatal warning construct. If raiden build or metadata check fails, broken
# artifacts are removed from WHEEL_DIR and execution continues so torch_tpu upload
# is never blocked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_DIR="${RAIDEN_DIR:-${SCRIPT_DIR}/../../tpu_raiden}"

if [[ ! -d "${RAIDEN_DIR}" ]]; then
  echo "===> [Non-Fatal Stage] tpu_raiden directory not found at '${RAIDEN_DIR}'. Cloning from GitHub..."
  git clone https://github.com/google/tpu-raiden.git "${RAIDEN_DIR}" || {
    echo "WARNING: Failed to clone tpu-raiden from GitHub." >&2
  }
fi

if [[ -f "${RAIDEN_DIR}/ci/build_wheel.sh" ]]; then
  echo "===> [Non-Fatal Stage] Invoking tpu_raiden wheel build: ${RAIDEN_DIR}/ci/build_wheel.sh..."
  (
    export WITH_TORCH=1
    export KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR}"
    export WHEEL_DIR="${WHEEL_DIR}"
    bash "${RAIDEN_DIR}/ci/build_wheel.sh"
  ) || {
    echo "WARNING: tpu_raiden wheel build failed. Continuing with torch_tpu wheels only..." >&2
  }

  # Warn-only Twine check on generated tpu_raiden wheels (remove broken wheel if invalid)
  if ls "${WHEEL_DIR}"/tpu_raiden_torch-*.whl >/dev/null 2>&1; then
    echo "===> [Non-Fatal Stage] Running Twine check over tpu_raiden wheels..."
    docker run --rm \
      -v "${WHEEL_DIR}:/dist" \
      "${CONTAINER_IMAGE}" \
      bash -c "uv run --isolated --with twine twine check /dist/tpu_raiden_torch-*.whl" || {
        echo "WARNING: tpu_raiden wheel failed Twine metadata check. Removing broken wheel so torch_tpu upload proceeds..." >&2
        rm -f "${WHEEL_DIR}"/tpu_raiden_torch-*.whl
      }
  fi
else
  echo "WARNING: RAIDEN_DIR/ci/build_wheel.sh not found at '${RAIDEN_DIR}'. Skipping tpu_raiden wheel build." >&2
fi
# ==============================================================================

echo "===> Kokoro wheel build successful!"
