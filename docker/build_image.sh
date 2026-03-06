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

# Script to build torch-tpu docker image from a local repository checkout.
# It uses Dockerfile.from_src which copies the local source and builds the
# wheel inside.

set -e

# Usage: ./build_image.sh [LOCAL_REPO_ROOT] [IMAGE_TAG]
# LOCAL_REPO_ROOT: Path to the root of the torch_tpu repository (optional,
#                  defaults to parent dir).
# IMAGE_TAG: Name and tag for the built image (optional, defaults to
#            torch-tpu-local).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default local repo root is the parent of this script's directory.
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

LOCAL_REPO_ROOT=${1:-"${DEFAULT_REPO_ROOT}"}
IMAGE_TAG=${2:-"torch-tpu-local"}

# Resolve absolute path for LOCAL_REPO_ROOT
ABS_REPO_ROOT=$(cd "${LOCAL_REPO_ROOT}" && pwd)

# The Dockerfile path relative to the REPO_ROOT
DOCKERFILE="docker/Dockerfile.from_src"

if [ ! -f "${ABS_REPO_ROOT}/${DOCKERFILE}" ]; then
  echo "Error: Dockerfile not found at ${ABS_REPO_ROOT}/${DOCKERFILE}"
  echo "Make sure LOCAL_REPO_ROOT points to the root of the torch_tpu repository."
  exit 1
fi

echo "===> Building Docker image '${IMAGE_TAG}' using local source at ${ABS_REPO_ROOT}..."

# Build with the root of the repo as the context
docker build \
  --progress=plain \
  -f "${ABS_REPO_ROOT}/${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${ABS_REPO_ROOT}"

echo "===> Done! Image '${IMAGE_TAG}' built successfully."
echo "Repository source is available at /workspace/torch_tpu_src"
echo "Examples are available at /workspace/examples"
echo "You can run it with: docker run -it ${IMAGE_TAG}"
