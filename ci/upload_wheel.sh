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
# A script to upload Python wheels for TorchTPU to Artifact Registry.

set -exu -o history -o allexport

# Define target wheel dir inside Kokoro artifacts folder
KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"

CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

# Upload to Google Artifact Registry via Twine and keyring auth, mirroring the GitHub action process.
export UPLOAD_WHEEL_TO_AR=${UPLOAD_WHEEL_TO_AR:-true}
if [[ "${UPLOAD_WHEEL_TO_AR}" == "true" ]]; then
  echo "===> Uploading wheels to internal Artifact Registry..."
  docker run --rm \
    -v "${WHEEL_DIR}:/dist" \
    "${CONTAINER_IMAGE}" \
    bash -c "
      uv run --isolated \
        --with twine \
        --with keyrings.google-artifactregistry-auth \
        twine upload --repository-url https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-testing-registry/ /dist/*.whl
    "
  echo "===> Wheels uploaded successfully to Artifact Registry!"
fi
