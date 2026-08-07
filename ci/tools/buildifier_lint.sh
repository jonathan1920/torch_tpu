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

# This script checks and applies Bazel BUILD/bzl formatting using `buildifier`.
#
# Design Intent (Incremental Formatting & Linting):
#   Instead of running buildifier across the entire repository, this script uses
#   `git diff` against a base tracking branch ($BASE_SHA). It strictly evaluates
#   and fixes only the specific Bazel syntax files modified in the current Pull
#   Request or local working tree.
#
# Modes:
#   - 'lint'   : Used by CI. Detects formatting and linting violations, outputs
#                self-serve fix instructions, and exits with a failure code.
#   - 'format' : Used locally by developers. Automatically applies buildifier fixes
#                and formatting in-place without failing.

set -e

# Pin specific release version of buildifier for reproducible CI behavior
BUILDIFIER_VERSION="8.5.1"

# Fall back to origin/main if BASE_SHA isn't passed by CI.
BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1}"

# Ensure buildifier binary is available or install/download dynamically
if ! command -v buildifier &> /dev/null; then
    echo "INFO: buildifier not found in PATH. Attempting to install or download v${BUILDIFIER_VERSION}..."
    if command -v go &> /dev/null; then
        echo "INFO: Installing buildifier v${BUILDIFIER_VERSION} via go install..."
        go install github.com/bazelbuild/buildtools/buildifier@v${BUILDIFIER_VERSION} 2>/dev/null || true
        export PATH="$PATH:$(go env GOPATH)/bin:$HOME/go/bin:/usr/local/bin"
    fi
    if ! command -v buildifier &> /dev/null; then
        echo "INFO: Downloading pre-compiled buildifier v${BUILDIFIER_VERSION} binary directly from GitHub Releases..."
        mkdir -p "$HOME/.local/bin"
        curl -sSL "https://github.com/bazelbuild/buildtools/releases/download/v${BUILDIFIER_VERSION}/buildifier-linux-amd64" -o "$HOME/.local/bin/buildifier"
        chmod +x "$HOME/.local/bin/buildifier"
        export PATH="$PATH:$HOME/.local/bin"
    fi
    if ! command -v buildifier &> /dev/null; then
        echo "ERROR: Could not find or install buildifier." >&2
        exit 3
    fi
fi

# Get the list of added, copied, modified, renamed, or type-changed Bazel/Starlark files against $BASE_SHA
BUILDIFIER_PATTERNS=(
    "*BUILD*"
    "*.bzl"
    "*.star"
    "BUILD"
    "BUILD.bazel"
)

GREP_PATTERN='(BUILD|BUILD\.bazel|\.bzl|\.star)$'

BAZEL_FILES=$(git diff --name-only --diff-filter=ACMRT "$BASE_SHA" -- "${BUILDIFIER_PATTERNS[@]}" 2>/dev/null | grep -E "$GREP_PATTERN" || true)

# Ensure a valid mode is explicitly passed; fail on unexpected args.
case "$MODE" in
  lint)
    echo "INFO: Running buildifier check on modified Bazel files against $BASE_SHA..."
    if [ -n "$BAZEL_FILES" ]; then
        # Check rule violations and formatting
        if ! echo "$BAZEL_FILES" | xargs buildifier -mode=check -lint=warn -warnings=-out-of-order-load; then
            echo "================================================================="
            echo "ERROR: Bazel BUILD/bzl style issues found in modified files."
            echo "================================================================="
            echo "To fix automatically:"
            echo "1. Run the formatter directly via nox: nox -e format"
            echo "   (or run: ci/tools/buildifier_lint.sh format)"
            echo "================================================================="
            exit 1
        fi
    else
        echo "INFO: No modified Bazel/Starlark files found in git diff against $BASE_SHA."
    fi
    echo "INFO: buildifier check passed successfully."
    ;;

  format)
    echo "INFO: Applying buildifier fixes and formatting to modified Bazel files..."
    if [ -n "$BAZEL_FILES" ]; then
        echo "$BAZEL_FILES" | xargs buildifier -mode=fix -lint=fix -warnings=-out-of-order-load || true
    else
        echo "INFO: No modified Bazel BUILD/bzl files found in git diff against $BASE_SHA."
    fi
    echo "INFO: Bazel buildifier formatting complete."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'format'." >&2
    echo "Usage: $0 {lint|format}" >&2
    exit 2
    ;;
esac
