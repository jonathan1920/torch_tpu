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

"""Upstream PyTorch Examples Runner for TorchTPU.

This script executes standalone scripts from the external pytorch/examples
repository (such as regression/main.py, fx/invert.py, fx/module_tracer.py)
against CPU or TPU devices in the TorchTPU CI environment.

Repo: https://github.com/pytorch/examples
"""

import os
import pathlib
import runpy
import shlex
import shutil
import sys
from absl import app

# Pre-import torch and torch_tpu to initialize C++ extensions and TPU runtime
# before any upstream script imports numpy or other native libraries.
import torch  # pylint: disable=unused-import  # noqa: F401
import torch_tpu  # pylint: disable=unused-import  # noqa: F401

_DATASET_BASE_DIR = "/tmp/gcsfuse/data"


def _copy_cora_dataset():
  """Stages Cora dataset from GCS Fuse mount into local './cora' directory.

  Upstream PyTorch GAT example (gat/main.py) looks for './cora/cora.content' and
  './cora/cora.cites' in the current working directory. If missing, it downloads
  cora.tgz from linqs-data.soe.ucsc.edu. In CI and internal sandbox
  environments,
  we avoid external network downloads by staging the Cora dataset from the
  pre-mounted GCS bucket ('torchtpu-shared').

  Raises:
    FileNotFoundError: If the expected dataset files do not exist under the
      GCS mount directory.
  """
  # 1. Locate source Cora dataset directory (pre-mounted via GCS fuse at _DATASET_BASE_DIR/cora).
  gcs_cora_dir = os.path.join(_DATASET_BASE_DIR, "cora")

  gcs_content = os.path.join(gcs_cora_dir, "cora.content")
  gcs_cites = os.path.join(gcs_cora_dir, "cora.cites")
  if not os.path.exists(gcs_content) or not os.path.exists(gcs_cites):
    raise FileNotFoundError(
        f"GCS dataset files '{gcs_content}' or '{gcs_cites}' do not exist."
        " Ensure GCS bucket 'torchtpu-shared' is mounted at"
        f" '{_DATASET_BASE_DIR}'."
    )

  # 2. Prepare local './cora' directory in the current working directory where gat/main.py expects it.
  local_cora_dir = os.path.join(os.getcwd(), "cora")
  os.makedirs(local_cora_dir, exist_ok=True)

  # 3. Copy dataset files locally so gat/main.py detects them and skips network downloads.
  content_path = os.path.join(local_cora_dir, "cora.content")
  cites_path = os.path.join(local_cora_dir, "cora.cites")

  if not os.path.exists(content_path):
    shutil.copyfile(gcs_content, content_path)
  if not os.path.exists(cites_path):
    shutil.copyfile(gcs_cites, cites_path)


def _setup_datasets(example: str):
  match example:
    case "gat/main.py":
      _copy_cora_dataset()
    case _:
      pass


def main(argv=None):
  del argv  # Unused.
  if not os.path.exists(_DATASET_BASE_DIR):
    raise FileNotFoundError(
        f"GCS bucket data directory '{_DATASET_BASE_DIR}' does not exist."
        " Ensure GCS bucket 'torchtpu-shared' is mounted at '/tmp/gcsfuse'."
    )

  pytorch_examples_dir = os.environ.get(
      "TORCH_TPU_INTERNAL_PYTORCH_EXAMPLES_DIR"
  )
  if not pytorch_examples_dir:
    raise EnvironmentError(
        "TORCH_TPU_INTERNAL_PYTORCH_EXAMPLES_DIR is not set."
    )

  rel_path = os.environ.get(
      "TORCH_TPU_INTERNAL_EXAMPLE_PATH", "regression/main.py"
  )
  example_file_path = str(pathlib.Path(pytorch_examples_dir) / rel_path)
  if not os.path.exists(example_file_path):
    raise FileNotFoundError(f"Example file not found: {example_file_path}")

  extra_args = os.environ.get("TORCH_TPU_INTERNAL_EXAMPLE_ARGS", "")
  sys.argv = [example_file_path] + (
      shlex.split(extra_args) if extra_args else []
  )

  _setup_datasets(rel_path)

  script_dir = str(pathlib.Path(example_file_path).parent)
  if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
  if pytorch_examples_dir not in sys.path:
    sys.path.insert(0, pytorch_examples_dir)

  runpy.run_path(example_file_path, run_name="__main__")


if __name__ == "__main__":
  app.run(main, argv=[sys.argv[0]])
