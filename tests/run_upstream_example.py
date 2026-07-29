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
import sys
from absl import app


def main(argv=None):
  del argv  # Unused.
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

  script_dir = str(pathlib.Path(example_file_path).parent)
  if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
  if pytorch_examples_dir not in sys.path:
    sys.path.insert(0, pytorch_examples_dir)

  runpy.run_path(example_file_path, run_name="__main__")


if __name__ == "__main__":
  app.run(main, argv=[sys.argv[0]])
