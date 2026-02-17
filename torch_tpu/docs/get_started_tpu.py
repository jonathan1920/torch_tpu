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

# pylint: skip-file
import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Get Started with TorchTPU

    This tutorial will guide you through your first successful computation on Google TPU hardware using the `torch-tpu` backend.

    ### Prerequisites
    Before running this notebook, ensure your environment is ready:
    1. **Hardware**: You are running on a TPU VM (e.g., v6e).
    2. **Driver**: `libtpu` is installed.
    3. **Backend**: `torch_tpu` is installed.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. Initialization: The Hardware Handshake

    Unlike a GPU, the TPU backend must be explicitly initialized. This triggers the **PjRt handshake**, which discovers the hardware and registers the "tpu" device string.

    **Note:** This MUST be called before creating any tensors.
    """)
  return


@app.cell
def _():
  import torch
  from tpu_utils import safe_init

  # Self-healing hardware initialization
  device = safe_init()

  # Verify the device
  print(f"Connected to: {device}")
  return device, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. Deferred Execution: Creating Tensors

    TorchTPU uses **Deferred Execution**. Tensors created on TPU are "promises" in a graph. No math happens until you specifically ask for the data.
    """)
  return


@app.cell
def _(device, torch):
  # These are recorded as graph nodes, not physical allocations.
  a = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
  b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)

  print("Graph updated with 'ones' and 'randn' operations.")
  return a, b


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Performing Computation

    We will now perform a Matrix Multiplication. This is also deferred and will be optimized by the XLA compiler later.
    """)
  return


@app.cell
def _(a, b, torch):
  # This adds a 'Dot' operation to our DAG recipe.
  c = torch.matmul(a, b)
  print("Matmul operation added to the deferred graph.")
  return (c,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 4. Materialization: The Execution Trigger

    To see the result, we call `.cpu()`. This triggers the **XLA Compiler** to fuse the operations and run them on the hardware in one optimized block.
    """)
  return


@app.cell
def _(c):
  # This triggers Compilation and Hardware Execution
  final_result = c.cpu()

  print(f"Result Checksum (Sum): {final_result.sum():.4f}")
  return (final_result,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### 🎉 Success!
    You have successfully run a deferred computation on a TPU.
    """)
  return


if __name__ == "__main__":
  app.run()
