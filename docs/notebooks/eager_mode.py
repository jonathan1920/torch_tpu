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
"""This notebook explains the Fused Eager execution model of TorchTPU."""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Fused Eager**

    This guide provides an overview of TorchTPU's most fundamental architectural concept: **Fused Eager**. Fused Eager enables high-performance execution by grouping multiple operations together, allowing the XLA compiler to optimize across operation boundaries.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Eager vs. Fused Eager**

    ### **Strict Eager Execution (CUDA/CPU)**
    In a standard CPU or CUDA backend, operations are executed one by one as they are called. When you execute `z = x + y`, the backend:
    1. Immediately dispatches a kernel to perform the addition.
    2. The result is available as soon as the kernel finishes running.

    ### **Fused Eager (TorchTPU)**
    TorchTPU defaults to **Fused Eager**. Instead of running every operation individually, it:
    1. **Fusion:** Records operations to be executed together as a group.
    2. **Optimization:** Uses the XLA compiler to optimize the entire group of operations at once.
    3. **Efficient Execution:** Dispatches highly optimized hardware kernels that combine multiple logical operations, reducing memory traffic and maximizing TPU hardware utilization.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Factory Methods: How Tensors Are Created**

    Different tensor creation methods produce different internal behaviors:

    | Method | Behavior |
    | :--- | :--- |
    | `torch.ones(...)`, `torch.zeros(...)`, `torch.arange(...)` | These are fused into subsequent operations. |
    | `torch.empty(...)` | Starts in the **Fused Eager** state. |
    | `cpu_tensor.to(device)` | Copies data from CPU to TPU HBM immediately. |
    """)
  return


@app.cell
def factory_demo():
  import torch
  from torch_tpu import api
  import tpu_utils

  device = api.tpu_device()

  # This is fused — no physical TPU compute is used yet
  t1 = torch.ones((10, 10), device="tpu", dtype=torch.bfloat16)
  print(f"t1 shape: {t1.shape}, dtype: {t1.dtype}")
  print("  (Fused — no TPU compute has happened yet)")

  # This is also fused — still no hardware execution
  t2 = torch.arange(100, device="tpu", dtype=torch.bfloat16)
  t3 = t1 + t2.reshape(10, 10)
  print(f"\nt3 shape: {t3.shape}")
  print("  (Still fused — the operations are waiting to be optimized)")

  # Transferring to CPU triggers the execution of the fused operations
  result = t3.cpu()
  print(f"\nresult shape: {result.shape}")
  print(
      "  (Executed — fused operations were optimized and executed, data copied"
      " to CPU)"
  )
  return (torch,)


@app.cell
def item_demo(torch):
  print("\n--- .item() Execution Demo ---")

  # Create two scalar tensors
  v1 = torch.tensor(5.0, device="tpu", dtype=torch.bfloat16)
  v2 = torch.tensor(3.0, device="tpu", dtype=torch.bfloat16)
  print("  (Scalars created — fused)")

  # Perform a comparison
  cond = v1 > v2
  print("  (Comparison recorded — fused)")

  # Using .item() triggers execution to get the python boolean
  is_greater = cond.item()
  print(f"  (Executed via .item() — Result: {is_greater})")

  # Using the tensor in an `if` statement implicitly triggers execution
  print("  (Evaluating `if cond:`...)")
  if cond:
    print("  (Executed implicitly via control flow)")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **View Operations**

    Many operations like `view()`, `transpose()`, and `permute()` are fused with subsequent operations.

    > [!WARNING]
    > **Performance Impact of Views:** While views themselves are efficient, they can influence how the XLA compiler fuses subsequent operations. If a view dramatically changes the memory layout, it may require the compiler to insert a layout transformation before the next computationally intensive kernel (like a `matmul` or `conv`).
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Execution Modes**

    While **Fused Eager** is the default for performance, TorchTPU provides other execution modes for debugging. If your model produces `NaN`s or `Inf`s, Fused Eager can make debugging difficult because the error only surfaces when the fused block is executed — not necessarily at the exact line that caused it.

    You can change the execution mode using `eager_mode`:

    ```python
    from torch_tpu._internal import execution_mode as em

    # Force every operation to execute immediately for easier debugging
    with em.eager_mode(em.EagerMode.DEFER_NEVER):
        # Now you can pinpoint the exact line where an error occurs
        output = model(input_data)
    ```

    | Mode | Concept | Description |
    | :--- | :--- | :--- |
    | `EagerMode.DEFER_AND_FUSE` | **Fused Eager** | **The default mode.** Fuses operations and uses aggressive XLA optimizations for high performance. |
    | `EagerMode.DEFER_NEVER` | **Strict Eager** | Every operation is dispatched immediately. Useful for basic debugging. |
    | `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` | **Debug Eager** | Executes operations immediately and waits for completion. Best for pinpointing errors. |

    > [!WARNING]
    > **Strict Eager** and **Debug Eager** modes are significantly slower because they eliminate compiler fusions. Use them only for debugging.
    """)
  return


@app.cell
def modes_demo(torch):
  from torch_tpu._internal import execution_mode as em
  from torch_tpu._internal.sync import sync

  def check_fusion(mode_name, mode):
    print(f"--- Mode: {mode_name} ---")
    with em.eager_mode(mode):
      x = torch.ones((2, 2), device="tpu")
      y = x + 1
      z = y * 2

      # computation_mlir returns the StableHLO graph for the given tensor.
      # If fused, it will contain 'add' and 'multiply' operations.
      mlir = sync.computation_mlir(z)
      has_add = "stablehlo.add" in mlir
      has_mul = "stablehlo.multiply" in mlir

      print(f"  Graph contains fused 'add': {has_add}")
      print(f"  Graph contains fused 'multiply': {has_mul}")
      if has_add and has_mul:
        print("  Result: Operations are FUSED into a single graph.")
      else:
        print("  Result: Operations were EXECUTED immediately (not fused).")
    print()

  check_fusion("Fused Eager (DEFER_AND_FUSE)", em.EagerMode.DEFER_AND_FUSE)
  check_fusion("Strict Eager (DEFER_NEVER)", em.EagerMode.DEFER_NEVER)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Why Fused Eager Matters**

    **Fused Eager** is TorchTPU's key performance advantage:

    1. **Fusion & Optimization:** The XLA compiler optimizes groups of operations together, enabling aggressive fusion that reduces memory traffic and maximizes TPU utilization.
    2. **Memory Efficiency:** Fusing operations allows the compiler to optimize intermediate memory usage, often eliminating the need to store intermediate tensors.
    3. **Reduced Overhead:** Grouping operations reduces the overhead of dispatching many small kernels, leading to better end-to-end performance.
    """)
  return


if __name__ == "__main__":
  app.run()
