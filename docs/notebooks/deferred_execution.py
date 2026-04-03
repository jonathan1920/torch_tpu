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
"""This notebook explains the deferred execution model of TorchTPU."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return mo


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **The Deferred Execution Model**

    This guide provides a deep dive into TorchTPU's most fundamental architectural concept: **Deferred Execution**. Understanding "Promises" and why materialization triggers matter is essential for writing efficient TPU code.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Eager vs. Deferred Execution**

    ### **Eager Execution (CUDA/CPU)**
    In a standard CPU or CUDA backend, when you execute `z = x + y`, the backend:
    1. **Immediately** allocates memory for `z`
    2. **Immediately** dispatches a kernel to perform the addition
    3. The result is available as soon as the kernel finishes running.

    ### **Deferred Execution (TorchTPU)**
    TorchTPU does **not** immediately run a kernel. Instead:
    1. **The Graph:** It records the operation into a **Directed Acyclic Graph (DAG)** — a "recipe" for a future computation.
    2. **The Promise:** The operation returns a tensor backed by a **"promise"** that holds metadata (shape, dtype), but the physical computation is deferred.
    3. **Compilation & Execution:** When data is required by the host (e.g., `.item()`) or triggered by heuristics, the backend **"materializes"** the tensor — optimizing, compiling, and executing the accumulated ops (often in a separate thread to avoid blocking operation recording).

    > [!NOTE]
    > **Balancing Deferral:** The timing of TorchTPU-initiated materializations is carefully balanced by the backend to maximize CPU/TPU overlap while allowing enough graph accumulation for the XLA compiler to generate high-performance executables.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **The "Promise"**

    Every tensor on the TPU is backed by an internal tracking mechanism. In most cases, this object exists in one of two primary states:

    | State | Description | Physical Memory? |
    | :--- | :--- | :--- |
    | **Deferred** | Contains a "recipe" (DAG node) for computing a value | ❌ No TPU memory used |
    | **Materialized** | Contains a physical buffer on TPU HBM | ✅ Occupies real memory |

    **Key insight:** Until materialization, intermediate tensors may never occupy physical memory. The XLA compiler can fuse or optimize them away entirely.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Factory Methods: How Tensors Are Created**

    Different tensor creation methods produce different internal representations:

    | Method | Internal Behavior |
    | :--- | :--- |
    | `torch.ones(...)`, `torch.zeros(...)`, `torch.arange(...)` | Records a StableHLO `Constant` op in the DAG. No physical memory yet. |
    | `torch.empty(...)` | Starts in the **Deferred** state. It uses internal uninitialized fill semantics (often optimized away). |
    | `cpu_tensor.to(device)` | Copies data from CPU to TPU HBM immediately (Host-to-Device transfer). |
    """)
  return


@app.cell
def factory_demo():
  import torch
  from torch_tpu import api
  import tpu_utils

  device = tpu_utils.safe_init()

  # This records a "recipe" to create ones — no physical TPU memory is used yet
  t1 = torch.ones((10, 10), device=device, dtype=torch.bfloat16)
  print(f"t1 shape: {t1.shape}, dtype: {t1.dtype}")
  print("  (Deferred — no TPU compute has happened yet)")

  # This also records a recipe — still no hardware execution
  t2 = torch.arange(
      100, device=device, dtype=torch.bfloat16
  )  # StableHLO Constant
  t3 = t1 + t2.reshape(10, 10)  # Records a reshape (DeferredOp) and add node
  print(f"\nt3 shape: {t3.shape}")
  print("  (Still deferred — the entire chain is just a recipe)")

  # NOW materialization happens — the whole graph is compiled and executed
  result = t3.cpu()
  print(f"\nresult shape: {result.shape}")
  print(
      "  (Materialized — graph was compiled, executed, and data copied to CPU)"
  )
  return device, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Materialization Triggers**

    Materialization is the process where the backend begins optimizing, compiling, and executing the recorded DAG. This often happens **asynchronously**; the backend can start compiling and executing accumulated operations in a separate thread while the main thread continues to record and defer new operations.
 
    Common triggers for materialization include:

    | Trigger | What Happens |
    | :--- | :--- |
    | `.item()` | Extracts a scalar — forces the graph to execute |
    | `.tolist()` / `.numpy()` | Converts to Python/NumPy — forces execution |
    | `print(tensor)` | Must fetch data to display it |
    | `torch.save(tensor)` | Must serialize actual bytes |
    | `if (x > 0).all():` | Data-dependent control flow — implicitly calls `.item()` to evaluate the boolean |
    | `.cpu()` | Device-to-Host transfer |
    | Control Flow / Heuristics | Deeply chained or extremely complex graphs can hit internal heuristics, triggering execution |

    > [!IMPORTANT]
    > Every materialization is a **"graph break."** Frequent graph breaks prevent the XLA compiler from fusing operations, reducing hardware efficiency.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **Heuristic Materialization**

    While host-syncs are the primary triggers, TorchTPU also uses internal heuristics like **"Loop Detection"** to trigger materialization. This prevents the deferred graph from growing indefinitely during repeated executions, which could otherwise lead to excessive memory consumption or compilation complexity.

    In general, standard operations simply add nodes to the DAG and do not trigger materialization on their own.
    """)
  return


@app.cell
def item_demo(device, torch):
  print("\n--- .item() Materialization Demo ---")

  # Create two scalar tensors
  v1 = torch.tensor(5.0, device=device, dtype=torch.bfloat16)
  v2 = torch.tensor(3.0, device=device, dtype=torch.bfloat16)
  print("  (Scalars created — deferred)")

  # Perform a comparison
  cond = v1 > v2
  print("  (Comparison recorded — deferred)")

  # Using .item() forces materialization to get the python boolean
  is_greater = cond.item()
  print(f"  (Materialized via .item() — Result: {is_greater})")

  # Using the tensor in an `if` statement implicitly calls .item()
  print("  (Evaluating `if cond:`...)")
  if cond:
    print("  (Materialized implicitly via control flow)")

  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **View-Only Operations**

    Many operations appear to be "free" transformations of the internal shape and stride metadata:

    * `view()` — Reshapes without moving data
    * `transpose()` — Reinterprets strides
    * `as_strided()` — Updates stride metadata
    * `permute()` — Reorders dimensions logically

    These operations are not strictly free. While they only update metadata initially, they create new "promises" (DeferredOps) representing the transformed state on their next use.

    > [!WARNING]
    > **When Views Are Expensive:** While the view operation itself is free, it can cause severe performance issues later. If a view dramatically changes the memory layout (strides) of a tensor, the XLA compiler may be forced to insert an expensive, physical memory re-layout operation (a `Copy`) before the next computationally intensive kernel (like a `matmul` or `conv`) can consume it. This is a common performance pitfall when transitioning from CUDA to TPU.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Debugging Mode: `EagerMode.DEFER_NEVER`**

    If your model produces `NaN`s or `Inf`s, deferred execution makes debugging hard because the error only surfaces at the next materialization point — not the line that caused it.

    You can force TorchTPU to execute every operation immediately using **Eager Mode Defer Never**:

    ```python
    from torch_tpu._internal import execution_mode as em

    # Force every operation to materialize immediately
    with em.eager_mode(em.EagerMode.DEFER_NEVER):
        # Now you can pinpoint the exact line where a NaN is generated
        output = model(input_data)
    ```

    | Mode | Description |
    | :--- | :--- |
    | `EagerMode.DEFER_AND_FUSE_WITH_O1` | **Default.** Defers operations into fusion clusters, except those that must be executed immediately. |
    | `EagerMode.DEFER_AND_FUSE` | **Optimized mode** Uses more aggressive XLA optimizations for eager execution. |
    | `EagerMode.DEFER_NEVER` | **Debug mode.** Every op is dispatched immediately, similar to how it's done in CUDA. |
    | `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` | **Extreme Debug mode.** Executes ops immediately and waits for completion before dispatching the next. |
    | `EagerMode.INTERNAL_DEFER_ALL` | Strictly defers all operations. Used exclusively for internal `torch.compile` workflows. |

    > [!WARNING]
    > `EagerMode.DEFER_NEVER` makes execution **significantly slower** because it eliminates compiler fusions. Use it only for debugging.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Why Deferred Execution Matters**

    The deferred model is TorchTPU's key performance advantage:

    1. **Global Optimization:** The XLA compiler sees a block of ops at once, enabling aggressive fusion that reduces memory traffic and maximizes MXU utilization.
    2. **Memory Efficiency:** Intermediate tensors that are never "read" by the host may never occupy physical memory — the compiler can optimize them out or fuse them.
    3. **Eliminates Kernel Launch Overhead:** Instead of dispatching dozens of individual kernels, the compiler generates a few highly optimized hardware executions.

    **The trade-off:** changes in deferral behavior can cause changes in compile times, runtimes, or both, even for the same logical operations. Excessive graph breaks negate the benefits of deferred execution, while too few graph breaks can be slow to compile.
    """)
  return


if __name__ == "__main__":
  app.run()
