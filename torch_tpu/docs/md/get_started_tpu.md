---
title: Get Started Tpu
marimo-version: 0.19.9
width: medium
---

```python {.marimo hide_code="true"}
import marimo as mo
```

# Get Started with TorchTPU

This tutorial will guide you through your first successful computation on Google TPU hardware using the `torch-tpu` backend.

### Prerequisites
Before running this notebook, ensure your environment is ready:
1. **Hardware**: You are running on a TPU VM (e.g., v6e).
2. **Driver**: `libtpu` is installed.
3. **Backend**: `torch-tpu` is installed.
<!---->
## 1. Initialization: The Hardware Handshake

Unlike a GPU, the TPU backend must be explicitly initialized. This triggers the **PjRt handshake**, which discovers the hardware and registers the "tpu" device string.

**Note:** This MUST be called before creating any tensors.

```python {.marimo}
import torch
from tpu_utils import safe_init

# Self-healing hardware initialization
device = safe_init()

# Verify the device
print(f"Connected to: {device}")
```

## 2. Deferred Execution: Creating Tensors

TorchTPU uses **Deferred Execution**. Tensors created on TPU are "promises" in a graph. No math happens until you specifically ask for the data.

```python {.marimo}
# These are recorded as graph nodes, not physical allocations.
a = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)

print("Graph updated with 'ones' and 'randn' operations.")
```

## 3. Performing Computation

We will now perform a Matrix Multiplication. This is also deferred and will be optimized by the XLA compiler later.

```python {.marimo}
# This adds a 'Dot' operation to our DAG recipe.
c = torch.matmul(a, b)
print("Matmul operation added to the deferred graph.")
```

## 4. Materialization: The Execution Trigger

To see the result, we call `.cpu()`. This triggers the **XLA Compiler** to fuse the operations and run them on the hardware in one optimized block.

```python {.marimo}
# This triggers Compilation and Hardware Execution
final_result = c.cpu()

print(f"Result Checksum (Sum): {final_result.sum():.4f}")
```

### 🎉 Success!
You have successfully run a deferred computation on a TPU.