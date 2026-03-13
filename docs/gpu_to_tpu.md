# What you should know when moving from GPU to TPU

When migrating workloads from PyTorch on GPUs (CUDA) to PyTorch on TPUs, there
are several key architectural and behavioral differences to be aware of.

## Precision Discrepancies

A major difference between GPU and TPU environments lies in how floating-point
precision is handled, particularly for matrix multiplications (e.g., `matmul`,
`conv`) when the inputs are nominally `float32`.

On newer generation NVIDIA GPUs (Ampere and later), PyTorch defaults
`torch.backends.cuda.matmul.allow_tf32 = True`. This means that when you provide
`float32` tensors as inputs to matrix multiplications, the actual internal
computation may use TensorFloat-32 (TF32). TF32 truncates the mantissa of
`float32` numbers to 10 bits (from 23), while keeping the 8-bit exponent, to
accelerate computation using the Tensor Cores. Accumulation still happens in
`float32`. This implicit downcasting can lead to unpredictable precision drops
across different GPU generations, as older GPUs would perform these operations
in full `float32`. You can control this behavior using
`torch.set_float32_matmul_precision`.

**On TPU, matrix multiplications in the MXU (Matrix Multiply Unit) natively
operate on `bfloat16` inputs and accumulate results in `float32`.** When
`float32` tensors are provided (or other data types like `float16`), they are
typically rounded to `bfloat16` before the multiplication. This default behavior
(equivalent to `torch_tpu.Precision`'s `DEFAULT` mode) provides maximum
performance but has less precision (7 mantissa bits) than IEEE `float32` (23
mantissa bits).

If you need higher precision on TPU for `float32` inputs, you MUST explicitly
configure the `torch_tpu.precision` context manager. This controls the precision
emulation used by the TPU MXU, often involving multiple internal passes (e.g., 3
passes for `HIGH` and 6 passes for `HIGHEST`) to simulate results closer to true
`float32` multiplication, at the cost of performance.

**Handling of `bfloat16` Inputs:**

It's also worth noting how inputs that are already in `bfloat16` format are
handled:

*   **TPU:** `bfloat16` inputs are processed directly by the MXU in their native
    format, with accumulation in `float32`.
*   **GPU (Ampere and later):** These GPUs also have native support for
    `bfloat16` operations within their Tensor Cores. Similar to TPUs, `bfloat16`
    inputs will be multiplied as `bfloat16`, with accumulation likely performed
    in `float32`.

So, when the input tensors are already `bfloat16`, the core matrix
multiplication behavior is similar between TPUs and newer NVIDIA GPUs, both
leveraging native `bfloat16` computation for speed. The primary difference
discussed above arises when the *input* tensors are `float32`, where the default
behaviors (TF32 on GPU vs. direct `bfloat16` conversion on TPU) and methods for
achieving higher precision differ.

```python
import torch
import torch_tpu

a = torch.randn(10, 10, dtype=torch.float32, device="tpu")
b = torch.randn(10, 10, dtype=torch.float32, device="tpu")

# Default computations are natively bfloat16 (fastest, lowest precision)
c = torch.matmul(a, b)

# Explicitly raise precision for more accuracy
with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
    c_accurate = torch.matmul(a, b)
```

The `torch_tpu.precision` context manager is a direct binding to the underlying
StableHLO specification. Please consult the
[StableHLO dot_general documentation](https://openxla.org/stablehlo/spec#dot_general)
for exact numerical definitions of `DEFAULT`, `HIGH`, and `HIGHEST`.
