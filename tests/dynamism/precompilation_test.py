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
"""Test demonstrating precompilation of padding and slice modules."""

import time
from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


class ToyModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 10)

  def forward(self, x):
    return self.linear(x)


class MockTpuCompiler:
  """Context manager to mock TPU pad/slice JIT compilation for testing.

  Injects a simulated compilation delay (sleep) on Cache Misses to make the
  speedup from precompilation easily measurable and robust against system noise.
  """

  def __init__(self, compile_delay: float = 1.0):
    self.compile_delay = compile_delay
    self.compiled_pad_shapes = set()
    self.compiled_slice_shapes = set()

    # Store reference to original C++ compiled_mode methods
    self._orig_precompile = (
        _backend.dynamic_compiler.tpu_torch_compile.precompile_pad_module
    )
    self._orig_get_or_compile = (
        _backend.dynamic_compiler.tpu_torch_compile.get_or_compile_pad_module
    )
    self._orig_precompile_slice = (
        _backend.dynamic_compiler.tpu_torch_compile.precompile_slice_module
    )
    self._orig_get_or_compile_slice = (
        _backend.dynamic_compiler.tpu_torch_compile.get_or_compile_slice_module
    )

  def _get_pad_key(self, args, kwargs) -> tuple[tuple[int, ...], ...]:
    tensor_info = args[0] if args else kwargs.get("tensor_info")
    return tuple(tuple(info.shape) for info in tensor_info)

  def _get_slice_key(self, args, kwargs) -> tuple[tuple[int, ...], ...]:
    target_shapes = args[0] if args else kwargs.get("target_shapes")
    return tuple(tuple(s) for s in target_shapes)

  def _precompile_pad(self, *args, **kwargs):
    key = self._get_pad_key(args, kwargs)
    print(f"[Mock] Precompiling shape: {key}")
    self.compiled_pad_shapes.add(key)
    return self._orig_precompile(*args, **kwargs)

  def _get_or_compile_pad(self, *args, **kwargs):
    key = self._get_pad_key(args, kwargs)
    if key in self.compiled_pad_shapes:
      print(f"[Mock] Cache HIT for shape: {key}")
    else:
      print(
          f"[Mock] Cache MISS for shape: {key}, simulating slow compilation..."
      )
      time.sleep(self.compile_delay)
      self.compiled_pad_shapes.add(key)
    return self._orig_get_or_compile(*args, **kwargs)

  def _precompile_slice(self, *args, **kwargs):
    key = self._get_slice_key(args, kwargs)
    print(f"[Mock] Precompiling SLICE shape: {key}")
    self.compiled_slice_shapes.add(key)
    return self._orig_precompile_slice(*args, **kwargs)

  def _get_or_compile_slice(self, *args, **kwargs):
    key = self._get_slice_key(args, kwargs)
    if key in self.compiled_slice_shapes:
      print(f"[Mock] SLICE Cache HIT for shape: {key}")
    else:
      print(
          f"[Mock] SLICE Cache MISS for shape: {key}, simulating slow"
          " compilation..."
      )
      time.sleep(self.compile_delay)
      self.compiled_slice_shapes.add(key)
    return self._orig_get_or_compile_slice(*args, **kwargs)

  def __enter__(self):
    self._patches = [
        mock.patch.object(
            _backend.dynamic_compiler.tpu_torch_compile,
            "precompile_pad_module",
            self._precompile_pad,
        ),
        mock.patch.object(
            _backend.dynamic_compiler.tpu_torch_compile,
            "get_or_compile_pad_module",
            self._get_or_compile_pad,
        ),
        mock.patch.object(
            _backend.dynamic_compiler.tpu_torch_compile,
            "precompile_slice_module",
            self._precompile_slice,
        ),
        mock.patch.object(
            _backend.dynamic_compiler.tpu_torch_compile,
            "get_or_compile_slice_module",
            self._get_or_compile_slice,
        ),
    ]
    for p in self._patches:
      p.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    for p in reversed(self._patches):
      p.stop()


class PrecompilationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch.accelerator.current_accelerator()
    self.prev_grad_status = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

  def tearDown(self):
    torch.set_grad_enabled(self.prev_grad_status)
    super().tearDown()

  def test_precompilation_comparison(self):
    model = ToyModel().to(self.device)
    model.eval()
    backend = _backend.TpuBackend(debug=True, dynamism=True, precompile_steps=0)
    compiled = torch.compile(model, backend=backend)
    with MockTpuCompiler(compile_delay=1.0):
      # --- Phase 1: Without Precompilation ---
      # Shapes: [2, 10], [4, 10], [6, 10]
      input_shapes_1 = [2, 4, 6]
      inputs_1 = [
          torch.randn(shape, 10, device=self.device) for shape in input_shapes_1
      ]
      for x in inputs_1:
        torch._dynamo.mark_dynamic(x, 0, min=2, max=16)
      print("\n--- Starting Phase 1: Without Precompilation ---")
      # Call 1: Compiles model and P2 (slow, 1s sleep)
      compiled(inputs_1[0])
      # Call 2: Compiles P4 (slow, 1s sleep)
      compiled(inputs_1[1])
      # Sleep to match Phase 2 timing
      time.sleep(5)
      # Call 3: Compiles P6 (slow, 1s sleep)
      t0 = time.perf_counter()
      out_without = compiled(inputs_1[2])
      torch.accelerator.synchronize()
      duration_without = time.perf_counter() - t0
      print(
          "Third call WITHOUT precompilation ([6, 10]) took"
          f" {duration_without:.4f} seconds"
      )
      # --- Phase 2: With Precompilation ---
      # Shapes: [3, 10], [5, 10], [7, 10] (different shapes from Phase 1)
      # Reset Dynamo cache to force re-compilation and reset the linear tracker
      torch._dynamo.reset()
      backend = _backend.TpuBackend(
          debug=True, dynamism=True, precompile_steps=1
      )
      compiled = torch.compile(model, backend=backend)
      input_shapes_2 = [3, 5, 7]
      inputs_2 = [
          torch.randn(shape, 10, device=self.device) for shape in input_shapes_2
      ]
      for x in inputs_2:
        torch._dynamo.mark_dynamic(x, 0, min=2, max=16)
      print("\n--- Starting Phase 2: With Precompilation ---")
      # Call 1: Compiles P3 (model is already compiled) (slow, 1s sleep)
      compiled(inputs_2[0])
      # Call 2: Compiles P5 (slow, 1s sleep), and schedules P7 in background
      compiled(inputs_2[1])
      # Sleep to allow background compilation of P7 to finish
      print("Waiting for precompilation to complete...")
      time.sleep(5)
      # Call 3: Should be cache hit for P7, so fast! (no sleep)
      t0 = time.perf_counter()
      out_with = compiled(inputs_2[2])
      torch.accelerator.synchronize()
      duration_with = time.perf_counter() - t0
      print(
          f"Third call WITH precompilation ([7, 10]) took {duration_with:.4f}"
          " seconds"
      )
    # --- Comparisons ---
    print("\nComparison for the 3rd shape on the line:")
    print(f"Without precompilation ([6, 10]): {duration_without:.4f}s")
    print(
        f"With precompilation    ([7, 10]): {duration_with:.4f}s (cache hit!)"
    )
    print(f"Speedup: {duration_without / duration_with:.2f}x")
    # Assertions
    self.assertGreater(
        duration_without, 2.0
    )  # Should include at least 2s sleep
    self.assertLess(duration_with, 0.5)  # Fast call should be < 0.5s
    self.assertGreater(
        duration_without, duration_with * 10
    )  # At least 10x speedup
    # Verifications
    utils.assert_close(out_without.to("cpu"), model(inputs_1[2]).to("cpu"))
    utils.assert_close(out_with.to("cpu"), model(inputs_2[2]).to("cpu"))


if __name__ == "__main__":
  absltest.main()
