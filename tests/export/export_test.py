# Copyright 2025 Google LLC
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

"""Small graph test for TPU backend."""

import os
from absl.testing import absltest
import torch
import torch.export
import torch.fx
from torch_tpu._internal.export import export as torch_tpu_export


class SimpleModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.Tensor(5))
    self.b = torch.nn.Parameter(torch.Tensor(5))
    self.c = torch.nn.Parameter(torch.Tensor(5))

  def forward(self, x, y):
    out = self.a + 0.8 * self.b + x
    out = out + 0.7 * self.c + y
    return out


class SimpleNN(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(8, 8)

  def forward(self, x):
    x = self.linear1(x)
    x = x + torch.ones_like(x)
    return x


class TestExportLinearMode(absltest.TestCase):

  def test_export_linear(self):
    sample_input = (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),)
    exported = torch.export.export(SimpleNN(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertRegex(
        mlir_text,
        r"module @tt_jit_export_test_L[0-9]+C[0-9]+_forward_add",
    )
    self.assertIn(
        "func.func @main(%arg0: tensor<8x8xf32> loc(unknown), %arg1:"
        " tensor<8xf32> loc(unknown), %arg2: tensor<8xf32> loc(unknown)) ->"
        " tensor<8xf32>",
        mlir_text,
    )

    # Check for StableHLO ops
    self.assertIn("stablehlo.add", mlir_text)

    # Check for namescope / opname
    self.assertIn('loc("addmm/as_strided"', mlir_text)
    self.assertIn('loc("addmm/addmm"', mlir_text)

    # Check for file info
    current_filename = os.path.basename(__file__)
    self.assertIn(current_filename, mlir_text)


class ExportTest(absltest.TestCase):

  def test_module(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    with torch_tpu_export.enable_tracebacks(None):  # override with nullopt
      exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertRegex(
        mlir_text,
        r"module @tt_jit_export_test_L[0-9]+C[0-9]+_forward_add",
    )
    self.assertIn(
        "func.func @main(%arg0: tensor<5xf32> loc(unknown), %arg1:"
        " tensor<5xf32> loc(unknown), %arg2: tensor<5xf32> loc(unknown), %arg3:"
        " tensor<5xf32> loc(unknown), %arg4: tensor<5xf32> loc(unknown)) ->"
        " tensor<5xf32>",
        mlir_text,
    )

    # Default export should include debuginfo.
    # Check that the current filename is in the MLIR text with debuginfo.
    # Currently checking for #loc1 so we don't match this filename higher up in
    # the stack trace.
    current_filename = os.path.basename(__file__)
    current_location_regex = f"#loc1 = loc.*{current_filename}"
    self.assertRegex(mlir_text, current_location_regex)

  def test_export_deduplication_different_views(self):
    """Tests that export deduplication handles different views of the same memory correctly."""

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return x, x.view(torch.int32)

    mod = MyModule()
    x = torch.randn(5)

    exported = torch.export.export(mod, args=(x,))
    mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()
    self.assertIn("-> (tensor<5xf32>, tensor<5xi32>)", mlir_text)

  def test_export_deduplication_different_slices(self):
    """Tests that export deduplication does not deduplicate different slices of the same buffer."""

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return x[0:2], x[1:3]

    mod = MyModule()
    x = torch.randn(4)

    exported = torch.export.export(mod, args=(x,))
    mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()

    self.assertIn("-> (tensor<2xf32>, tensor<2xf32>)", mlir_text)

  def test_module_traceback_disabled(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    with torch_tpu_export.enable_tracebacks(False):
      exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertIn("module @tt_jit_add", mlir_text)
    self.assertIn(
        "func.func @main(%arg0: tensor<5xf32> loc(unknown), %arg1:"
        " tensor<5xf32> loc(unknown), %arg2: tensor<5xf32> loc(unknown), %arg3:"
        " tensor<5xf32> loc(unknown), %arg4: tensor<5xf32> loc(unknown)) ->"
        " tensor<5xf32>",
        mlir_text,
    )

    # If tracebacks are explicitly disabled, we should not see any location
    # information in the MLIR text.
    # Expect to see op names and redispatches even if tracebacks are disabled.
    self.assertIn('loc("add/add"', mlir_text)
    self.assertIn('loc("mul/mul"', mlir_text)

    # Expect no python filenames if tracebacks are disabled.
    current_filename = os.path.basename(__file__)
    self.assertNotIn(current_filename, mlir_text)
    self.assertNotRegex(mlir_text, r"loc.*\.py")

  def test_serialize_text_no_debug_info(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)

    # Check for the absence of location #loc(...) in mlir text
    mlir_text = exported_mlir.serialize_text(enable_debug_info=False)

    # Check for module name and main func
    self.assertIn("module @tt_jit", mlir_text)
    self.assertIn("func @main", mlir_text)
    self.assertNotIn("loc(", mlir_text)

  def test_serialize_text_debug_info(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for the presence of location #loc(...) in mlir text
    self.assertIn("loc(", mlir_text)

  def test_serialize_bytecode(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_bytecode = exported_mlir.serialize_bytecode()
    # Check for the MLïR magic string that denotes bytecode.
    self.assertIn(b"ML\xefR", mlir_bytecode)

  def test_serialize_portable_artifact(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_portable_artifact = exported_mlir.serialize_portable_artifact()
    # Check for the MLïR magic string that denotes bytecode.
    # Check for the StableHLO_v1.X.Y producer string to indicate versioned
    # StableHLO.
    self.assertIn(b"ML\xefR", mlir_portable_artifact)
    self.assertIn(b"StableHLO_v1.", mlir_portable_artifact)


class EmptyOutputGraphTest(absltest.TestCase):
  """Graphs with no computed output tensors must compile, not abort.

  A graph whose flattened outputs are all ``None`` has no computed result
  tensors, so ``_process_fx_outputs`` returns an empty ``deduped_outputs``;
  ``fx_to_mlir`` then also drops the unused default generator-state tensor,
  leaving ``result_tensors`` empty and aborting ``traverse_and_compile``
  ("no result tensors provided"). Such all-None segments arise from
  ``torch.compile(fullgraph=False)`` partitioning -- e.g. the in-place
  ``masked_fill_`` before the tensor-parallel all-reduce in a vocab-parallel
  embedding, where the all-reduce forces a graph break.
  """

  def test_all_none_output_graph_compiles(self):
    graph = torch.fx.Graph()
    graph.placeholder("x")  # a real tensor input...
    graph.output((None,))  # ...but no computed output tensor.
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    x = torch.randn(8).to(torch.accelerator.current_accelerator())
    # Before the fix this aborted the process inside traverse_and_compile.
    exported_mlir = torch_tpu_export.fx_to_mlir(gm, [x])
    self.assertIsInstance(exported_mlir, torch_tpu_export.ExportedMlir)


if __name__ == "__main__":
  absltest.main()
