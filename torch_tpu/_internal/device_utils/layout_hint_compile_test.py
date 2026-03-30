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

"""Tests for layout hint annotations in the resulting MLIR."""

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal.compile import tpu_torch_compile
import torch_tpu._internal.device_utils.annotations as tpu_annotations


class LayoutHintCompileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()

  def test_output_layout_hint_in_mlir(self):
    x = tpu_torch_compile.placeholder([10], torch.float32, False)
    y = tpu_torch_compile.placeholder([10], torch.float32, False)
    layout = "output_layout_hint"

    z = x + y
    # Annotate output z
    z_hint = tpu_annotations.layout_hint(z, layout)

    mlir = tpu_torch_compile.build_mlir([z_hint], [x, y])
    mlir_str = str(mlir)

    # Verify that the layout hint is in the MLIR as an attribute on the result
    self.assertIn(f'mhlo.layout_mode = "{layout}"', mlir_str)

  def test_multiple_layout_hints_in_mlir(self):
    x = tpu_torch_compile.placeholder([10], torch.float32, False)
    y = tpu_torch_compile.placeholder([10], torch.float32, False)
    layout_out_1 = "out_hint_1"
    layout_out_2 = "out_hint_2"

    z = x + y
    z_hint = tpu_annotations.layout_hint(z, layout_out_1)

    w = x * y
    w_hint = tpu_annotations.layout_hint(w, layout_out_2)

    mlir = tpu_torch_compile.build_mlir([z_hint, w_hint], [x, y])
    mlir_str = str(mlir)

    self.assertIn(f'mhlo.layout_mode = "{layout_out_1}"', mlir_str)
    self.assertIn(f'mhlo.layout_mode = "{layout_out_2}"', mlir_str)


if __name__ == "__main__":
  absltest.main()
