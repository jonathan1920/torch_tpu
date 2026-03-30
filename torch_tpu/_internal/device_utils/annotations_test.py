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

"""Tests for layout hint annotations."""

from absl.testing import absltest
import torch
import torch.testing
from torch_tpu import api
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
import torch_tpu._internal.device_utils.annotations as tpu_annotations


class AnnotationsTest(absltest.TestCase):

  def test_layout_hint_eager_direct(self):
    device = api.tpu_device()
    x = torch.ones((10, 10), device=device)
    layout = "{1,0:T(16,128)(2,1)}"

    device_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertIsNone(
        device_layout
    )  # We expect this tensor is still not materialized.

    x = tpu_annotations.layout_hint(x, layout)

    # Materialize the tensor.
    sync.synchronize([x], wait=True)
    device_layout = tpu_torch_compile.get_device_layout_if_materialized(x)
    self.assertEqual(device_layout, layout)

  def test_set_layout_decorator(self):
    device = api.tpu_device()
    layout = "{1,0:T(16,128)(2,1)}"

    @torch.compile(dynamic=False, backend=torch_tpu_compile.TpuBackend())
    @tpu_annotations.set_layout(layout)
    def add(a, b):
      return a + b

    x = tpu_annotations.layout_hint(torch.ones((10, 10), device=device), layout)

    sync.synchronize([x], wait=True)
    y = tpu_annotations.layout_hint(
        torch.zeros((10, 10), device=device), layout
    )
    sync.synchronize([y], wait=True)

    z = add(x, y)
    sync.synchronize([z], wait=True)

    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(x), layout
    )
    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(y), layout
    )
    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(z), layout
    )

  def test_set_layouts_on_inputs_and_output_in_eager_mode(self):
    device = api.tpu_device()
    layout = "{1,0:T(16,128)(2,1)}"

    a = tpu_annotations.layout_hint(
        torch.randn((10, 10), device=device), layout
    )
    b = tpu_annotations.layout_hint(
        torch.randn((10, 10), device=device), layout
    )
    c = a + b
    c = tpu_annotations.layout_hint(c, layout)
    sync.synchronize([c], wait=True)

    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(a), layout
    )

    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(b), layout
    )

    self.assertEqual(
        tpu_torch_compile.get_device_layout_if_materialized(c), layout
    )


if __name__ == "__main__":
  absltest.main()
