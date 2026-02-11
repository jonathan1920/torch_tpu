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

"""Tests for the module_registry.

This test suite verifies the functionality of the `ModuleRegistry` class and its
underlying providers (Torchvision, TIMM, and Transformers). It ensures that:
1.  All providers can list their available modules.
2.  The registry can aggregate these lists correctly.
3.  Specific modules from each provider can be instantiated and executed
    successfully with their generated sample inputs.
"""

from absl.testing import absltest
from tests import module_registry


class ModuleRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module_registry = module_registry.ModuleRegistry()

  def test_list_all_modules(self):
    modules = self.module_registry.list_all_modules()

    self.assertIn("timm/resnest50d", modules)
    self.assertIn("transformers/google/gemma-3-270m", modules)

  def test_torchvision_list_modules(self):
    # torchvision doesn't return a stable set of models when running in Forge so
    # we can't assert on existence of a specific model name.
    self.assertNotEmpty(self.module_registry.list_modules("torchvision"))

  def test_timm_list_modules(self):
    modules = self.module_registry.list_modules("timm")

    self.assertIn("resnest50d", modules)

  def test_transformers_list_modules(self):
    modules = self.module_registry.list_modules("transformers")

    self.assertIn("google/gemma-3-270m", modules)

  def test_torchvision_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "torchvision", "convnext_small"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_timm_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "timm", "mobilenetv3_small_050"
    )
    model = module_spec.module_factory()
    args, _ = module_spec.sample_inputs_factory()
    expected_output_shape = (args[0].shape[0], 1000)
    model.eval()

    out = model(*args)

    self.assertEqual(out.shape, expected_output_shape)

  def test_transformers_get_module_spec(self):
    module_spec = self.module_registry.get_module_spec(
        "transformers", "google/gemma-3-270m"
    )
    model = module_spec.module_factory()
    _, kwargs = module_spec.sample_inputs_factory()
    expected_logits_shape = (
        *kwargs["input_ids"].shape,
        module_spec.config.vocab_size,
    )
    model.eval()

    out = model(**kwargs)

    self.assertEqual(out.logits.shape, expected_logits_shape)


if __name__ == "__main__":
  absltest.main()
