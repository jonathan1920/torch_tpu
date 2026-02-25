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

"""Tests inference and training with torch.compile across Pytorch Image models."""

import copy
import random

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu import api
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import utils
from tests import module_registry


def _get_logits(outputs):
  return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
  """Performs a single training step (Forward, Backward, Optimizer Step).

  Args:
    model: The model to train.
    optimizer: The optimizer to use.
    inputs: The input tensor.
    target: The target tensor.

  Returns:
    The calculated loss for the training step.
  """
  optimizer.zero_grad()

  outputs = model(inputs)
  logits = _get_logits(outputs)

  if target.dtype == torch.long:
    # For classification tasks (integer labels)
    loss = torch.nn.functional.cross_entropy(logits, target)
  else:
    # For feature extraction tasks (float targets)
    loss = torch.nn.functional.mse_loss(logits, target)

  loss.backward()
  optimizer.step()
  return loss


class ModelCoverageTimmTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.tpu_device = api.tpu_device()
    cls.module_registry = module_registry.ModuleRegistry()

  def setUp(self):
    super().setUp()
    random.seed(42)
    torch.manual_seed(42)

  def _create_model_and_inputs(
      self, provider: str, module_name: str, *, is_training=False
  ):
    module_spec = self.module_registry.get_module_spec(
        provider, module_name, load_weights=False
    )
    cpu_model = module_spec.module_factory()
    if is_training:
      cpu_model.train()
    else:
      cpu_model.eval()

    tpu_model = copy.deepcopy(cpu_model).to(self.tpu_device)

    args, _ = module_spec.sample_inputs_factory()
    cpu_inputs = args[0]
    tpu_inputs = cpu_inputs.to(self.tpu_device)

    cpu_target = None
    tpu_target = None

    # For training, generate targets based on the model's task
    if is_training:
      num_classes = getattr(cpu_model, "num_classes", 0)

      if num_classes > 0:
        # For classification tasks, generate integer labels
        cpu_target = torch.randint(0, num_classes, (cpu_inputs.shape[0],))
      else:
        # For feature extraction tasks, generate random float targets
        with torch.no_grad():
          dummy_output = _get_logits(cpu_model(cpu_inputs))
          out_shape = dummy_output.shape
        cpu_target = torch.randn(out_shape)

      tpu_target = cpu_target.to(self.tpu_device)

    compiled_tpu_model = torch.compile(
        tpu_model,
        dynamic=False,
        backend=torch_tpu_compile.TpuBackend(),
    )

    return (
        cpu_model,
        tpu_model,
        compiled_tpu_model,
        cpu_inputs,
        tpu_inputs,
        cpu_target,
        tpu_target,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="timm/convnext_small",
          provider="timm",
          module_name="convnext_small",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/resnet50d",
          provider="timm",
          module_name="resnet50d",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/vit_small_patch8_224",
          provider="timm",
          module_name="vit_small_patch8_224",
          rtol=1e-3,
          atol=1e-3,
      ),
  )
  def test_timm_training(
      self,
      provider: str,
      module_name: str,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    (
        cpu_model,
        tpu_model,
        compiled_tpu_model,
        cpu_inputs,
        tpu_inputs,
        cpu_target,
        tpu_target,
    ) = self._create_model_and_inputs(provider, module_name, is_training=True)

    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
    tpu_optimizer = torch.optim.SGD(tpu_model.parameters(), lr=0.01)

    tpu_loss = _train_step(
        compiled_tpu_model, tpu_optimizer, tpu_inputs, tpu_target
    )
    cpu_loss = _train_step(cpu_model, cpu_optimizer, cpu_inputs, cpu_target)

    utils.assert_close(
        actual=tpu_loss,
        expected=cpu_loss,
        rtol=rtol,
        atol=atol,
        preamble="Timm Model TPU vs CPU Training Loss",
        check_value=utils.CheckValueMode.LOOSE,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="timm/convnext_small",
          provider="timm",
          module_name="convnext_small",
          rtol=1e-2,
          atol=1e-2,
      ),
      dict(
          testcase_name="timm/resnet50d",
          provider="timm",
          module_name="resnet50d",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/vit_small_patch8_224",
          provider="timm",
          module_name="vit_small_patch8_224",
          rtol=1e-3,
          atol=3e-2,
      ),
  )
  def test_timm_inference(
      self,
      provider: str,
      module_name: str,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    cpu_model, _, compiled_tpu_model, cpu_inputs, tpu_inputs, _, _ = (
        self._create_model_and_inputs(provider, module_name, is_training=False)
    )

    cpu_out = cpu_model(cpu_inputs)
    tpu_out = compiled_tpu_model(tpu_inputs)

    cpu_logits = _get_logits(cpu_out)
    tpu_logits = _get_logits(tpu_out)

    utils.assert_close(
        actual=tpu_logits,
        expected=cpu_logits,
        rtol=rtol,
        atol=atol,
        preamble=f"Timm Model {module_name} TPU vs CPU Inference Logits",
        check_value=utils.CheckValueMode.LOOSE,
    )


if __name__ == "__main__":
  absltest.main()
