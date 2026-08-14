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

"""Simple toy torch.nn.Module test running a single Linear layer training step on TPU with xprof tracing."""

import copy
from absl.testing import absltest
import torch
from examples.benchmarks import single_trace_trainer


class ToyLinearModule(torch.nn.Module):
  """Simple toy module with a single linear layer."""

  def __init__(self, in_features=128, out_features=64):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.linear(x)


class SingleTraceTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("tpu")

  def _get_test_models(
      self, in_features=128, out_features=64, dtype=torch.float32
  ):
    model = ToyLinearModule(in_features, out_features)
    model.train()
    model_eager = copy.deepcopy(model)
    model_eager.train()
    return model.to(device=self.device, dtype=dtype), model_eager.to(
        device=self.device, dtype=dtype
    )

  def _assert_params_and_state_close(
      self,
      trainer,
      model_eager,
      optimizer_eager,
  ):
    for safe_name, p_comp in trainer.param_group.params.items():
      name = trainer._key_map[safe_name]
      p_eager = dict(model_eager.named_parameters())[name]

      torch.testing.assert_close(
          p_comp,
          p_eager,
          rtol=1e-4,
          atol=1e-4,
      )

      if isinstance(optimizer_eager, torch.optim.AdamW):
        if safe_name in trainer.param_group.opt_state_m:
          m_comp = trainer.param_group.opt_state_m[safe_name]
          m_eager = optimizer_eager.state[p_eager]["exp_avg"]
          torch.testing.assert_close(
              m_comp,
              m_eager,
              rtol=1e-4,
              atol=1e-4,
          )

        if safe_name in trainer.param_group.opt_state_v:
          v_comp = trainer.param_group.opt_state_v[safe_name]
          v_eager = optimizer_eager.state[p_eager]["exp_avg_sq"]
          torch.testing.assert_close(
              v_comp,
              v_eager,
              rtol=1e-4,
              atol=1e-4,
          )

      elif isinstance(optimizer_eager, torch.optim.SGD):
        if safe_name in trainer.param_group.opt_state_m:
          m_comp = trainer.param_group.opt_state_m[safe_name]
          if "momentum_buffer" in optimizer_eager.state[p_eager]:
            m_eager = optimizer_eager.state[p_eager]["momentum_buffer"]
            torch.testing.assert_close(
                m_comp,
                m_eager,
                rtol=1e-4,
                atol=1e-4,
            )

  def _verify_train_step_vs_eager(
      self, trainer, model_eager, x, target, ref_opt_cls, **ref_opt_kwargs
  ):
    optimizer_eager = ref_opt_cls(model_eager.parameters(), **ref_opt_kwargs)
    train_step = trainer.make_compiled_train_step(x, target)

    # ===== Step 1 =====
    loss = train_step(x, target)
    optimizer_eager.zero_grad()
    out_eager = model_eager(x)
    loss_eager = torch.nn.functional.mse_loss(out_eager, target)
    loss_eager.backward()
    optimizer_eager.step()

    torch.accelerator.synchronize()
    torch.testing.assert_close(loss, loss_eager, rtol=1e-4, atol=1e-4)

    self._assert_params_and_state_close(trainer, model_eager, optimizer_eager)

    # ===== Step 2 =====
    loss = train_step(x, target)
    optimizer_eager.zero_grad()
    out_eager = model_eager(x)
    loss_eager = torch.nn.functional.mse_loss(out_eager, target)
    loss_eager.backward()
    optimizer_eager.step()

    torch.accelerator.synchronize()
    torch.testing.assert_close(loss, loss_eager, rtol=1e-4, atol=1e-4)

    self._assert_params_and_state_close(
        trainer,
        model_eager,
        optimizer_eager,
    )

  def test_linear_layer_training_compiled_reference_adamw(self):
    model, model_eager = self._get_test_models()

    trainer = single_trace_trainer.SingleTraceTrainer(
        model,
        single_trace_trainer.ReferenceAdamw(use_bfloat16_moments=False),
    )

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.AdamW,
    )

  def test_linear_layer_training_compiled_fused_adamw(self):
    model, model_eager = self._get_test_models()

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    trainer = single_trace_trainer.SingleTraceTrainer(
        model,
        single_trace_trainer.FusedAdamw(use_bfloat16_moments=False),
    )

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.AdamW,
    )

  def test_linear_layer_training_compiled_reference_sgd(self):
    model, model_eager = self._get_test_models()
    trainer = single_trace_trainer.SingleTraceTrainer(
        model, single_trace_trainer.ReferenceSgd(lr=1e-2, momentum=0.9)
    )
    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.SGD,
        lr=1e-2,
        momentum=0.9,
    )

  def test_linear_layer_training_compiled_fused_sgd(self):
    model, model_eager = self._get_test_models()
    trainer = single_trace_trainer.SingleTraceTrainer(
        model, single_trace_trainer.FusedSgd(lr=1e-2, momentum=0.9)
    )
    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.SGD,
        lr=1e-2,
        momentum=0.9,
    )


if __name__ == "__main__":
  absltest.main()
