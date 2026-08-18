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

"""Unit tests for production multi-host benchmark runner."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import discovery as discovery_lib
from examples.benchmarks.e2e.harness import measure as measure_lib
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import multihost_runner
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import steps
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from examples.benchmarks.e2e.harness.models import llama
from torch_tpu._internal.distributed import multiprocessing


def setUpModule():
  discovery_lib.import_submodules(steps)


class MultihostRunnerTest(parameterized.TestCase):

  @mock.patch.dict(registry_lib.REGISTRY, {}, clear=True)
  def test_resolve_benchmark_targets_deterministic(self):
    spec1 = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    spec1.name = "llama_8b_inference"
    registry_lib.REGISTRY["llama_8b_inference"] = spec1

    # No filter returns all modes
    targets = multihost_runner.resolve_benchmark_targets(test_filter=None)
    self.assertLen(targets, len(common.RunMode))
    self.assertEqual(targets[0][0].name, "llama_8b_inference")

    # Exact target_name match
    targets = multihost_runner.resolve_benchmark_targets(
        test_filter=(
            "llama_8b_inference_eager_default"
            " llama_8b_inference_eager_optimized"
        )
    )
    self.assertLen(targets, 2)
    self.assertEqual(targets[0][1], common.RunMode.EAGER_DEFAULT)
    self.assertEqual(targets[1][1], common.RunMode.EAGER_OPTIMIZED)

    # Spec name without mode suffix throws ValueError
    with self.assertRaises(ValueError):
      multihost_runner.resolve_benchmark_targets(
          test_filter="llama_8b_inference"
      )

    # Invalid target throws ValueError
    with self.assertRaises(ValueError):
      multihost_runner.resolve_benchmark_targets(
          test_filter="invalid_target_xyz"
      )

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(measure_lib, "measure")
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_run_single_target_task(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    del mock_is_initialized, mock_device_ops
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.factory = mock.MagicMock(
        return_value=(mock.MagicMock(), (), {}, None)
    )
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None

    registry_lib.REGISTRY["dummy"] = fake_spec

    # Setup fake target so it doesn't fail on device/dtype resolution
    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.CPU
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    multihost_runner._run_single_target_task(
        spec_name="dummy",
        mode_value="eager_default",
        platform_name="cpu",
    )

    mock_init_group.assert_called_once_with(backend="tpu_dist")
    self.assertTrue(mock_barrier.called)
    mock_measure.assert_called_once()
    mock_compiler_reset.assert_called_once()
    mock_destroy_group.assert_called_once()

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(
      measure_lib, "measure", side_effect=RuntimeError("Model OOM")
  )
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_run_single_target_task_exception_safety(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    """Verifies cleanup runs even when benchmark raises an exception."""
    del mock_is_initialized, mock_init_group, mock_measure, mock_device_ops
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.factory = mock.MagicMock(
        return_value=(mock.MagicMock(), (), {}, None)
    )
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None

    registry_lib.REGISTRY["dummy"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.CPU
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    with self.assertRaisesRegex(RuntimeError, "Model OOM"):
      multihost_runner._run_single_target_task(
          spec_name="dummy",
          mode_value="eager_default",
          platform_name="cpu",
      )

    mock_compiler_reset.assert_called_once()
    mock_destroy_group.assert_called_once()
    self.assertEqual(mock_barrier.call_count, 1)

  @mock.patch.dict(os.environ, {"BENCHMARK_PLATFORM": "v7_2x2x2"})
  @mock.patch.object(multihost_runner.distributed_utils, "dist_run")
  @mock.patch.object(multihost_runner, "resolve_benchmark_targets")
  def test_main_successful_dispatch(self, mock_resolve_targets, mock_dist_run):
    dummy_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    dummy_spec.name = "dummy"
    mock_resolve_targets.return_value = [
        (dummy_spec, common.RunMode.EAGER_DEFAULT)
    ]

    with mock.patch.dict(
        os.environ, {"TESTBRIDGE_TEST_ONLY": "dummy_eager_default"}
    ):
      multihost_runner.main(["multihost_runner"])

    mock_dist_run.assert_called_once_with(
        8,  # V7_2X2X2 has 8 procs per node
        multihost_runner._run_single_target_task,
        "dummy",
        "eager_default",
        "v7_2x2x2",
    )

  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_cleanup_multihost_worker_state_skip_barrier_on_exception(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
  ):
    """Verifies barrier is skipped when unwinding from benchmark exception."""
    del mock_compiler_reset, mock_is_initialized
    multihost_runner._cleanup_multihost_worker_state(
        rank=0, barrier_policy=multihost_runner.BarrierPolicy.SKIP
    )

    mock_barrier.assert_not_called()
    mock_destroy_group.assert_called_once()

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(measure_lib, "measure")
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  @mock.patch.object(mode_lib, "run_mode_context")
  def test_run_single_target_uses_run_mode_context_and_default_run_scope(
      self,
      mock_run_mode_context,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    """Verifies run_mode_context wraps execution and default RUN_SCOPE is 'full'."""
    del (
        mock_is_initialized,
        mock_device_ops,
        mock_init_group,
        mock_barrier,
        mock_measure,
        mock_compiler_reset,
        mock_destroy_group,
    )
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy"
    fake_spec.dtype = target_lib.DType.BF16
    captured_ctx = None

    def fake_factory(ctx):
      nonlocal captured_ctx
      captured_ctx = ctx
      return (mock.MagicMock(), (), {}, None)

    fake_spec.factory = fake_factory
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None

    registry_lib.REGISTRY["dummy"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.CPU
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    with mock.patch.dict(os.environ, {}, clear=True):
      multihost_runner._run_single_target_task(
          spec_name="dummy",
          mode_value="eager_optimized",
          platform_name="cpu",
      )

    # 1. Verify run_mode_context was invoked with
    # (RunMode.EAGER_OPTIMIZED, fake_target)
    mock_run_mode_context.assert_called_once_with(
        common.RunMode.EAGER_OPTIMIZED, fake_target
    )
    # 2. Verify default RUN_SCOPE passed to Context is FULL
    self.assertIsNotNone(captured_ctx)
    self.assertEqual(captured_ctx.run_scope, context_lib.RunScope.FULL)

  def test_runner_flags_registered(self):
    """Verifies that runner flags (e.g.

    mlcompass_tracking_id, base_cl) are registered in absl.flags.
    """
    from absl import flags  # pylint: disable=g-import-not-at-top

    self.assertIn("mlcompass_tracking_id", flags.FLAGS)
    self.assertIn("base_cl", flags.FLAGS)

  @mock.patch.object(llama, "_load_meta_llama")
  def test_meta_llama_8b_forward_supports_single_and_multihost(
      self, mock_loader
  ):
    """Verifies meta_llama_8b_forward supports both single-host and multi-host."""

    fake_model = mock.MagicMock()
    fake_inputs = (mock.MagicMock(), 0)
    mock_loader.return_value = (fake_model, fake_inputs)

    for platform in [
        target_lib.Platform.V7_2X2X1,
        target_lib.Platform.V7_2X2X2,
    ]:
      fake_target = mock.MagicMock(spec=target_lib.Target)
      fake_target.platform = platform
      fake_ctx = mock.MagicMock()
      fake_ctx.target = fake_target
      fake_ctx.dtype = target_lib.DType.BF16
      fake_ctx.device_kind = target_lib.DeviceKind.CPU
      res = llama.meta_llama_8b_forward(fake_ctx)
      self.assertIsNotNone(res[0])
      self.assertIsNotNone(res[1])


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
