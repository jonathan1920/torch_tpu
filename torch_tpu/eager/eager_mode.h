/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_EAGER_EAGER_MODE_H_
#define TORCH_TPU_EAGER_EAGER_MODE_H_

namespace torch_tpu {

// The op defer mode.
enum class EagerMode {
  // kDeferAndFuseWithO1 defers all ops except those that cannot be deferred.
  // This is used
  // in eager mode (normal operation, not torch.compile).
  kDeferAndFuseWithO1,
  // Similar to kDeferAndFuseWithO1, but uses more aggressive XLA optimizations.
  kDeferAndFuse,
  // kDeferNever marks all ops to be executed immediately, similarly to PyTorch
  // on CUDA eager mode. This is primarily used for debugging, as it has
  // suboptimal compile and execution performance.
  kDeferNever,
  // kDeferNeverAndLaunchBlocking marks all ops to be executed immediately and
  // waits for the compilation of one op before dispatching the next one.
  // Some expensive runtime checks are also performed in this mode. This
  // should be used only for debugging, as it has the worst execution
  // performance.
  kDeferNeverAndLaunchBlocking,
  // kInternalDeferAll attempts to defer all ops. If an op cannot be deferred,
  // it will
  // raise a runtime exception. This should be used only in torch.compile mode.
  kInternalDeferAll,
};

// Sets the defer mode for the current thread. Thread-safe.
void SetEagerMode(EagerMode mode);

// Returns the defer mode for the current thread. Thread-safe.
[[nodiscard]] EagerMode GetEagerMode();

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_OP_DISPATCHER_H_
