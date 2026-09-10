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

#ifndef TORCH_TPU_CSRC_EAGER_MATERIALIZATION_HEURISTICS_H_
#define TORCH_TPU_CSRC_EAGER_MATERIALIZATION_HEURISTICS_H_

#include "csrc/eager/device_buffer.h"

namespace torch_tpu {

// Returns true if the deferred op is a 2D spatial convolution that
// significantly benefits from FastRuntime (O2) compilation over default
// FastCompile (O1).
//
// Performance regimes:
// - O1 wins on: small batches (batch <= 1), vector-aligned widths (W % 128 ==
// 0),
//   channels_last (NHWC) layouts, and low compute volume (< 100M MACs). In
//   these cases, compact binaries fit in on-chip IMEM without overlay paging,
//   avoiding O2's fixed ~2.0 ms execution latency floor.
// - O2 wins on: heavy unaligned NCHW convolutions (batch > 1, W % 128 != 0,
//   MACs >= 100M). In these cases, O2 tiles unaligned rows to eliminate sublane
//   padding serialization, achieving 35x-106x speedups that easily amortize
//   the ~2.0 ms overlay paging cost.
[[nodiscard]] bool IsFastRuntimeConvolutionCandidate(const DeferredOp& op);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_MATERIALIZATION_HEURISTICS_H_
