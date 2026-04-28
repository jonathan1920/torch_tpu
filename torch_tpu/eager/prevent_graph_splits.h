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

#ifndef TORCH_TPU_EAGER_PREVENT_GRAPH_SPLITS_H_
#define TORCH_TPU_EAGER_PREVENT_GRAPH_SPLITS_H_

#include "absl/status/statusor.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// TODO(b/498334411): make this logic safe in distributed code
// Returns true if the traversal should be prevented from splitting.
//
// This is disabled by default; if enabled (by setting
// --torch_tpu_internal_prevent_graph_splits > 0), then frequent Traversal
// patterns will be compiled in the background, and prevented from splitting
// once the fused compilation is finished.
absl::StatusOr<bool> PreventGraphSplit(const Traversal& traversal);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_PREVENT_GRAPH_SPLITS_H_
