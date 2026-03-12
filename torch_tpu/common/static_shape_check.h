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

#ifndef TORCH_TPU_COMMON_STATIC_SHAPE_CHECK_H_
#define TORCH_TPU_COMMON_STATIC_SHAPE_CHECK_H_

#include <string_view>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Checks that the given `type` only has static dimensions.
//
// Overload for `mlir::RankedTensorType`. It is called when the op's StableHLO
// builder function is called.
//
// `arg_name` should be set to the corresponding argument name of the op. This
// will be used for displaying a better error message.
absl::Status CheckStaticShape(mlir::RankedTensorType type,
                              std::string_view arg_name);

// Checks that the given `tensor` only has static dimensions.
//
// Overload for `at::Tensor`. It is called inside the function that PyTorch
// dispatches the op to.
//
// `arg_name` should be set to the corresponding argument name of the op. This
// will be used for displaying a better error message.
absl::Status CheckStaticShape(const at::Tensor& tensor,
                              std::string_view arg_name);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_STATIC_SHAPE_CHECK_H_
