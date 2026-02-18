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

#ifndef TORCH_TPU_OPS_FOREACH_FOREACH_MUL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FOREACH_FOREACH_MUL_ATEN_KERNELS_H_

#include <vector>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

std::vector<at::Tensor> AtenForeachMulList(at::TensorList self,
                                           at::TensorList other);

std::vector<at::Tensor> AtenForeachMulScalar(at::TensorList self,
                                             const at::Scalar& scalar);

std::vector<at::Tensor> AtenForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachMulTensor(at::TensorList self,
                                             const at::Tensor& other);
void AtenForeachMul_List(at::TensorList self, at::TensorList other);
void AtenForeachMul_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachMul_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);
void AtenForeachMul_Tensor(at::TensorList self, const at::Tensor& other);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FOREACH_FOREACH_MUL_ATEN_KERNELS_H_
