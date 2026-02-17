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

#ifndef TORCH_TPU_OPS_FOREACH_FOREACH_ADD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FOREACH_FOREACH_ADD_ATEN_KERNELS_H_

#include <vector>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha);

std::vector<at::Tensor> AtenForeachAddScalar(at::TensorList self,
                                             const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachAddScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachAddTensor(at::TensorList self,
                                             const at::Tensor& other,
                                             const at::Scalar& alpha);

void AtenForeachAdd_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha);

void AtenForeachAdd_Scalar(at::TensorList self, const at::Scalar& scalar);

void AtenForeachAdd_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);
void AtenForeachAdd_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FOREACH_FOREACH_ADD_ATEN_KERNELS_H_
