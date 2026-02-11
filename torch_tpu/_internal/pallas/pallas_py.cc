/*
 * Copyright 2025 Google LLC
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace py = pybind11;
namespace {

void PyRegisterCustomKernel(c10::string_view name, c10::string_view kwargs_str,
                            c10::string_view mlir_module_string) {
  if (RegisterCustomKernel(name, kwargs_str, mlir_module_string)) {
    ABSL_VLOG(1) << "Registered new custom kernel: name=" << name
                 << ", kwargs=" << kwargs_str;
  } else {
    ABSL_VLOG(2) << "Custom kernel already registered: name=" << name
                 << ", kwargs=" << kwargs_str;
  }
}

bool PyLookupCustomKernel(c10::string_view name, c10::string_view kwargs_str) {
  return LookupCustomKernel(name, kwargs_str);
}

std::vector<at::Tensor> PyCallCustomKernel(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& output_shapes, c10::string_view name,
    c10::string_view kwargs_str) {
  TT_KERNEL(OpName::kCustomKernel, _, (name, kwargs_str), {
    TT_ASSIGN_OR_THROW(OpParamCacheKeys op_param_cache_keys,
                       *OpParamCacheKeys::SetParam("custom_kernel_name", name)
                            .SetParam("custom_kernel_kwargs", kwargs_str));
    auto custom_op_builder =
        [name = std::string(name), kwargs_str = std::string(kwargs_str)](
            absl::Span<const mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
          return CallCustomKernel(builder, inputs, name, kwargs_str);
        };

    std::vector<mlir::ElementType> output_dtypes;
    std::vector<absl::Span<const int64_t>> output_dims_list;
    output_dtypes.reserve(output_shapes.size());
    output_dims_list.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
      TT_ASSIGN_OR_THROW(
          const auto output_dtype,
          ConvertTo<mlir::ElementType>(output_shape.scalar_type()));
      output_dtypes.push_back(output_dtype);
      output_dims_list.push_back(output_shape.sizes());
    }

    absl::StatusOr<std::vector<DeviceBufferRef>> results_status =
        DispatchOp<kDynamicSize, kDynamicSize>(
            OpName::kCustomKernel, std::move(custom_op_builder), inputs,
            {.out_dtypes = output_dtypes,
             .out_dims_list = output_dims_list,
             .computation_dtype = std::nullopt,
             .op_param_cache_keys = std::move(op_param_cache_keys)});
    TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> results, results_status);

    std::vector<at::Tensor> result_tensors;
    result_tensors.reserve(results.size());
    for (const auto& result : results) {
      result_tensors.push_back(MakeTensor(result));
    }
    return result_tensors;
  });
}
}  // namespace

PYBIND11_MODULE(tpu_torch_pallas, m) {
  m.def("register_custom_kernel", PyRegisterCustomKernel, py::arg("name"),
        py::arg("kwargs_str"), py::arg("mlir_module_string"));
  m.def("lookup_custom_kernel", PyLookupCustomKernel, py::arg("name"),
        py::arg("kwargs_str"));
  m.def("call_custom_kernel", PyCallCustomKernel, py::arg("inputs"),
        py::arg("output_shapes"), py::arg("name"), py::arg("kwargs_str"));
}

}  // namespace torch_tpu
