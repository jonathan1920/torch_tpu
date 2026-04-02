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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace py = pybind11;
namespace {

void PyRegisterCustomKernel(c10::string_view name, c10::string_view kernel_key,
                            c10::string_view mlir_module_string) {
  if (RegisterCustomKernel(name, kernel_key, mlir_module_string)) {
    ABSL_VLOG(1) << "Registered new custom kernel: name=" << name
                 << ", kwargs=" << kernel_key;
  } else {
    ABSL_VLOG(2) << "Custom kernel already registered: name=" << name
                 << ", kwargs=" << kernel_key;
  }
}

bool PyLookupCustomKernel(c10::string_view name, c10::string_view kernel_key) {
  return LookupCustomKernel(name, kernel_key);
}

std::vector<at::Tensor> PyCallCustomKernel(
    const c10::string_view name, const c10::string_view kernel_key,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& output_shapes,
    // Cannot use absl::flat_hash_map here because it is not supported by
    // pybind11.
    // NOLINTNEXTLINE(google3-runtime-unneeded-pointer-stability-check)
    const std::unordered_map<int64_t, int64_t>& input_output_aliases) {
  TT_KERNEL(
      OpName::kCustomKernel, op_param_cache_keys,
      (name, kernel_key, inputs, output_shapes, input_output_aliases), {
        Indices aliased_input_indices;
        absl::flat_hash_map<int64_t, int64_t> output_to_input_alias_map;
        for (const auto& [input_index, output_index] : input_output_aliases) {
          aliased_input_indices.push_back(input_index);
          output_to_input_alias_map[output_index] = input_index;
        }

        auto custom_op_builder = [name = std::string(name),
                                  kernel_key = std::string(kernel_key)](
                                     absl::Span<const mlir::MlirOp> inputs,
                                     mlir::MlirBuilder& builder) {
          return CallCustomKernel(builder, inputs, name, kernel_key);
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

        DispatchOpOptions<kDynamicSize> options{
            .out_dtypes = output_dtypes,
            .out_dims_list = output_dims_list,
            .computation_dtype = std::nullopt,
            .op_param_cache_keys = std::move(op_param_cache_keys)};
        if (!input_output_aliases.empty()) {
          // If there are any aliased input/output pairs, we need to materialize
          // both before and after this deferred op to ensure the donation
          // behavior proceeds correctly.
          options.split_mode = OpSplitMode::kSplitBoth;
          // Note that this DeferredOp needs its inputs to be marked with
          // jax.buffer_donor if they are leaf inputs to the MLIR module.
          options.aliased_input_indices = std::move(aliased_input_indices);
        }

        absl::StatusOr<std::vector<DeviceBufferRef>> results_status =
            DispatchOp<kDynamicSize, kDynamicSize>(OpName::kCustomKernel,
                                                   std::move(custom_op_builder),
                                                   inputs, std::move(options));
        TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> results,
                           results_status);

        std::vector<at::Tensor> result_tensors;
        result_tensors.reserve(results.size());
        for (auto i = 0; i < results.size(); ++i) {
          if (output_to_input_alias_map.contains(i)) {
            // Assign the computed output to the input, and then copy the input
            // tensor to use as the output tensor as well.
            const auto input_index = output_to_input_alias_map[i];
            TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(results[i]),
                                                     inputs[input_index]));
            result_tensors.push_back(inputs[input_index]);  // intentional copy
          } else {
            // Non-aliased outputs are made into new tensors.
            result_tensors.push_back(MakeTensor(std::move(results[i])));
          }
        }
        return result_tensors;
      });
}

}  // namespace

PYBIND11_MODULE(tpu_torch_pallas, m) {
  m.def("register_custom_kernel", PyRegisterCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"),
        py::kw_only(),  // Everything after this is keyword-only
        py::arg("serialized_mlir_module"));
  m.def("lookup_custom_kernel", PyLookupCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"));
  m.def("call_custom_kernel", PyCallCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"),
        py::kw_only(),  // Everything after this is keyword-only
        py::arg("inputs"), py::arg("output_shapes"),
        py::arg("input_output_aliases")
        // Cannot use absl::flat_hash_map here because it is not supported
        // by pybind11.
        // NOLINTNEXTLINE(google3-runtime-unneeded-pointer-stability-check)
        = std::unordered_map<int64_t, int64_t>());
}

}  // namespace torch_tpu
