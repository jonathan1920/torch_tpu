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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/_internal/compile/compiled_mode.h"
#include "torch_tpu/_internal/dynamism/dynamism_ops.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/hlo/translate/register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

// Enum class for supported MLIR printing configurations.
enum class MlirPrintConfig {
  kMlirPretty,
  kMlirDebugInfo,
  kMlirSerialized,
  kMlirVersionedSerialized,
};

// Converts between torch_tpu.export.MlirPrintConfig and the C++ enum class.
MlirPrintConfig PyToMlirPrintConfig(const std::string& print_config) {
  if (print_config == "MlirPretty") {
    return MlirPrintConfig::kMlirPretty;
  } else if (print_config == "MlirDebugInfo") {
    return MlirPrintConfig::kMlirDebugInfo;
  } else if (print_config == "MlirSerialized") {
    return MlirPrintConfig::kMlirSerialized;
  } else if (print_config == "MlirVersionedSerialized") {
    return MlirPrintConfig::kMlirVersionedSerialized;
  }
  TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                    << "Unknown MLIR print config.");
  llvm_unreachable("throws");
}

// Serializes MLIR module to pass back to python.
// Supports `MlirPrintConfig` options.
std::string PrintMlirModule(mlir::ModuleOp module,
                            MlirPrintConfig print_config) {
  switch (print_config) {
    case MlirPrintConfig::kMlirPretty:
      return DebugString(module, DebugStringOptions::kDisableDebugInfo);
    case MlirPrintConfig::kMlirDebugInfo:
      return DebugString(module, DebugStringOptions::kEnableDebugInfo);
    case MlirPrintConfig::kMlirSerialized: {
      std::string mlir_str;
      llvm::raw_string_ostream os(mlir_str);
      if (mlir::failed(mlir::writeBytecodeToFile(module, os))) {
        TT_THROW_IF_ERROR(TT_ERROR(error::kInternal)
                          << "Failed to serialize MLIR module to bytecode.");
      }
      return mlir_str;
    }
    case MlirPrintConfig::kMlirVersionedSerialized: {
      TT_ASSIGN_OR_THROW(
          auto mlir_str,
          xla::SerializeUsingVersionedStablehlo(
              module, mlir::vhlo::Version::fromCompatibilityRequirement(
                          mlir::vhlo::Version::CompatibilityRequirement::WEEK_4)
                          .toString()));
      return mlir_str;
    }
  }
  llvm::report_fatal_error(
      "[PrintMlirModule] Unreachable, unknown MlirPrintConfig");
}

at::Tensor PyMakePlaceholder(const std::vector<int64_t>& sizes,  // INT_VEC_OK
                             at::ScalarType dtype, bool requires_grad) {
  TT_ASSIGN_OR_THROW(at::Tensor prepared_tensor,
                     MakePlaceholder(sizes, dtype, requires_grad),
                     _.SetPrepend() << "failed to create placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakePlaceholderLike(const at::Tensor& arg_tensor) {
  const std::vector<int64_t> sizes(arg_tensor.sizes().begin(),  // INT_VEC_OK
                                   arg_tensor.sizes().end());
  return PyMakePlaceholder(sizes, arg_tensor.scalar_type(),
                           arg_tensor.requires_grad());
}

// Returns MLIR module corresponding to the graph terminating at result_tensors
// and taking argument_tensors as inputs.
//
// Supports `MlirPrintConfig` options.
py::bytes PyExtractMlirModule(const std::vector<at::Tensor>& result_tensors,
                              const std::vector<at::Tensor>& argument_tensors,
                              const std::string& print_config) {  // INT_VEC_OK
  TT_ASSIGN_OR_THROW(ContextedModule module,
                     ContextedModule::Make([&](mlir::MLIRContext& context) {
                       return ExtractMlirFromGraph(context, argument_tensors,
                                                   result_tensors);
                     }));

  mlir::BaseScopedDiagnosticHandler diag_handler(&module.context());
  if (failed(mlir::verify(module.get()))) {
    TT_THROW_IF_ERROR(
        TT_ERROR(error::kInternal)
        << "MLIR module is invalid:\n"
        << diag_handler.ConsumeStatus().message() << "\n"
        << DebugString(module.get(), DebugStringOptions::kEnableDebugInfo));
  }
  std::string mlir_str =
      PrintMlirModule(module.get(), PyToMlirPrintConfig(print_config));
  return py::bytes(mlir_str.data(), mlir_str.size());
}

// Parses a string containing MLIR, verifies it, and serializes it to bytecode.
// This function will throw a RuntimeError in Python if:
//  1. The input mlir_text is not syntactically valid MLIR.
//  2. The MLIR is semantically invalid (e.g., type mismatches) and fails
//     verification.
//  3. An internal error occurs during the final bytecode serialization.
py::bytes PySerializeMlirTextModule(const std::string& mlir_text) {
  TT_ASSIGN_OR_THROW(
      ContextedModule module,
      ContextedModule::Make(
          [&](mlir::MLIRContext& context)
              -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
            return mlir::parseSourceString<mlir::ModuleOp>(
                llvm::StringRef(mlir_text.data(), mlir_text.size()),
                mlir::ParserConfig{&context});
          }));
  if (!module.get()) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                      << "Failed to parse MLIR module text.");
  }

  if (failed(mlir::verify(module.get()))) {
    TT_THROW_IF_ERROR(
        TT_ERROR(error::kInternal)
        << "MLIR module is invalid.\n"
        << DebugString(module.get(), DebugStringOptions::kEnableDebugInfo));
  }

  std::string bytecode_str;
  llvm::raw_string_ostream os(bytecode_str);
  if (mlir::failed(mlir::writeBytecodeToFile(module.get(), os))) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInternal)
                      << "Failed to serialize MLIR module to bytecode.");
  }

  return py::bytes(bytecode_str);
}

py::str PyPrintMlirBytecode(const py::bytes& bytecode) {
  const auto bytecode_view = py::cast<std::string_view>(bytecode);
  TT_ASSIGN_OR_THROW(
      ContextedModule module,
      ContextedModule::Make(
          [&](mlir::MLIRContext& context)
              -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
            return mlir::parseSourceString<mlir::ModuleOp>(
                llvm::StringRef(bytecode_view.data(), bytecode_view.size()),
                mlir::ParserConfig{&context});
          }));
  if (!module.get()) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                      << "Failed to parse MLIR bytecode.");
  }

  std::string mlir_str;
  llvm::raw_string_ostream os(mlir_str);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false, /*prettyForm=*/false);
  module.get()->print(os, flags);

  return py::str(mlir_str);
}

// Compiles an MLIR module.
// Args:
//   mlir_module: The serialized MLIR module.
//   eager: If true, use the compiler profile optimized for eager execution;
//          otherwise, use the profile optimized for torch.compile.
// Returns:
//   The compiled executable.
SharedLoadedExecutable PyCompileMlir(py::bytes& mlir_module,
                                     const bool fast_compile) {
  const auto module_bytecode = py::cast<std::string_view>(mlir_module);
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  // Provide the current python context to the compilation function so that
  // it can generate readable Mlir module names.
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());
  TT_ASSIGN_OR_THROW(
      auto executable,
      CompileMlirExecutable(module_bytecode,
                            fast_compile ? CompilationMode::kFastCompile
                                         : CompilationMode::kFastRuntime));
  return executable;
}

// Executes a compiled PJRT executable with the given arguments.
//
// Args:
//   executable: The loaded PJRT executable to run.
//   argument_tensors: A list of PyTorch tensors to pass as inputs.
//   output_shapes: Optional list of shapes for the output tensors.
//     If not empty, the number of elements in output_shapes must match the
//     number of output tensors. These shapes override the static shapes
//     inferred from the executable. This is useful for bounded dynamic programs
//     where the actual output shape might differ from the static
//     upper bound.
std::vector<at::Tensor> PyExecuteCompiledModel(
    const SharedLoadedExecutable& executable,
    const std::vector<at::Tensor>& argument_tensors,
    const std::vector<std::vector<int64_t>>& output_shapes) {  // INT_VEC_OK
  // Get the flattened output shapes from the executable.
  // Executables can return tuples, we want individual tensor shapes.
  TT_ASSIGN_OR_THROW(std::vector<xla::Shape> output_shapes_vec,
                     executable->GetOutputShapes());
  std::vector<const xla::Shape*> output_shapes_flat_vec;
  for (const auto& output_shape : output_shapes_vec) {
    xla::ShapeUtil::FlattenTupleShape(output_shape, output_shapes_flat_vec);
  }

  // Determine output shapes from the executable.
  std::vector<Shape> result_shapes_vec;
  result_shapes_vec.reserve(output_shapes_flat_vec.size());

  if (!output_shapes.empty()) {
    TT_CHECK_THROW(output_shapes.size() == output_shapes_flat_vec.size(),
                   error::kInvalidArgument)
        << "output shapes must be specified for all outputs or none, "
        << "got " << output_shapes.size() << " output shapes for "
        << output_shapes_flat_vec.size() << " output tensors";
  }

  for (size_t i = 0; i < output_shapes_flat_vec.size(); ++i) {
    const xla::Shape* output_shape = output_shapes_flat_vec[i];
    TT_ASSIGN_OR_THROW(Shape result_shape, MakeShape(*output_shape));
    if (!output_shapes.empty()) {
      const auto& shape = output_shapes[i];
      TT_CHECK_THROW(shape.size() == result_shape.dimensions().size(),
                     error::kInvalidArgument)
          << "output shape number of dimensions must match the statically "
             "inferred dimensions, got output shape dimensions "
          << shape.size() << " and inferred dimensions "
          << result_shape.dimensions().size() << " for output tensor " << i;

      for (size_t j = 0; j < shape.size(); ++j) {
        TT_CHECK_THROW(shape[j] <= result_shape.dimensions()[j],
                       error::kInvalidArgument)
            << "output shape dimension must not exceed the statically "
               "inferred bound, got output shape "
            << ToString(shape) << " and inferred shape "
            << ToString(result_shape.dimensions());
      }

      result_shape.dimensions().assign(shape.begin(), shape.end());
    }
    result_shapes_vec.push_back(result_shape);
  }
  return ExecuteCompiledModel(executable, argument_tensors, result_shapes_vec);
}

bool PyGetMlirTracebacksEnabled() {
  return GetTracebackMode() == TracebackMode::kEnabled;
}

void PySetMlirTracebacksEnabled(bool enabled) {
  SetTracebackMode(enabled ? TracebackMode::kEnabled
                           : TracebackMode::kDisabled);
}

}  // namespace

// Returns the MLIR module for a pad subgraph.
// Args:
//   tensor_info: A list of pairs, where each pair contains the shape of a
//     tensor and its scalar type.
//   bounds_list: A list of pairs, where each pair contains the dynamic
//     dimensions of a tensor and their upper bounds.
// Returns:
//   The MLIR module for the pad subgraph as bytes.
// Example:
//   tensor_info = [([1, 4], torch.int64)]
//   bounds_list = [([1], [8])]. // Dynamic dimension is 1, upper bound is 8.
//  MLIR module:
// module @pad_module {
//   func.func @main(%arg0: tensor<1x4xi64>) -> (tensor<1x8xi64>, tensor<i32>) {
//     %c = stablehlo.constant dense<0> : tensor<i64>
//     %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4],
//       interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
//     %1 = stablehlo.get_dimension_size %arg0, dim = 1 :
//        (tensor<1x4xi64>) -> tensor<i32>
//     return %0, %1 : tensor<1x8xi64>, tensor<i32>
//   }
// }

py::bytes PyGetPadModuleMlir(
    const std::vector<
        std::pair<std::vector<int64_t>, at::ScalarType>>&  // INT_VEC_OK
        tensor_info,
    const std::vector<
        std::pair<std::vector<int64_t>, std::vector<int64_t>>>&  // INT_VEC_OK
        bounds_list) {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  std::vector<Shape> shapes;
  shapes.reserve(tensor_info.size());
  for (int64_t i = 0; i < tensor_info.size(); ++i) {
    const auto& info = tensor_info[i];
    Shape shape;
    shape.dimensions().assign(info.first.begin(), info.first.end());
    TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                       internal::ToElementType(info.second));
    shape.set_dtype(element_type);

    if (i < bounds_list.size()) {
      const auto& dims = bounds_list[i].first;
      const auto& bounds = bounds_list[i].second;
      if (!bounds.empty() && !dims.empty()) {
        TT_CHECK_THROW(dims.size() == bounds.size(), error::kInvalidArgument)
            << "dimension indices and upper bounds must have the same size, "
            << "got " << dims.size() << " and " << bounds.size();
        for (size_t j = 0; j < dims.size(); ++j) {
          const int64_t dim_index = dims[j];
          const int64_t dim_upper_bound = bounds[j];
          TT_CHECK_THROW(
              dim_index >= 0 && dim_index < shape.dimensions().size(),
              error::kInvalidArgument)
              << "dimension index must be within bounds [0, "
              << shape.dimensions().size() - 1 << "], got " << dim_index
              << " for input tensor " << i << " with shape "
              << ToString(shape.dimensions());

          const int64_t dim_size = shape.dimensions()[dim_index];
          TT_CHECK_THROW(dim_upper_bound >= dim_size, error::kInvalidArgument)
              << "upper bound must be greater than or equal to the static "
                 "shape's dimension size, got upper bound "
              << dim_upper_bound << " for dimension " << dim_index
              << " for input tensor " << i << " with shape "
              << ToString(shape.dimensions());

          shape.dynamic_dimensions().push_back(
              {dim_index, dim_size, dim_upper_bound});
        }
      }
    }
    shapes.push_back(shape);
  }

  TT_ASSIGN_OR_THROW(mlir::OwningOpRef<mlir::ModuleOp> module,
                     GetPadModule(context, shapes));

  std::string mlir_str;
  llvm::raw_string_ostream os(mlir_str);
  if (mlir::failed(mlir::writeBytecodeToFile(module.get(), os))) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInternal)
                      << "failed to serialize MLIR module to bytecode");
  }
  return py::bytes(mlir_str);
}

py::bytes PySerializeExecutable(const SharedLoadedExecutable& executable) {
  TT_ASSIGN_OR_THROW(const std::string serialized,
                     executable->GetExecutable()->SerializeExecutable(),
                     _.SetPrepend() << "Failed to serialize executable: ");
  return py::bytes(serialized);
}

SharedLoadedExecutable PyLoadSerializedExecutable(py::bytes& serialized_bytes) {
  const auto data = py::cast<std::string_view>(serialized_bytes);
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_CHECK_THROW(client != nullptr, error::kFailedPrecondition)
      << "PjRtClient must be initialized before loading a serialized "
         "executable.";
  TT_ASSIGN_OR_THROW(SharedLoadedExecutable executable,
                     client->LoadSerializedExecutable(
                         data, /*options=*/std::nullopt, xla::LoadOptions()),
                     _.SetPrepend()
                         << "Failed to load serialized executable: ");
  return executable;
}

PYBIND11_MODULE(tpu_torch_compile, m) {
  py::class_<xla::PjRtLoadedExecutable,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<xla::PjRtLoadedExecutable>>(
      m, "PjRtLoadedExecutable");
  m.def("execute", PyExecuteCompiledModel,
        // Type: PjRtLoadedExecutable
        py::arg("executable"), py::arg("argument_tensors"),
        py::arg("output_shapes") =
            std::vector<std::vector<int64_t>>(),  // INT_VEC_OK
        "Executes a compiled PJRT executable with the given arguments.\n\n"
        "Args:\n"
        "  executable: The loaded PJRT executable to run.\n"
        "  argument_tensors: A list of PyTorch tensors to pass as inputs.\n"
        "  output_shapes: Optional list of shapes for the output tensors.\n"
        "    If provided, the number of elements must match the number of "
        "output tensors. These shapes override the static shapes inferred\n"
        "    from the executable.\n"
        "    This is useful for bounded dynamic programs where the actual\n"
        "    output shape might differ from the static upper bound.");
  m.def("placeholder", PyMakePlaceholder, py::arg("sizes"), py::arg("dtype"),
        py::arg("requires_grad"));
  m.def("placeholder_like", PyMakePlaceholderLike, py::arg("arg_tensor"));
  m.def("build_mlir", PyExtractMlirModule, py::arg("result_tensors"),
        py::arg("argument_tensors"),
        py::arg("print_config") = "MlirPretty");  // INT_VEC_OK
  // Returns: PjRtLoadedExecutable
  m.def("compile_mlir", PyCompileMlir, py::arg("mlir_module_bytecode"),
        py::arg("fast_compile") = false);
  m.def("serialize_mlir_text", PySerializeMlirTextModule, py::arg("mlir_text"),
        "Parses a StableHLO MLIR text string and returns its serialized "
        "bytecode.");
  m.def("print_mlir_bytecode", PyPrintMlirBytecode, py::arg("bytecode"),
        "Prints an MLIR bytecode as human-readable string without location "
        "information.");
  m.def("get_pad_module_mlir", PyGetPadModuleMlir, py::arg("tensor_info"),
        py::arg("bounds_list"),
        "Returns the MLIR module for a pad subgraph as bytecode.\n\n"
        "Args:\n"
        "  tensor_info: A list of (shape, dtype) pairs for each tensor.\n"
        "  bounds_list: A list of (dynamic_dimensions, upper_bounds) pairs.");
  m.def("get_mlir_tracebacks_enabled", &PyGetMlirTracebacksEnabled,
        "Return whether MLIR location tracebacks are currently enabled.");
  m.def(
      "set_mlir_tracebacks_enabled", &PySetMlirTracebacksEnabled,
      py::arg("enabled"),
      "Sets whether MLIR location tracebacks should be captured with each op.");
  m.def("serialize_executable", PySerializeExecutable, py::arg("executable"),
        "Serializes a PjRtLoadedExecutable to bytes for caching.");
  m.def("load_serialized_executable", PyLoadSerializedExecutable,
        py::arg("serialized_bytes"),
        "Loads a PjRtLoadedExecutable from serialized bytes.");
}

}  // namespace torch_tpu
