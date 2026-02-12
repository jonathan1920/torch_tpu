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

#include "torch_tpu/ops/randperm/randperm_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {
absl::StatusOr<mlir::MlirOp> BuildRandpermStateUpdateShlo(mlir::MlirOp state,
                                                          int64_t n) {
  auto& builder = state.getBuilder();
  const auto state_type = GetTensorTypeOrDie(state);
  const int64_t state_size = state_type.getShape()[0];

  // key: output_state[0] = initial_state[0]
  mlir::MlirOp state_key = mlir::stablehlo::Slice(state, {0}, {1}, {1});
  // counter: output_state[1] = initial_state[1] + ceil(num_bits / 128)
  // Note that num_bits = n * 64, and therefore
  //   ceil(num_bits / 128) = ceil(n / 2) = (n + 1) / 2.
  mlir::MlirOp state_counter = mlir::stablehlo::Slice(state, {1}, {2}, {1});
  mlir::MlirOp increment =
      MakeConstant(builder, (n + 1) / 2, mlir::ElementType::UI64, {1});
  mlir::MlirOp new_counter = mlir::stablehlo::Add(state_counter, increment);

  mlir::SmallVector<mlir::MlirOp, 3> concat_inputs = {state_key, new_counter};
  // Keep any additional state (i.e. beyond the first two u64's) unmodified.
  if (state_size > 2) {
    concat_inputs.push_back(
        mlir::stablehlo::Slice(state, {2}, {state_size}, {1}));
  }
  return mlir::stablehlo::Concatenate(builder, concat_inputs, 0);
}

absl::StatusOr<mlir::MlirOp> BuildRandpermShlo(mlir::MlirOp state, int64_t n,
                                               mlir::ElementType output_dtype) {
  const auto rng_input_state_type = GetTensorTypeOrDie(state);
  auto& builder = state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto& ctx = builder.getContext();

  // 1. Iota: [0, 1, ..., n-1], use I64 to avoid overflow
  auto iota_type =
      makeTensorType(ctx, {n}, builder.getOpBuilder().getI64Type());
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  // 2. Prepare keys: Generate UI64 random numbers as keys for sorting.
  // UI64 is the only dtype StableHLO RngBitGenerator supports.
  auto key_type = makeTensorType(ctx, {n}, mlir::ElementType::UI64);

  // Call rng bit generator using Philox algorithm to generate UI64.
  // We ignore the updated state returned by this op because we compute it
  // independently in BuildRandpermStateUpdateShlo.
  auto rng_op = mlir::stablehlo::RngBitGeneratorOp::create(
      op_builder, state.getValue().getLoc(), rng_input_state_type, key_type,
      mlir::stablehlo::RngAlgorithmAttr::get(
          op_builder.getContext(), mlir::stablehlo::RngAlgorithm::PHILOX),
      state.getValue());

  mlir::MlirOp keys = mlir::MlirOp(builder, rng_op.getOutput());

  // 3. Sort the keys and values
  // Input: [keys, values] -> Output: [sorted_keys, sorted_values]
  auto comparator = [key_type, iota_type](mlir::RegionBuilder& rb) {
    auto op_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {key_type.getElementType(), iota_type.getElementType()},
        mlir::stablehlo::ComparisonDirection::LT,
        /*compareType=*/std::nullopt, &rb.getRegion(), &op_builder);
  };

  auto sorted_keys_and_values = mlir::stablehlo::Sort(
      builder, {keys, iota}, comparator, /*dim*/ 0, /*is_stable=*/false);

  auto result_op = sorted_keys_and_values[1];
  if (output_dtype != mlir::ElementType::I64) {
    auto result_tensor_type = makeTensorType(ctx, {n}, output_dtype);
    result_op =
        mlir::stablehlo::Convert(result_tensor_type, sorted_keys_and_values[1]);
  }

  return result_op;
}
}  // namespace

at::Tensor& AtenRandpermGeneratorOut(c10::SymInt n,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(OpName::kRandpermGeneratorOut, param_keys, (n, generator, out), {
    const int64_t n_int = n.guard_int(__FILE__, __LINE__);

    TT_ASSIGN_OR_THROW(auto rng_input_state, GetRngState(generator));
    TT_CHECK_THROW(rng_input_state.size(0) >= 2, error::kFailedPrecondition)
        << "Expected rng_input_state to have at least 2 elements, got "
        << rng_input_state.size(0);
    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    // 1. Dispatch independent state update.
    // Decoupling the state update from the actual randperm makes the state
    // update resilient to failures in randperm's execution, e.g. out-of-memory
    // due to 'n' being too large for the device.
    TT_ASSIGN_OR_THROW(auto state_param_keys,
                       TT_MAKE_OP_PARAM_CACHE_KEYS(n_int, "state_update"));
    auto state_op_builder = [n_int](mlir::MlirOp rng_input_state) {
      return BuildRandpermStateUpdateShlo(rng_input_state, n_int);
    };
    TT_ASSIGN_OR_THROW(
        auto rng_output_state_buf,
        (DispatchOp<1>(OpName::kRandpermGeneratorOut,
                       std::move(state_op_builder), {rng_input_state},
                       {.out_dtype = mlir::ElementType::UI64,
                        .out_dims = {2},
                        .op_param_cache_keys = std::move(state_param_keys)})));

    // Give back the updated state to the generator.
    auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
    TT_THROW_IF_ERROR(UpdateRngState(generator, rng_output_state));

    // 2. Dispatch the actual randperm.
    // Note that the input state is the *original* state, not the updated one.
    auto randperm_op_builder = [n_int,
                                output_dtype](mlir::MlirOp rng_input_state) {
      return BuildRandpermShlo(rng_input_state, n_int, output_dtype);
    };

    TT_ASSIGN_OR_THROW(
        auto output_buf,
        (DispatchOp<1>(OpName::kRandpermGeneratorOut,
                       std::move(randperm_op_builder), {rng_input_state},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(out.sizes()),
                        .op_param_cache_keys = std::move(param_keys)})));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
