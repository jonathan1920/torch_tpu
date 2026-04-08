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

#include "torch_tpu/eager/device_gen_impl.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/full.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/GeneratorImpl.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/CallOnce.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> UpdateRngState(at::Tensor rng_state,
                                               uint64_t value,
                                               int64_t position) {
  ABSL_VLOG(3) << "[UpdateRngState] rng_state: "
               << ", value: " << value << ", position: " << position;
  auto set_seed_builder =
      [value, position](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    auto& builder = input.getBuilder();
    auto& op_builder = builder.getOpBuilder();
    mlir::MlirOp value_scalar =
        MakeScalarConstant(builder, value,
                           op_builder.getIntegerType(/*width=*/64,
                                                     /*isSigned=*/false));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp value_1d,
                        Unsqueeze(value_scalar, /*dim=*/0));
    mlir::MlirOp start_indices =
        MakeScalarConstant(builder, position, op_builder.getI64Type());
    return mlir::stablehlo::DynamicUpdateSlice(input, value_1d, start_indices);
  };

  TT_ASSIGN_OR_RETURN(auto params,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(value, position));

  return DispatchOp<1>(OpName::kRngSetSeed, std::move(set_seed_builder),
                       {rng_state},
                       {.out_dtype = mlir::ElementType::UI64,
                        .out_dims = {2},
                        .op_param_cache_keys = std::move(params)});
}

absl::StatusOr<DeviceBufferRef> UpdateRngSeed(at::Tensor rng_state,
                                              uint64_t value) {
  return UpdateRngState(rng_state, value, /*position=*/0);
}

absl::StatusOr<DeviceBufferRef> UpdateRngOffset(at::Tensor rng_state,
                                                uint64_t value) {
  return UpdateRngState(rng_state, value, /*position=*/1);
}

// A singleton that holds one generator per device. The generators are lazily
// initialized.
class DeviceGenerators {
 public:
  // This class is neither copyable nor movable.
  DeviceGenerators(const DeviceGenerators&) = delete;
  DeviceGenerators& operator=(const DeviceGenerators&) = delete;
  DeviceGenerators(DeviceGenerators&&) = delete;
  DeviceGenerators& operator=(DeviceGenerators&&) = delete;

  static DeviceGenerators& GetDefaultInstance();

  // Returns the default generator for the given device index. If idx is -1,
  // returns the default generator for the current device.
  const at::Generator& GetDefaultGenerator(c10::DeviceIndex idx = -1);

  // Creates a new generator for the given device index. If idx is -1, creates a
  // new generator for the current device.
  at::Generator CreateGenerator(c10::DeviceIndex idx = -1) const;

  int64_t num_devices() const { return num_devices_; }

 private:
  DeviceGenerators();
  std::vector<at::Generator> generators_;
  std::deque<c10::once_flag> generator_init_flags_;
  int64_t num_devices_;
};

absl::Status CheckRngState(const at::Tensor& rng_state) {
  TT_RET_CHECK(rng_state.dtype() == at::kUInt64, error::kFailedPrecondition)
      << "expected rng_state to be of dtype UInt64, got " << rng_state.dtype();
  TT_RET_CHECK(rng_state.dim() == 1, error::kFailedPrecondition)
      << "expected rng_state to be a 1D tensor, got " << rng_state.dim();
  TT_RET_CHECK(rng_state.size(0) == 2, error::kFailedPrecondition)
      << "expected rng_state to be a vector to be of size 2, got "
      << rng_state.size(0);
  return absl::OkStatus();
}

}  // namespace

DeviceGenerators& DeviceGenerators::GetDefaultInstance() {
  // We cannot use absl::NoDestructor here because the constructor is private.
  static auto* const kInstance = new DeviceGenerators();
  return *kInstance;
}

DeviceGenerators::DeviceGenerators() {
  const auto* guard =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
  TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
      << "TPU device guard is not registered.";
  num_devices_ = static_cast<int32_t>(guard->deviceCount());
  generators_.resize(num_devices_);
  generator_init_flags_.resize(num_devices_);
}

const at::Generator& DeviceGenerators::GetDefaultGenerator(
    at::DeviceIndex idx) {
  ABSL_VLOG(1) << "[DeviceGenerators::GetDefaultGenerator] idx: "
               << static_cast<int32_t>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  } else {
    TT_CHECK_THROW(idx >= 0 && idx < num_devices_, error::kFailedPrecondition)
        << "The device_index is invalid, expected an index between 0 and"
        << num_devices_ - 1 << " got " << static_cast<int32_t>(idx);
  }
  c10::call_once(generator_init_flags_[idx], [&] {
    generators_[idx] = at::make_generator<DeviceGeneratorImpl>(idx);
    generators_[idx].seed();
  });
  return generators_[idx];
}

at::Generator DeviceGenerators::CreateGenerator(c10::DeviceIndex idx) const {
  ABSL_VLOG(1) << "[CreateGenerator] idx: " << static_cast<int32_t>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  }
  TT_CHECK_THROW(idx >= 0 && idx < num_devices_, error::kFailedPrecondition)
      << "The device_index is invalid, expected an index between 0 and"
      << num_devices_ - 1 << " got " << static_cast<int32_t>(idx);
  return at::make_generator<DeviceGeneratorImpl>(idx);
}

c10::DeviceType DeviceGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {
  rng_state_ =
      at::full(/*size=*/{2}, /*fill_value=*/0.0, /*dtype_opt=*/at::kUInt64,
               /*layout_opt=*/std::nullopt, /*device_opt=*/this->device(),
               /*pin_memory_opt=*/std::nullopt);
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index,
                                         at::Tensor rng_state)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      rng_state_(rng_state) {}

void DeviceGeneratorImpl::set_current_seed(uint64_t seed) {
  // set_current_seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetSeed, _, (IgnoreInCacheKey(seed, "Legacy usage")), {
    TT_ASSIGN_OR_THROW(auto rng_state_buffer, UpdateRngSeed(rng_state_, seed));
    auto new_rng_state = MakeTensor(std::move(rng_state_buffer));
    rng_state_ = new_rng_state;
  });
}

void DeviceGeneratorImpl::set_offset(uint64_t offset) {
  // set_offset() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetOffset, _,
            (IgnoreInCacheKey(offset, "Legacy usage")), {
              TT_ASSIGN_OR_THROW(auto rng_state_buffer,
                                 UpdateRngOffset(rng_state_, offset));
              auto new_rng_state = MakeTensor(std::move(rng_state_buffer));
              rng_state_ = new_rng_state;
            });
}

uint64_t DeviceGeneratorImpl::get_offset() const {
  return rng_state_[1].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::current_seed() const {
  return rng_state_[0].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::seed() {
  // seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSeed, _, (), {
    auto random = c10::detail::getNonDeterministicRandom(false);
    this->set_current_seed(random);
    return random;
  });
}

c10::intrusive_ptr<c10::TensorImpl> DeviceGeneratorImpl::get_state() const {
  ABSL_VLOG(1) << "[get_state]";
  return rng_state_.getIntrusivePtr();
}

void DeviceGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  ABSL_VLOG(1) << "[DeviceGeneratorImpl::set_state]" << new_state.dtype();
  TT_CHECK_THROW(new_state.device() == this->device(),
                 error::kFailedPrecondition)
      << "The device of the new state is not the same as the device of the "
         "generator.";
  TT_CHECK_THROW(new_state.sizes() == c10::IntArrayRef({2}),
                 error::kFailedPrecondition)
      << "The new state must be shape (2,)";
  TT_CHECK_THROW(rng_state_.dtype() == at::kUInt64, error::kFailedPrecondition)
      << "The dtype of the new state is not UInt64.";
  TT_ASSIGN_OR_THROW(DeviceBufferRef new_state_buffer,
                     GetBufferFromAtTensor(new_state));

  auto new_rng_state = MakeTensor(std::move(new_state_buffer));
  rng_state_ = new_rng_state;
}

void DeviceGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& state) {
  TT_CHECK_THROW(false, error::kUnimplemented)
      << "graphsafe_set_state is not supported in this Generator";
}

c10::intrusive_ptr<c10::GeneratorImpl>
DeviceGeneratorImpl::graphsafe_get_state() const {
  TT_CHECK_THROW(false, error::kUnimplemented)
      << "graphsafe_get_state is not supported in this Generator";
  return nullptr;
}

DeviceGeneratorImpl* DeviceGeneratorImpl::clone_impl() const {
  return new DeviceGeneratorImpl(device().index(), rng_state_.clone());
}

const at::Generator& GetDefaultDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
}

absl::StatusOr<at::Tensor> GetRngState(c10::optional<at::Generator> generator) {
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());
  auto rng_state = at::Tensor(gen->get_state());
  TT_RETURN_IF_ERROR(CheckRngState(rng_state));
  return rng_state;
}

absl::StatusOr<at::Tensor> GetAndAdvanceRngState(
    c10::optional<at::Generator> generator, int64_t num_elements,
    int64_t bit_width) {
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());
  auto rng_input_state = at::Tensor(gen->get_state());
  TT_RETURN_IF_ERROR(CheckRngState(rng_input_state));
  if (num_elements <= 0 || bit_width <= 0) {
    return rng_input_state;
  }

  // Snapshot the original buffer so that we can return it after updating the
  // state.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef original_buf,
                      GetBufferFromAtTensor(rng_input_state));

  // Dispatch the state update.
  TT_ASSIGN_OR_RETURN(auto state_param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(num_elements, bit_width));
  auto state_op_builder = [num_elements,
                           bit_width](mlir::MlirOp rng_input_state) {
    return BuildRngStateUpdateShlo(rng_input_state, num_elements, bit_width);
  };
  TT_ASSIGN_OR_RETURN(
      auto rng_output_state_buf,
      (DispatchOp<1>(OpName::kRngStateUpdate, std::move(state_op_builder),
                     {rng_input_state},
                     {.out_dtype = mlir::ElementType::UI64,
                      .out_dims = {2},
                      .op_param_cache_keys = std::move(state_param_keys)})));

  // Give back the updated state to the generator.
  auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
  gen->set_state(*rng_output_state.unsafeGetTensorImpl());

  // Return a tensor pointing to the original buffer.
  return MakeTensor(std::move(original_buf));
}

absl::Status UpdateRngState(c10::optional<at::Generator> generator,
                            at::Tensor& rng_state) {
  TT_RETURN_IF_ERROR(CheckRngState(rng_state));
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());
  gen->set_state(*rng_state.unsafeGetTensorImpl());
  return absl::OkStatus();
}

at::Generator MakeDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().CreateGenerator(idx);
}

void SetManualSeed(uint64_t seed, c10::DeviceIndex idx) {
  auto gen = DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
  gen.set_current_seed(seed);
  gen.set_offset(0);
}

void SetManualSeedAll(uint64_t seed) {
  const int64_t num_devices =
      DeviceGenerators::GetDefaultInstance().num_devices();
  for (int i = 0; i < num_devices; ++i) {
    SetManualSeed(seed, i);
  }
}

}  // namespace torch_tpu
