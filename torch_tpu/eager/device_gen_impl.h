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

#ifndef TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_
#define TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/tensor.h"  // IWYU pragma: keep
#include "c10/core/Device.h"
#include "c10/core/GeneratorImpl.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

// Holds the device-resident state of the generator (seed and offset) as a
// tensor. This is the TPU equivalent of PyTorch's CUDAGeneratorState,
// encapsulating the current generation state of the device.
struct DeviceGeneratorState : public c10::intrusive_ptr_target {
  DeviceGeneratorState() = default;
  explicit DeviceGeneratorState(at::Tensor device_state_tensor)
      : device_state_tensor(std::move(device_state_tensor)) {}

  c10::intrusive_ptr<DeviceGeneratorState> clone() const {
    return c10::make_intrusive<DeviceGeneratorState>(
        device_state_tensor.clone());
  }

  // 1D uint64 tensor of 2 elements: {seed, offset}
  at::Tensor device_state_tensor;  // UNINITIALIZED_TENSOR_OK
};

// A wrapper on a tensor of two integers, representing the current state (seed
// and offset) of the generator. Given these, we can deterministically generate
// random bits. Kernels that require random bits, will use this tensor as an
// extra argument, and will output the new state of the generator.
class DeviceGeneratorImpl : public c10::GeneratorImpl {
 public:
  DeviceGeneratorImpl(c10::DeviceIndex device_index = -1);
  DeviceGeneratorImpl(c10::DeviceIndex device_index, at::Tensor rng_state);
  DeviceGeneratorImpl(c10::DeviceIndex device_index,
                      c10::intrusive_ptr<DeviceGeneratorState> state);

  // This class is neither copyable nor movable.
  DeviceGeneratorImpl(const DeviceGeneratorImpl&) = delete;
  DeviceGeneratorImpl& operator=(const DeviceGeneratorImpl&) = delete;
  DeviceGeneratorImpl(DeviceGeneratorImpl&&) = delete;
  DeviceGeneratorImpl& operator=(DeviceGeneratorImpl&&) = delete;

  ~DeviceGeneratorImpl() override = default;

  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

  static c10::DeviceType device_type();

  at::Tensor DeviceStateTensor() const;
  absl::Status SetDeviceStateTensor(at::Tensor device_state_tensor);
  // Validates the shape, dtype, and device of the RNG state tensor.
  absl::Status CheckDeviceStateTensor(const at::Tensor& rng_state) const;

  // Returns the current device-resident rng_state tensor. This also
  // advances the generator's state by the amount specified
  // (num_elements * bit_width).
  absl::StatusOr<at::Tensor> GetAndAdvanceDeviceStateTensor(
      int64_t num_elements, int64_t bit_width = 64);

 private:
  DeviceGeneratorImpl* clone_impl() const override;

  // 1D uint64 tensor of 2 elements: {seed, offset}
  c10::intrusive_ptr<DeviceGeneratorState> state_;
};

// Convenience functions.

// Returns the default generator for the given device index. If idx is -1,
// returns the default generator for the current device.
at::Generator& GetDefaultDeviceGenerator(c10::DeviceIndex idx = -1);

// Creates a new generator for the given device index. If idx is -1, creates a
// new generator for the current device.
at::Generator MakeDeviceGenerator(c10::DeviceIndex idx = -1);

// Sets the seed for the generator of the given device index. If idx is -1,
// sets the seed for the generator of the current device.
void SetManualSeed(uint64_t seed, c10::DeviceIndex idx = -1);

// Sets the seed for all TPU generators.
void SetManualSeedAll(uint64_t seed);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_
