// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/multinomial/multinomial_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/multinomial/multinomial.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

at::Tensor AtenMultinomial(const at::Tensor& self, int64_t num_samples,
                           bool replacement,
                           c10::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kMultinomial, param_keys,
      (self, num_samples, replacement, generator), {
        TT_CHECK_THROW(!generator.has_value(), error::kUnimplemented)
            << "generator is not yet supported.";
        TT_CHECK_THROW(self.dim() >= 1 && self.dim() <= 2,
                       error::kInvalidArgument)
            << "input must be 1D or 2D";
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "input must be a floating point tensor";
        TT_CHECK_THROW(num_samples >= 0, error::kInvalidArgument)
            << "num_samples must be non-negative";

        if (!replacement) {
          TT_CHECK_THROW(num_samples <= self.size(-1), error::kInvalidArgument)
              << "cannot sample n_sample > prob_dist.size(-1) samples "
                 "without "
                 "replacement";
        }

        Dimensions output_dims;
        if (self.dim() == 1) {
          output_dims.push_back(num_samples);
        } else {
          output_dims.push_back(self.size(0));
          output_dims.push_back(num_samples);
        }

        auto op_builder =
            [num_samples,
             replacement](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildMultinomialShlo(input, num_samples, replacement);
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRefArray<1> result_buf,
            DispatchOp<1>(OpName::kMultinomial, std::move(op_builder), self,
                          {.out_dtype = mlir::ElementType::I64,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = OpSplitMode::kSplitAfter}));
        return MakeTensor(std::move(result_buf));
      });
}

at::Tensor& AtenMultinomialOut(const at::Tensor& self, int64_t num_samples,
                               bool replacement,
                               c10::optional<at::Generator> generator,
                               at::Tensor& out) {
  TT_KERNEL(
      OpName::kMultinomialOut, param_keys,
      (self, num_samples, replacement, generator, out), {
        TT_CHECK_THROW(!generator.has_value(), error::kUnimplemented)
            << "generator is not yet supported.";
        TT_CHECK_THROW(self.dim() >= 1 && self.dim() <= 2,
                       error::kInvalidArgument)
            << "input must be 1D or 2D";
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "input must be a floating point tensor";
        TT_CHECK_THROW(num_samples >= 0, error::kInvalidArgument)
            << "num_samples must be non-negative";

        if (!replacement) {
          TT_CHECK_THROW(num_samples <= self.size(-1), error::kInvalidArgument)
              << "cannot sample n_sample > prob_dist.size(-1) samples "
                 "without "
                 "replacement";
        }

        Dimensions output_dims;
        if (self.dim() == 1) {
          output_dims.push_back(num_samples);
        } else {
          output_dims.push_back(self.size(0));
          output_dims.push_back(num_samples);
        }

        auto op_builder =
            [num_samples,
             replacement](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildMultinomialShlo(input, num_samples, replacement);
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRefArray<1> result_buf,
            DispatchOp<1>(OpName::kMultinomial, std::move(op_builder), self,
                          {.out_dtype = mlir::ElementType::I64,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = OpSplitMode::kSplitAfter}));
        at::native::resize_output(out, result_buf.dimensions());
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
