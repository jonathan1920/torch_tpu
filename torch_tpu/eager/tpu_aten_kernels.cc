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

#include "torch_tpu/eager/tpu_aten_kernels.h"

#include <cstdint>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/stack.h"
#include "ATen/native/CPUFallback.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/native/Resize.h"
#include "ATen/native/transformers/attention.h"
#include "ATen/ops/div.h"
#include "ATen/ops/mean.h"
#include "ATen/ops/mul.h"
#include "ATen/ops/pow.h"
#include "ATen/ops/sub.h"
#include "ATen/ops/sum.h"
#include "c10/core/TensorImpl.h"
#include "torch/library.h"
#include "torch_tpu/ops/fmax/fmax_aten_kernels.h"
#include "xla/xla_data.pb.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/a_min_max/a_min_max_aten_kernels.h"
#include "torch_tpu/ops/addcdiv/addcdiv_aten_kernels.h"
#include "torch_tpu/ops/addcmul/addcmul_aten_kernels.h"
#include "torch_tpu/ops/addmm/addmm_aten_kernels.h"
#include "torch_tpu/ops/addmv/addmv_aten_kernels.h"
#include "torch_tpu/ops/all_any/all_any_aten_kernels.h"
#include "torch_tpu/ops/arange/arange_aten_kernels.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/binary_aten_kernels.h"  // IWYU pragma: keep for AtenMulTensor, etc
#include "torch_tpu/ops/bincount/bincount_aten_kernels.h"
#include "torch_tpu/ops/bmm/bmm_aten_kernels.h"
#include "torch_tpu/ops/cat/cat_aten_kernels.h"
#include "torch_tpu/ops/clamp/clamp_aten_kernels.h"
#include "torch_tpu/ops/col2im/col2im_aten_kernels.h"
#include "torch_tpu/ops/convolution/convolution_aten_kernels.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/cumprod/cumprod_aten_kernels.h"
#include "torch_tpu/ops/cumsum/cumsum_aten_kernels.h"
#include "torch_tpu/ops/distance/dist_aten_kernels.h"
#include "torch_tpu/ops/dot/dot_aten_kernels.h"
#include "torch_tpu/ops/dot/vdot_aten_kernels.h"
#include "torch_tpu/ops/dropout/dropout_aten_kernels.h"
#include "torch_tpu/ops/elu/elu_aten_kernels.h"
#include "torch_tpu/ops/embedding/embedding_aten_kernels.h"
#include "torch_tpu/ops/equal/equal_aten_kernels.h"
#include "torch_tpu/ops/experimental/ragged_dot_aten_kernels.h"
#include "torch_tpu/ops/exponential/exponential_aten_kernels.h"
#include "torch_tpu/ops/fft/fft_aten_kernels.h"
#include "torch_tpu/ops/fill/fill_aten_kernels.h"
#include "torch_tpu/ops/flip/flip_aten_kernels.h"
#include "torch_tpu/ops/foreach/foreach_add_aten_kernels.h"
#include "torch_tpu/ops/foreach/foreach_mul_aten_kernels.h"
#include "torch_tpu/ops/gather/gather_aten_kernels.h"
#include "torch_tpu/ops/gelu/gelu_aten_kernels.h"
#include "torch_tpu/ops/group_norm/group_norm_aten_kernels.h"
#include "torch_tpu/ops/hardtanh/hardtanh_aten_kernels.h"
#include "torch_tpu/ops/histc/histc_aten_kernels.h"
#include "torch_tpu/ops/index/index_aten_kernels.h"
#include "torch_tpu/ops/index_add/index_add_aten_kernels.h"
#include "torch_tpu/ops/index_copy/index_copy_aten_kernels.h"
#include "torch_tpu/ops/index_put/index_put_aten_kernels.h"
#include "torch_tpu/ops/index_select/index_select_aten_kernels.h"
#include "torch_tpu/ops/is/is_aten_kernels.h"
#include "torch_tpu/ops/isin/isin_aten_kernels.h"
#include "torch_tpu/ops/layer_norm/layer_norm_aten_kernels.h"
#include "torch_tpu/ops/leaky_relu/leaky_relu_aten_kernels.h"
#include "torch_tpu/ops/lerp/lerp_aten_kernels.h"
#include "torch_tpu/ops/linalg/linalg_kernels.h"
#include "torch_tpu/ops/linalg/lu/linalg_lu_kernels.h"
#include "torch_tpu/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"
#include "torch_tpu/ops/linalg/vector_norm/aten_vector_norm_kernels.h"
#include "torch_tpu/ops/logical/logical_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/masked_fill/masked_fill_aten_kernels.h"  // IWYU pragma: keep for AtenMaskedFill
#include "torch_tpu/ops/masked_scatter/masked_scatter_aten_kernels.h"
#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"
#include "torch_tpu/ops/min_max/min_max_aten_kernels.h"
#include "torch_tpu/ops/mm/mm_aten_kernels.h"
#include "torch_tpu/ops/multinomial/multinomial_aten_kernels.h"
#include "torch_tpu/ops/native_batch_norm/native_batch_norm_aten_kernels.h"
#include "torch_tpu/ops/nll_loss/nll_loss_aten_kernels.h"
#include "torch_tpu/ops/nonzero/nonzero_aten_kernels.h"
#include "torch_tpu/ops/normal/normal_aten_kernels.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/pooling/adaptive_avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/max_pool_aten_kernels.h"
#include "torch_tpu/ops/prod/prod_aten_kernels.h"
#include "torch_tpu/ops/random/random_aten_kernels.h"
#include "torch_tpu/ops/randperm/randperm_aten_kernels.h"
#include "torch_tpu/ops/reductions/mean_aten_kernels.h"
#include "torch_tpu/ops/reductions/sum_aten_kernels.h"
#include "torch_tpu/ops/reductions/var_aten_kernels.h"
#include "torch_tpu/ops/reflection_pad/reflection_pad_aten_kernels.h"
#include "torch_tpu/ops/replication_pad/replication_pad_aten_kernels.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/rms_norm/rms_norm_aten_kernels.h"
#include "torch_tpu/ops/roll/roll_aten_kernels.h"
#include "torch_tpu/ops/round/round_aten_kernels.h"
#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"
#include "torch_tpu/ops/scatter/scatter_aten_kernels.h"
#include "torch_tpu/ops/set/set_aten_kernels.h"
#include "torch_tpu/ops/sigmoid/sigmoid_aten_kernels.h"
#include "torch_tpu/ops/softmax/softmax_aten_kernels.h"
#include "torch_tpu/ops/sort/sort_aten_kernels.h"
#include "torch_tpu/ops/split_with_sizes_copy/split_with_sizes_copy_aten_kernels.h"
#include "torch_tpu/ops/take/take_aten_kernels.h"
#include "torch_tpu/ops/tanh/tanh_aten_kernels.h"
#include "torch_tpu/ops/threshold/threshold_aten_kernels.h"
#include "torch_tpu/ops/topk/topk_aten_kernels.h"
#include "torch_tpu/ops/triangular/triangular_aten_kernels.h"
#include "torch_tpu/ops/tril_indices/tril_indices_aten_kernels.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "torch_tpu/ops/unfold/unfold_aten_kernels.h"
#include "torch_tpu/ops/uniform/uniform_aten_kernels.h"
#include "torch_tpu/ops/upsample/upsample_aten_kernels.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"
#include "torch_tpu/ops/where/where_aten_kernels.h"

namespace torch_tpu {
namespace {

at::Tensor& tpu_aten_mse_loss_out(const at::Tensor& self,
                                  const at::Tensor& target, int64_t reduction,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kMseLossOut, _, (self, target, reduction, out), {
    TT_CHECK_THROW(self.scalar_type() == target.scalar_type(),
                   error::kInvalidArgument)
        << "input and target must have the same dtype";

    {
      // All compositional ATen calls must be within a guard to prevent
      // re-dispatching to the Autograd layer, which would cause infinite
      // recursion. This ensures we are calling the primitive PrivateUse1
      // kernels.
      at::AutoDispatchBelowAutograd guard;

      at::Tensor squared_error = at::pow(at::sub(self, target), 2);
      at::Tensor result;  // UNINITIALIZED_TENSOR_OK=initialized in the if-else
      if (reduction == at::Reduction::None) {
        result = squared_error;
      } else if (reduction == at::Reduction::Mean) {
        result = at::mean(squared_error);
      } else if (reduction == at::Reduction::Sum) {
        result = at::sum(squared_error);
      } else {
        TT_CHECK_THROW(false, error::kInvalidArgument)
            << "unrecognized reduction mode " << reduction;
      }

      // Resize the output tensor and copy the result into it.
      // This is the standard way to handle `out=` variants.
      at::native::resize_output(out, result.sizes());
      out.copy_(result);
    }

    ABSL_VLOG(1)
        << "[C++ KERNEL tpu_aten_mse_loss_out (Compositional)] out(final): "
        << ToString(out);
    return out;
  });
}

at::Tensor tpu_aten_mse_loss_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Tensor& target,
                                      int64_t reduction) {
  TT_KERNEL(
      OpName::kMseLossBackward, _, (grad_output, self, target, reduction), {
        at::Tensor grad_input;  // UNINITIALIZED_TENSOR_OK=initialized below
        {
          at::AutoDispatchBelowAutograd guard;
          at::Tensor grad_raw = at::mul(at::sub(self, target), 2);
          if (reduction == at::Reduction::None) {
            grad_input = at::mul(grad_raw, grad_output);
          } else {
            grad_input = at::mul(grad_raw, grad_output);
            if (reduction == at::Reduction::Mean) {
              grad_input = at::div(grad_input, self.numel());
            }
          }
        }
        ABSL_VLOG(1)
            << "[C++ KERNEL tpu_aten_mse_loss_backward (Compositional)] "
            << "Result grad_input: " << ToString(grad_input);
        return grad_input;
      });
}

// Registers the kernel function for the given op name in the given library.
template <typename KernelFn>
void Impl(torch::Library& m, const OpName op_name, KernelFn kernel_fn) {
  // ToString() returns a string_view, but impl() requires a const char*.
  // We need to convert the string_view to a std::string so that we can
  // get a NUL-terminated const char* from it. Note that string_view::data()
  // is not guaranteed to be NUL-terminated.
  m.impl(std::string(ToString(op_name)).c_str(), kernel_fn);
}

}  // namespace

// When the dispatch key set is {PrivateUse1} (i.e. for TPU tensors in the
// eager mode), pytorch will try this dispatch table first. If the op is not
// found here, pytorch will then try the (_, PrivateUse1, m) dispatch table
// defined later in this file.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // go/keep-sorted start
  Impl(m, OpName::kAcosOut, AtenAcosOut);
  Impl(m, OpName::kAcoshOut, AtenAcoshOut);
  Impl(m, OpName::kAdaptiveAvgPool2d, AtenAdaptiveAvgPool2d);
  Impl(m, OpName::kAdaptiveAvgPool2dBackward, AtenAdaptiveAvgPool2dBackward);
  Impl(m, OpName::kAdaptiveAvgPool2dOut, AtenAdaptiveAvgPool2dOut);
  Impl(m, OpName::kAdaptiveAvgPool3d, AtenAdaptiveAvgPool3d);
  Impl(m, OpName::kAdaptiveAvgPool3dBackward, AtenAdaptiveAvgPool3dBackward);
  Impl(m, OpName::kAdaptiveAvgPool3dBackwardGradInput,
       AtenAdaptiveAvgPool3dBackwardGradInput);
  Impl(m, OpName::kAdaptiveAvgPool3dOut, AtenAdaptiveAvgPool3dOut);
  Impl(m, OpName::kAddOut, AtenAddOut);
  Impl(m, OpName::kAddcdivOut, AtenAddcdivOut);
  Impl(m, OpName::kAddcmulOut, AtenAddcmulOut);
  Impl(m, OpName::kAddmvOut, AtenAddmvOut);
  Impl(m, OpName::kAminmaxOut, AtenAminmaxOut);
  Impl(m, OpName::kArangeStartOut, AtenArangeStartOut);
  Impl(m, OpName::kArgMaxOut, AtenArgmaxOut);
  Impl(m, OpName::kArgMinOut, AtenArgminOut);
  Impl(m, OpName::kAsinOut, AtenAsinOut);
  Impl(m, OpName::kAsinhOut, AtenAsinhOut);
  Impl(m, OpName::kAtanOut, AtenAtanOut);
  Impl(m, OpName::kAtanhOut, AtenAtanhOut);
  Impl(m, OpName::kAvgPool2dBackwardGradInput, AtenAvgPool2dBackwardGradInput);
  Impl(m, OpName::kAvgPool2dOut, AtenAvgPool2dOut);
  Impl(m, OpName::kAvgPool3dBackwardGradInput, AtenAvgPool3dBackwardGradInput);
  Impl(m, OpName::kAvgPool3dOut, AtenAvgPool3dOut);
  Impl(m, OpName::kBitwiseLeftShiftTensorOut, AtenBitwiseLeftShiftTensorOut);
  Impl(m, OpName::kBitwiseRightShiftTensorOut, AtenBitwiseRightShiftTensorOut);
  Impl(m, OpName::kCatOut, AtenCatOut);
  Impl(m, OpName::kCdistForward, AtenCdistForward);
  Impl(m, OpName::kCeilOut, AtenCeilOut);
  Impl(m, OpName::kClampMaxOut, AtenClampMaxOut);
  Impl(m, OpName::kClampMaxTensorOut, AtenClampMaxTensorOut);
  Impl(m, OpName::kClampMinOut, AtenClampMinOut);
  Impl(m, OpName::kClampMinTensorOut, AtenClampMinTensorOut);
  Impl(m, OpName::kClampOut, AtenClampOut);
  Impl(m, OpName::kClampTensorOut, AtenClampTensorOut);
  Impl(m, OpName::kCol2Im, AtenCol2Im);
  Impl(m, OpName::kCol2ImOut, AtenCol2ImOut);
  Impl(m, OpName::kComplexOut, AtenComplexOut);
  Impl(m, OpName::kConjPhysicalOut, AtenConjPhysicalOut);
  Impl(m, OpName::kConvolution, AtenConvolution);
  Impl(m, OpName::kConvolutionBackward, AtenConvolutionBackward);
  Impl(m, OpName::kConvolutionOut, AtenConvolutionOut);
  // per https://github.com/pytorch/xla/issues/2881, this function was added
  // to fix aten:cpufallback for pytorch_xla in 2021.
  // But it isn't registered for CPU and is not called from copy_ in Python.
  // TODO: Revisit if we can replace it with _copy_from. // NOLINT
  Impl(m, OpName::kCopyFromAndResize, AtenCopyFromAndResize);
  Impl(m, OpName::kCosOut, AtenCosOut);
  Impl(m, OpName::kCoshOut, AtenCoshOut);
  Impl(m, OpName::kCumprodOut, AtenCumprodOut);
  Impl(m, OpName::kCumsumOut, AtenCumsumOut);
  Impl(m, OpName::kDivOut, AtenDivOut);
  Impl(m, OpName::kDivOutMode, AtenDivOutMode);
  Impl(m, OpName::kEluBackwardGradInput, AtenEluBackwardGradInput);
  Impl(m, OpName::kEluOut, AtenEluOut);
  Impl(m, OpName::kEmbeddingDenseBackward, AtenEmbeddingDenseBackward);
  Impl(m, OpName::kEmbeddingRenorm_, AtenEmbeddingRenorm_);
  Impl(m, OpName::kEqual, AtenEqual);
  Impl(m, OpName::kErfInvOut, AtenErfInvOut);
  Impl(m, OpName::kExpOut, AtenExpOut);
  Impl(m, OpName::kExponential_, AtenExponential_);
  Impl(m, OpName::kFftR2c, AtenFftR2c);
  Impl(m, OpName::kFftR2cOut, AtenFftR2cOut);
  Impl(m, OpName::kFloorDivide, AtenFloorDivide);
  Impl(m, OpName::kFloorDivideOut, AtenFloorDivideOut);
  Impl(m, OpName::kFmaxOut, AtenFmaxOut);
  Impl(m, OpName::kFmodTensorOut, AtenFmodTensorOut);
  Impl(m, OpName::kForeachAddList, AtenForeachAddList);
  Impl(m, OpName::kForeachAddScalar, AtenForeachAddScalar);
  Impl(m, OpName::kForeachAddScalarList, AtenForeachAddScalarList);
  Impl(m, OpName::kForeachAddTensor, AtenForeachAddTensor);
  Impl(m, OpName::kForeachAdd_List, AtenForeachAdd_List);
  Impl(m, OpName::kForeachAdd_Scalar, AtenForeachAdd_Scalar);
  Impl(m, OpName::kForeachAdd_ScalarList, AtenForeachAdd_ScalarList);
  Impl(m, OpName::kForeachAdd_Tensor, AtenForeachAdd_Tensor);
  Impl(m, OpName::kForeachMulList, AtenForeachMulList);
  Impl(m, OpName::kForeachMulScalar, AtenForeachMulScalar);
  Impl(m, OpName::kForeachMulScalarList, AtenForeachMulScalarList);
  Impl(m, OpName::kForeachMulTensor, AtenForeachMulTensor);
  Impl(m, OpName::kForeachMul_List, AtenForeachMul_List);
  Impl(m, OpName::kForeachMul_Scalar, AtenForeachMul_Scalar);
  Impl(m, OpName::kForeachMul_ScalarList, AtenForeachMul_ScalarList);
  Impl(m, OpName::kForeachMul_Tensor, AtenForeachMul_Tensor);
  Impl(m, OpName::kFusedRmsNorm, AtenFusedRmsNorm);
  Impl(m, OpName::kFusedRmsNormBackward, AtenFusedRmsNormBackward);
  Impl(m, OpName::kGatherOut, AtenGatherOut);
  Impl(m, OpName::kGeluBackwardGradInput, AtenGeluBackwardGradInput);
  Impl(m, OpName::kGeluOut, AtenGeluOut);
  Impl(m, OpName::kHardtanh, AtenHardtanh);
  Impl(m, OpName::kHardtanhOut, AtenHardtanhOut);
  Impl(m, OpName::kHardtanh_, AtenHardtanh_);
  Impl(m, OpName::kHistc, AtenHistc);
  Impl(m, OpName::kHistcOut, AtenHistcOut);
  Impl(m, OpName::kIlshiftScalar, AtenIlshiftScalar);
  Impl(m, OpName::kIlshiftTensor, AtenIlshiftTensor);
  Impl(m, OpName::kIndexAddOut, TpuAtenIndexAddOut);
  Impl(m, OpName::kIndexCopyOut, AtenIndexCopyOut);
  Impl(m, OpName::kIndexPutImpl_, TpuAtenIndexPutImpl_);
  Impl(m, OpName::kIndexSelect, TpuAtenIndexSelect);
  Impl(m, OpName::kIndexTensorOut, AtenIndexTensorOut);
  Impl(m, OpName::kIrshiftScalar, AtenIrshiftScalar);
  Impl(m, OpName::kIrshiftTensor, AtenIrshiftTensor);
  Impl(m, OpName::kIsNan, AtenIsNan);
  Impl(m, OpName::kIsNegInfOut, AtenIsNegInfOut);
  Impl(m, OpName::kIsPosInfOut, AtenIsPosInfOut);
  Impl(m, OpName::kLayerNormBackward, AtenLayerNormBackward);
  Impl(m, OpName::kLeakyReluOut, AtenLeakyReluOut);
  Impl(m, OpName::kLerpScalarOut, AtenLerpScalarOut);
  Impl(m, OpName::kLerpTensorOut, AtenLerpTensorOut);
  Impl(m, OpName::kLgammaOut, AtenLgammaOut);
  Impl(m, OpName::kLinalgInvExOut, AtenLinalgInvExOut);
  Impl(m, OpName::kLinalgLuFactorExOut, AtenLinalgLuFactorExOut);
  Impl(m, OpName::kLinalgLuOut, AtenLinalgLuOut);
  Impl(m, OpName::kLinalgLuSolveOut, AtenLinalgLuSolveOut);
  Impl(m, OpName::kLinalgSolveExOut, AtenLinalgSolveExOut);
  Impl(m, OpName::kLinalgSolveTriangular, AtenLinalgSolveTriangular);
  Impl(m, OpName::kLinalgSolveTriangularOut, AtenLinalgSolveTriangularOut);
  Impl(m, OpName::kLinalgVectorNormOut, AtenLinalgVectorNormOut);
  Impl(m, OpName::kLocalScalarDense, AtenLocalScalarDense);
  Impl(m, OpName::kLog10Out, AtenLog10Out);
  Impl(m, OpName::kLog1pOut, AtenLog1pOut);
  Impl(m, OpName::kLog2Out, AtenLog2Out);
  Impl(m, OpName::kLogSoftmaxBackwardDataOut, AtenLogSoftmaxBackwardDataOut);
  Impl(m, OpName::kLogSoftmaxOut, AtenLogSoftmaxOut);
  Impl(m, OpName::kLshiftScalar, AtenLshiftScalar);
  Impl(m, OpName::kLshiftTensor, AtenLshiftTensor);
  Impl(m, OpName::kLuUnpackOut, AtenLuUnpackOut);
  Impl(m, OpName::kMaskedFill_Scalar, AtenMaskedFill_Scalar);
  Impl(m, OpName::kMaskedFill_Tensor, AtenMaskedFill_Tensor);
  Impl(m, OpName::kMaskedScatter_, AtenMaskedScatter_);
  Impl(m, OpName::kMaskedSelect, AtenMaskedSelect);
  Impl(m, OpName::kMaximumOut, AtenMaximumOut);
  Impl(m, OpName::kMeanOut, AtenMeanOut);
  Impl(m, OpName::kMinimumOut, AtenMinimumOut);
  Impl(m, OpName::kMmOut, AtenMmOut);
  Impl(m, OpName::kMulOut, AtenMulOut);
  Impl(m, OpName::kNativeBatchNorm, AtenNativeBatchNorm);
  Impl(m, OpName::kNativeBatchNormBackward, AtenNativeBatchNormBackward);
  Impl(m, OpName::kNativeBatchNormLegit, AtenNativeBatchNormLegit);
  Impl(m, OpName::kNativeBatchNormLegitNoStats,
       AtenNativeBatchNormLegitNoStats);
  Impl(m, OpName::kNativeBatchNormLegitNoStatsOut,
       AtenNativeBatchNormLegitNoStatsOut);
  Impl(m, OpName::kNativeBatchNormLegitOut, AtenNativeBatchNormLegitOut);
  Impl(m, OpName::kNativeBatchNormOut, AtenNativeBatchNormOut);
  Impl(m, OpName::kNativeGroupNormBackward, AtenNativeGroupNormBackward);
  Impl(m, OpName::kNegOut, AtenNegOut);
  Impl(m, OpName::kNllLoss2dForward, AtenNllLoss2dForward);
  Impl(m, OpName::kNllLoss2dForwardOut, AtenNllLoss2dForwardOut);
  Impl(m, OpName::kNllLossBackwardGradInput, AtenNllLossBackwardGradInput);
  Impl(m, OpName::kNllLossForwardOut, AtenNllLossForwardOut);
  Impl(m, OpName::kNonzero, AtenNonzero);
  Impl(m, OpName::kNonzeroOut, AtenNonzeroOut);
  Impl(m, OpName::kNormalFloatTensor, AtenNormalFloatTensor);
  Impl(m, OpName::kNormalFloatTensorOut, AtenNormalFloatTensorOut);
  Impl(m, OpName::kNormalTensorFloat, AtenNormalTensorFloat);
  Impl(m, OpName::kNormalTensorFloatOut, AtenNormalTensorFloatOut);
  Impl(m, OpName::kNormalTensorTensor, AtenNormalTensorTensor);
  Impl(m, OpName::kNormalTensorTensorOut, AtenNormalTensorTensorOut);
  Impl(m, OpName::kNormal_, AtenNormal_);
  Impl(m, OpName::kPdistForward, AtenPdistForward);
  Impl(m, OpName::kPolarOut, AtenPolarOut);
  Impl(m, OpName::kProd, AtenProd);
  Impl(m, OpName::kProdDimOut, AtenProdDimOut);
  Impl(m, OpName::kRandom_, AtenRandom_);
  Impl(m, OpName::kRandom_From, AtenRandom_From);
  Impl(m, OpName::kRandom_To, AtenRandom_To);
  Impl(m, OpName::kRandpermGeneratorOut, AtenRandpermGeneratorOut);
  Impl(m, OpName::kReflectionPad1dBackwardGradInput,
       AtenReflectionPad1dBackwardGradInput);
  Impl(m, OpName::kReflectionPad1dOut, AtenReflectionPad1dOut);
  Impl(m, OpName::kReflectionPad2d, AtenReflectionPad2d);
  Impl(m, OpName::kReflectionPad2dBackward, AtenReflectionPad2dBackward);
  Impl(m, OpName::kReflectionPad2dBackwardGradInput,
       AtenReflectionPad2dBackwardGradInput);
  Impl(m, OpName::kReflectionPad2dOut, AtenReflectionPad2dOut);
  Impl(m, OpName::kReflectionPad3dBackwardGradInput,
       AtenReflectionPad3dBackwardGradInput);
  Impl(m, OpName::kReflectionPad3dOut, AtenReflectionPad3dOut);
  Impl(m, OpName::kRelu, AtenRelu);
  Impl(m, OpName::kRelu_, AtenRelu_);
  Impl(m, OpName::kReplicationPad1dBackwardGradInput,
       AtenReplicationPad1dBackwardGradInput);
  Impl(m, OpName::kReplicationPad1dOut, AtenReplicationPad1dOut);
  Impl(m, OpName::kReplicationPad2dBackward, AtenReplicationPad2dBackward);
  Impl(m, OpName::kReplicationPad2dBackwardGradInput,
       AtenReplicationPad2dBackwardGradInput);
  Impl(m, OpName::kReplicationPad2dOut, AtenReplicationPad2dOut);
  Impl(m, OpName::kReplicationPad3dBackward, AtenReplicationPad3dBackward);
  Impl(m, OpName::kReplicationPad3dBackwardGradInput,
       AtenReplicationPad3dBackwardGradInput);
  Impl(m, OpName::kReplicationPad3dOut, AtenReplicationPad3dOut);
  Impl(m, OpName::kResize_, AtenResize_);
  Impl(m, OpName::kRoll, AtenRoll);
  Impl(m, OpName::kRoundDecimalsOut, AtenRoundDecimalsOut);
  Impl(m, OpName::kRoundOut, AtenRoundOut);
  Impl(m, OpName::kRshiftScalar, AtenRshiftScalar);
  Impl(m, OpName::kRshiftTensor, AtenRshiftTensor);
  Impl(m, OpName::kScaledDotProductFusedAttentionOverrideable,
       AtenScaledDotProductFusedAttentionOverrideable);
  Impl(m, OpName::kScaledDotProductFusedAttentionOverrideableBackward,
       AtenScaledDotProductFusedAttentionOverrideableBackward);
  Impl(m, OpName::kScatterAddOut, AtenScatterAddOut);
  Impl(m, OpName::kScatterReduceOut, AtenScatterReduceOut);
  Impl(m, OpName::kScatterSrcOut, AtenScatterSrcOut);
  Impl(m, OpName::kScatterValueOut, AtenScatterValueOut);
  Impl(m, OpName::kScatterValueReduceOut, AtenScatterValueReduceOut);
  Impl(m, OpName::kSet_, AtenSet_);
  Impl(m, OpName::kSet_SourceStorage, AtenSet_SourceStorage);
  Impl(m, OpName::kSet_SourceStorageOffset, AtenSet_SourceStorageOffset);
  Impl(m, OpName::kSet_SourceTensor, AtenSet_SourceTensor);
  Impl(m, OpName::kSgnOut, AtenSgnOut);
  Impl(m, OpName::kSigmoidBackwardGradInput, AtenSigmoidBackwardGradInput);
  Impl(m, OpName::kSigmoidOut, AtenSigmoidOut);
  Impl(m, OpName::kSignOut, AtenSignOut);
  Impl(m, OpName::kSignbitOut, AtenSignbitOut);
  Impl(m, OpName::kSiluOut, AtenSiluOut);
  Impl(m, OpName::kSinOut, AtenSinOut);
  Impl(m, OpName::kSinhOut, AtenSinhOut);
  Impl(m, OpName::kSoftmaxBackwardDataOut, AtenSoftmaxBackwardDataOut);
  Impl(m, OpName::kSoftmaxOut, AtenSoftmaxOut);
  Impl(m, OpName::kSplitWithSizesCopyOut, AtenSplitWithSizesCopyOut);
  Impl(m, OpName::kSubOut, AtenSubOut);
  Impl(m, OpName::kSumIntListOut, AtenSumIntListOut);
  Impl(m, OpName::kTake, AtenTake);
  Impl(m, OpName::kTakeOut, AtenTakeOut);
  Impl(m, OpName::kTanOut, AtenTanOut);
  Impl(m, OpName::kTanhBackwardGradInput, AtenTanhBackwardGradInput);
  Impl(m, OpName::kTanhOut, AtenTanhOut);
  Impl(m, OpName::kThresholdBackwardGradInput, AtenThresholdBackwardGradInput);
  Impl(m, OpName::kThresholdOut, AtenThresholdOut);
  Impl(m, OpName::kTopkValues, AtenTopKValues);
  Impl(m, OpName::kTrilIndices, AtenTrilIndices);
  Impl(m, OpName::kTrilOut, AtenTrilOut);
  Impl(m, OpName::kTriuOut, AtenTriuOut);
  Impl(m, OpName::kTruncOut, AtenTruncOut);
  Impl(m, OpName::kUniform_, AtenUniform_);
  Impl(m, OpName::kUpsampleBilinear2dBackwardGradInput,
       AtenUpsampleBilinear2dBackwardGradInput);
  Impl(m, OpName::kUpsampleBilinear2dOut, AtenUpsampleBilinear2dOut);
  Impl(m, OpName::kUpsampleNearest1dBackwardGradInput,
       AtenUpsampleNearest1dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest1dOut, AtenUpsampleNearest1dOut);
  Impl(m, OpName::kUpsampleNearest2dBackwardGradInput,
       AtenUpsampleNearest2dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest2dOut, AtenUpsampleNearest2dOut);
  Impl(m, OpName::kUpsampleNearest3dBackwardGradInput,
       AtenUpsampleNearest3dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest3dOut, AtenUpsampleNearest3dOut);
  Impl(m, OpName::kUpsampleNearestExact1dBackwardGradInput,
       AtenUpsampleNearestExact1dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact1dOut, AtenUpsampleNearestExact1dOut);
  Impl(m, OpName::kUpsampleNearestExact2dBackwardGradInput,
       AtenUpsampleNearestExact2dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact2dOut, AtenUpsampleNearestExact2dOut);
  Impl(m, OpName::kUpsampleNearestExact3dBackwardGradInput,
       AtenUpsampleNearestExact3dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact3dOut, AtenUpsampleNearestExact3dOut);
  Impl(m, OpName::kVdot, AtenVdot);
  Impl(m, OpName::kViewAsReal, AtenViewAsReal);
  Impl(m, OpName::kWhereSelf, AtenWhereSelf);
  Impl(m, OpName::kWhereSelfOut, AtenWhereSelfOut);
  m.impl("_copy_from", AtenCopyFrom);
  m.impl("_foreach_add.List", AtenForeachAddList);
  m.impl("_reshape_alias", AtenReshapeAlias);
  m.impl("abs.out", AtenAbsOut);
  m.impl("addmm.dtype", AtenAddmmDtype);
  m.impl("addmm.dtype_out", AtenAddmmDtypeOut);
  m.impl("addmm.out", AtenAddmmOut);
  m.impl("all.all_out", AtenAllAllOut);
  m.impl("all.out", AtenAllOut);
  m.impl("amax.out", AtenAmaxOut);
  m.impl("amin.out", AtenAminOut);
  m.impl("any.all_out", AtenAnyAllOut);
  m.impl("any.out", AtenAnyOut);
  m.impl("as_strided", AtenAsStrided);
  m.impl("atan2.out", AtenAtan2Out);
  m.impl("bincount", AtenBinCount);
  m.impl("bitwise_and.Tensor_out", AtenBitwiseAndTensorOut);
  m.impl("bitwise_not.out", AtenNotOut);
  m.impl("bitwise_or.Tensor_out", AtenBitwiseOrTensorOut);
  m.impl("bitwise_xor.Tensor_out", AtenBitwiseXorTensorOut);
  m.impl("bmm.dtype", AtenBmmDtype);
  m.impl("bmm.dtype_out", AtenBmmDtypeOut);
  m.impl("bmm.out", AtenBmmOut);
  m.impl("copy_", AtenCopy_);
  m.impl("dot", AtenDot);
  m.impl("empty.memory_format", AtenEmptyMemoryFormat);
  m.impl("empty_strided", AtenEmptyStrided);
  m.impl("eq.Scalar_out", AtenEqScalarOut);
  m.impl("eq.Tensor_out", AtenEqTensorOut);
  m.impl("erf.out", AtenErfOut);
  m.impl("expm1.out", AtenExpm1Out);
  m.impl("fill_.Scalar", AtenFillScalar_);
  m.impl("fill_.Tensor", AtenFillTensor_);
  m.impl("flip", AtenFlip);
  m.impl("floor.out", AtenFloorOut);
  m.impl("floor_divide_.Tensor", AtenFloorDivide_Tensor);
  m.impl("ge.Scalar_out", AtenGeScalarOut);
  m.impl("ge.Tensor_out", AtenGeTensorOut);
  m.impl("gt.Scalar_out", AtenGtScalarOut);
  m.impl("gt.Tensor_out", AtenGtTensorOut);
  m.impl("isin.Scalar_Tensor_out", AtenIsInScalarTensorOut);
  m.impl("isin.Tensor_Scalar_out", AtenIsInTensorScalarOut);
  m.impl("isin.Tensor_Tensor_out", AtenIsInTensorTensorOut);
  m.impl("le.Scalar_out", AtenLeScalarOut);
  m.impl("le.Tensor_out", AtenLeTensorOut);
  m.impl("log.out", AtenLogOut);
  m.impl("logical_and.out", AtenLogicalAndOut);
  m.impl("logical_not.out", AtenLogicalNotOut);
  m.impl("logical_or.out", AtenLogicalOrOut);
  m.impl("logical_xor.out", AtenLogicalXorOut);
  m.impl("lt.Scalar_out", AtenLtScalarOut);
  m.impl("lt.Tensor_out", AtenLtTensorOut);
  m.impl("masked_select.out", AtenMaskedSelectOut);
  m.impl("max", AtenMax);
  m.impl("max.dim_max", AtenMaxDimMax);
  m.impl("max.unary_out", AtenMaxUnaryOut);
  m.impl("max_pool2d_with_indices.out", AtenMaxPool2dWithIndicesOut);
  m.impl("max_pool2d_with_indices_backward.grad_input",
         AtenMaxPool2dWithIndicesBackwardGradInput);
  m.impl("max_pool3d_with_indices", AtenMaxPool3dWithIndices);
  m.impl("max_pool3d_with_indices.out", AtenMaxPool3dWithIndicesOut);
  m.impl("max_pool3d_with_indices_backward", AtenMaxPool3dWithIndicesBackward);
  m.impl("max_pool3d_with_indices_backward.grad_input",
         AtenMaxPool3dWithIndicesBackwardGradInput);
  m.impl("min", AtenMin);
  m.impl("min.dim_min", AtenMinDimMin);
  m.impl("min.unary_out", AtenMinUnaryOut);
  m.impl("mse_loss.out", tpu_aten_mse_loss_out);
  m.impl("mse_loss_backward", tpu_aten_mse_loss_backward);
  m.impl("multinomial", AtenMultinomial);
  m.impl("multinomial.out", AtenMultinomialOut);
  m.impl("native_dropout", AtenDropout);
  m.impl("native_dropout_backward", AtenNativeDropoutBackward);
  m.impl("native_layer_norm", AtenNativeLayerNorm);
  m.impl("ne.Scalar_out", AtenNeScalarOut);
  m.impl("ne.Tensor_out", AtenNeTensorOut);
  m.impl("pow.Scalar_out", AtenPowScalarOut);
  m.impl("pow.Tensor_Scalar_out", AtenPowTensorScalarOut);
  m.impl("pow.Tensor_Tensor_out", AtenPowTensorTensorOut);
  m.impl("reciprocal.out", AtenReciprocalOut);
  m.impl("remainder.Scalar_Tensor", AtenRemainderScalarTensor);
  m.impl("remainder.Tensor_out", AtenRemainderTensorOut);
  m.impl("rsqrt.out", AtenRsqrtOut);
  m.impl("rsub.Tensor", AtenRsubTensor);
  m.impl("sort.values_stable", AtenSortValuesStable);
  m.impl("sqrt.out", AtenSqrtOut);
  m.impl("unfold", AtenUnfold);
  m.impl("var.correction", AtenVar);
  m.impl("var.correction_out", AtenVarOut);
  m.impl("view", AtenView);
  m.impl("view_as_complex", AtenViewAsComplex);
  m.impl("zero_", AtenZero_);
  // go/keep-sorted end
}

// Register the cpu_fallback for missing operators. This is mostly for testing
// and debugging. This way, as least we have models running, without being
// blocked by certain missing operators. By default, the fallback is disabled.
//
// To change fallback mode the intended way, use python context manager:
// `with fallback_mode(FallbackMode.ALLOW_FALLBACK):`
//
void TpuMissingOpFallback(const c10::OperatorHandle& op,
                          torch::jit::Stack* const stack) {
  const auto& op_name = op.schema().operator_name();
  if (!IsCpuFallbackEnabled()) {
    TT_CHECK_THROW(false, error::kUnimplemented)
        << "operator '" << op_name
        << "' is not implemented for TPU. Please file a feature request";
  } else {
    ABSL_LOG(WARNING) << "operator '" << op_name
                      << "' is not implemented for TPU. Falling back to CPU";
    at::native::cpu_fallback(op, stack);
  }
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&TpuMissingOpFallback>());
}

// TODO: (b/448113143) -- Once TPU is upstreamed, remove this fallthrough
//  and add an AutogradTPU entry to VariableFallbackKernel.cpp.
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY(torch_tpu, m) {
  m.def("ragged_dot(Tensor lhs, Tensor rhs, Tensor group_sizes) -> Tensor");
  m.def(
      "ragged_dot.out(Tensor lhs, Tensor rhs, Tensor group_sizes, *, "
      "Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_tpu, PrivateUse1, m) {
  m.impl("ragged_dot", AtenRaggedDot);
  m.impl("ragged_dot.out", AtenRaggedDotOut);
}

TORCH_LIBRARY_IMPL(torch_tpu, CPU, m) {
  m.impl("ragged_dot", AtenRaggedDot);
  m.impl("ragged_dot.out", AtenRaggedDotOut);
}

bool& MutableIsCpuFallbackEnabled() {
  // Backward runs in a different thread, the static state is shared with the
  // entire process.
  static absl::NoDestructor<bool> cpu_fallback_enabled{false};
  return *cpu_fallback_enabled;
}

void EnableCpuFallback(bool enabled) {
  MutableIsCpuFallbackEnabled() = enabled;
}

bool IsCpuFallbackEnabled() { return MutableIsCpuFallbackEnabled(); }

}  // namespace torch_tpu

namespace at::native {
// Per https://github.com/pytorch/pytorch/issues/162989, to override
// _fused_sdp_choice We will need to REGISTER_PRIVATEUSE1_DISPATCH against
// _fused_sdp_choice_stub. See the example here:
// https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub,
                              &torch_tpu::AtenFusedSdpChoice);
}  // namespace at::native
