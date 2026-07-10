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

#include "torch_tpu/ops/polygamma/polygamma_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/digamma/digamma_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

double Factorial(int64_t n) {
  double f = 1.0;
  for (int64_t i = 2; i <= n; ++i) {
    f *= i;
  }
  return f;
}

constexpr double kPi = 3.14159265358979323846;

std::vector<double> ComputeCotDerivativeCoeffs(int n) {
  if (n == 0) return {0.0, 1.0};

  std::vector<double> prev = ComputeCotDerivativeCoeffs(n - 1);
  std::vector<double> coeffs(n + 2, 0.0);

  coeffs[0] = -prev[1];
  if (prev.size() > 2) {
    coeffs[1] = -2.0 * prev[2];
  }
  for (int k = 2; k <= n + 1; ++k) {
    double term1 = 0.0;
    if (k + 1 < prev.size()) {
      term1 = -(k + 1) * prev[k + 1];
    }
    double term2 = -(k - 1) * prev[k - 1];
    coeffs[k] = term1 + term2;
  }
  return coeffs;
}

mlir::MlirOp EvaluatePolynomial(mlir::MlirOp y,
                                const std::vector<double>& coeffs) {
  auto accum = MakeConstantLike(y, coeffs.back());
  for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
    accum = mlir::stablehlo::Mul(accum, y);
    auto c = MakeConstantLike(y, coeffs[i]);
    accum = mlir::stablehlo::Add(accum, c);
  }
  return accum;
}

absl::StatusOr<mlir::MlirOp> BuildPolygammaShlo(
    mlir::MlirOp self_op, int64_t n, mlir::ElementType out_mlir_type) {
  mlir::ElementType input_dtype = GetElementTypeOrDie(self_op);
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_type,
                      InferComputationDtype(input_dtype));

  mlir::MlirOp input_op = self_op;
  if (input_dtype != computation_type) {
    input_op = mlir::stablehlo::ConvertElementType(self_op, computation_type);
  }

  if (n == 0) {  // Digamma
    return BuildDigammaShlo(self_op, out_mlir_type);
  }

  // 1. Direct path (x >= 0.5)
  // psi^(n)(x) = (-1)^(n+1) * n! * zeta(n+1, x)
  double sign_val = ((n + 1) % 2 == 0) ? 1.0 : -1.0;
  double coeff_val = sign_val * Factorial(n);
  auto coeff_cst = MakeConstantLike(input_op, coeff_val);
  auto n_plus_one_cst = MakeConstantLike(input_op, static_cast<double>(n + 1));
  auto zeta_op = mlir::chlo::BroadcastZeta(n_plus_one_cst, input_op);
  auto direct_op = mlir::stablehlo::Mul(coeff_cst, zeta_op);

  // 2. Reflection path (x < 0.5)
  // psi^(n)(x) = (-1)^n * psi^(n)(1-x) - pi^(n+1) * d^n/dx^n (cot(pi x))
  auto one = MakeConstantLike(input_op, 1.0);
  auto one_minus_x = mlir::stablehlo::Subtract(one, input_op);
  auto zeta_1_minus_x = mlir::chlo::BroadcastZeta(n_plus_one_cst, one_minus_x);

  // First term: (-1)^n * psi^(n)(1-x) = -n! * zeta(n+1, 1-x)
  double first_term_coeff = -Factorial(n);
  auto first_term_coeff_cst = MakeConstantLike(input_op, first_term_coeff);
  auto first_term = mlir::stablehlo::Mul(first_term_coeff_cst, zeta_1_minus_x);

  // Second term: -pi^(n+1) * poly_op
  auto pi_cst = MakeConstantLike(input_op, kPi);
  auto pi_x = mlir::stablehlo::Mul(input_op, pi_cst);
  auto cos_pi_x = mlir::stablehlo::Cosine(pi_x);
  auto sin_pi_x = mlir::stablehlo::Sine(pi_x);
  auto cot_pi_x = mlir::stablehlo::Div(cos_pi_x, sin_pi_x);

  std::vector<double> coeffs = ComputeCotDerivativeCoeffs(n);
  mlir::MlirOp poly_op = EvaluatePolynomial(cot_pi_x, coeffs);

  double pi_pow_val = -std::pow(kPi, n + 1);
  auto pi_pow_cst = MakeConstantLike(input_op, pi_pow_val);
  auto second_term = mlir::stablehlo::Mul(pi_pow_cst, poly_op);

  auto reflection_op = mlir::stablehlo::Add(first_term, second_term);

  // 3. Selection and Pole Handling
  auto point_five = MakeConstantLike(input_op, 0.5);
  auto is_lt_point_five = mlir::stablehlo::Compare(
      input_op, point_five, mlir::stablehlo::ComparisonDirection::LT);
  auto selected_op =
      mlir::stablehlo::Select(is_lt_point_five, reflection_op, direct_op);

  mlir::MlirOp result;
  if (n > 1) {
    // Polygamma (n > 1): returns inf/-inf for non-positive integers.
    auto floor_op = mlir::stablehlo::Floor(input_op);
    auto is_integer = mlir::stablehlo::Compare(
        input_op, floor_op, mlir::stablehlo::ComparisonDirection::EQ);
    auto zero_cst = MakeConstantLike(input_op, 0.0);
    auto is_le_zero = mlir::stablehlo::Compare(
        input_op, zero_cst, mlir::stablehlo::ComparisonDirection::LE);
    auto is_pole = mlir::stablehlo::And(is_le_zero, is_integer);

    bool is_n_even = (n % 2 == 0);
    double inf_val = is_n_even ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
    auto inf_cst = MakeConstantLike(input_op, inf_val);
    result = mlir::stablehlo::Select(is_pole, inf_cst, selected_op);
  } else {
    // Trigamma (n=1): returns finite values at poles (using reflection
    // directly).
    result = selected_op;
  }

  if (GetElementTypeOrDie(result) != out_mlir_type) {
    return mlir::stablehlo::ConvertElementType(result, out_mlir_type);
  }
  return result;
}

}  // namespace

at::Tensor& AtenPolygammaOut(int64_t n, const at::Tensor& self,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kPolygammaOut, param_keys, (n, self, out), {
    TT_CHECK_THROW(n >= 0, error::kInvalidArgument)
        << "expected n to be non-negative, got " << n;
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());

    auto computation_type = self.scalar_type();
    if (c10::isIntegralType(computation_type, /*includeBool=*/true)) {
      computation_type = at::ScalarType::Float;
    }
    // Enforce safe type promotion (casting to integer outputs is not safe)
    TT_CHECK_THROW(out.is_floating_point() || out.is_complex(),
                   error::kInvalidArgument)
        << "expected the output dtype to be floating point or complex, got "
        << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_ASSIGN_OR_THROW(auto computation_mlir_type,
                       ConvertTo<mlir::ElementType>(computation_type));

    auto op_builder =
        [n, out_dtype](mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPolygammaShlo(self_op, n, out_dtype);
    };

    TT_THROW_IF_ERROR(UnaryOpOut(self, out, std::move(op_builder),
                                 {.op_param_cache_keys = std::move(param_keys),
                                  .out_dtype = out_dtype,
                                  .computation_dtype = computation_mlir_type}));
    return out;
  });
}
}  // namespace torch_tpu
