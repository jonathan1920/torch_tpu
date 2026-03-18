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

#include "torch_tpu/ops/macros/kernel.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::Pair;
using testing::StartsWith;

// Per
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html,
// this disables C++ context in pytorch errors. This can be done only once per
// process as pytorch caches the value of this variable.
static const auto kInitShowCppContext =
    setenv("TORCH_SHOW_CPP_STACKTRACES", "0", /*overwrite=*/1);

void Kernel1(int ndim) {
  TT_KERNEL(OpName::kAdd, _, (ndim), {
    throw  // For testing error API.
        TtError(TT_ERROR(error::kInvalidArgument) << "test error",
                c10::SourceLocation({"foo()", "bar.cc", 42}));
  });
}

TEST(TtKernel, PrependOpName) {
  bool thrown_c10_error = false;
  try {
    Kernel1(1);
  } catch (const c10::Error& e) {
    thrown_c10_error = true;
    EXPECT_EQ(std::string(e.what_without_backtrace()), "add(): test error");
    EXPECT_THAT(std::string(e.what()),
                StartsWith("add(): test error\n"
                           "Exception raised from foo() at bar.cc:42 "));
  }
  EXPECT_TRUE(thrown_c10_error);
}

// A delegated-to op.
at::Tensor AtenOp2(bool cond) {
  TT_KERNEL(OpName::kBmm, _, (cond), {
    // Even though this error is thrown in op2, the root op is op3 as we
    // are calling op2 from op3.
    throw  // For testing error API.
        TtError(TT_ERROR(error::kInvalidArgument) << "some error",
                c10::SourceLocation({"foo()", "bar.cc", 43}));
  });
}

void NewKernel1(int ndim) {
  TT_KERNEL(OpName::kCatOut, _, (ndim), { AtenOp2(true); });
}

TEST(TtKernel, PrependRootOpName) {
  bool thrown_c10_error = false;
  const int ndim = 1;
  try {
    // op3 calls op2, which throws an error.
    NewKernel1(ndim);
  } catch (const c10::Error& e) {
    thrown_c10_error = true;
    // The error thrown by op2 is associated with op3, the root op.
    EXPECT_EQ(std::string(e.what_without_backtrace()), "cat(): some error");
    EXPECT_THAT(std::string(e.what()),
                StartsWith("cat(): some error\n"
                           "Exception raised from foo() at bar.cc:43 "));
  }
  EXPECT_TRUE(thrown_c10_error);
}

void Kernel3(int ndim, double alpha, std::optional<int> seed) {
  TT_KERNEL(OpName::kAdd, param_keys, (ndim, alpha, seed), {
    EXPECT_THAT(param_keys, ElementsAre(
                                // go/keep-sorted start
                                Pair("alpha", "2.5"),  //
                                Pair("ndim", "3"),     //
                                Pair("seed", "42")     //
                                // go/keep-sorted end
                                ));
  });
}

TEST(TtKernel, ComputesCacheKeysWithNonTensors) {
  const int ndim = 3;
  const double alpha = 2.5;
  const std::optional<int> seed = 42;
  Kernel3(ndim, alpha, seed);
}

void Kernel4(const at::Tensor& self, int ndim, bool expand,
             std::optional<int> seed) {
  TT_KERNEL(OpName::kAdd, param_keys, (self, ndim, expand, seed), {
    // `self` is a Tensor and thus shouldn't be in the cache keys.
    // `seed` is nullopt and thus should be omitted from the cache keys.
    EXPECT_THAT(param_keys, ElementsAre(
                                // go/keep-sorted start
                                Pair("expand", "t"),  //
                                Pair("ndim", "3")     //
                                // go/keep-sorted end
                                ));
  });
}

TEST(TtKernel, ComputesCacheKeysWithTensorAndNonTensors) {
  const at::Tensor self = at::ones(1);
  const int ndim = 3;
  const bool expand = true;
  const std::optional<int> seed = std::nullopt;
  Kernel4(self, ndim, expand, seed);
}

void Kernel0() {
  TT_KERNEL(OpName::kAdd, _, (),
            {
                // Do nothing.
            });
}

TEST(TtKernel, SupportsNullaryOps) { Kernel0(); }

}  // namespace
}  // namespace torch_tpu
