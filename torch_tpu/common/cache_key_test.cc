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

#include "torch_tpu/common/cache_key.h"

#include <cstdint>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Dimname.h"
#include "ATen/core/symbol.h"
#include "c10/core/ConstantSymNodeImpl.h"
#include "c10/core/Device.h"
#include "c10/core/Layout.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/SymNodeImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/xla_data.pb.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::IsEmpty;
using testing::Pair;

TEST(OpParamCacheKeys, DefaultIsEmpty) {
  OpParamCacheKeys params;
  EXPECT_TRUE(params.begin() == params.end());
  EXPECT_THAT(params, IsEmpty());
  EXPECT_EQ(params.size(), 0);
}

TEST(OpParamCacheKeys, SetParamScalar) {
  auto params = (*OpParamCacheKeys::SetParam("foo", at::Scalar(123))).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "123:Long")));
}

TEST(OpParamCacheKeys, SetParamScalarType) {
  auto params =
      (*OpParamCacheKeys::SetParam("foo", at::ScalarType::Float)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "Float")));
}

TEST(OpParamCacheKeys, SetParamReduceOp) {
  const c10d::ReduceOp reduce_op = c10d::ReduceOp::SUM;
  auto params = (*OpParamCacheKeys::SetParam("foo", reduce_op)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "sum")));

  const c10d::ReduceOp reduce_op2 = c10d::ReduceOp::MAX;
  auto params2 = (*OpParamCacheKeys::SetParam("foo", reduce_op2)).value();
  EXPECT_THAT(params2, ElementsAre(Pair("foo", "max")));
}

TEST(OpParamCacheKeys, SetParamMlirElementType) {
  auto params =
      (*OpParamCacheKeys::SetParam("foo", mlir::ElementType::F32)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "f32")));
}

TEST(OpParamCacheKeys, SetParamInteger) {
  auto params = (*OpParamCacheKeys::SetParam("foo", 1234567890L)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "1234567890")));
}

TEST(OpParamCacheKeys, SetParamBool) {
  auto params = (*OpParamCacheKeys::SetParam("bar", true)).value();
  EXPECT_THAT(params, ElementsAre(Pair("bar", "true")));

  auto params2 = (*OpParamCacheKeys::SetParam("bar", false)).value();
  EXPECT_THAT(params2, ElementsAre(Pair("bar", "false")));
}

TEST(OpParamCacheKeys, SetParamString) {
  auto params = (*OpParamCacheKeys::SetParam("foo", "a,bar=b")).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "a,bar=b")));

  auto params2 = (*OpParamCacheKeys::SetParam("foo", "\"\n")).value();
  EXPECT_THAT(params2, ElementsAre(Pair("foo", "\"\n")));
}

TEST(OpParamCacheKeys, SetParamIntSpan) {
  auto params =
      (*OpParamCacheKeys::SetParam("foo", Dimensions({1, 2, 3}))).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "[1,2,3]")));
}

TEST(OpParamCacheKeys, SetParamDouble) {
  auto params = (*OpParamCacheKeys::SetParam("foo", 4.5)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "4.5")));
}

TEST(OpParamCacheKeys, SetParamNullopt) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  auto params = (*OpParamCacheKeys::SetParam("foo", no_scalar)).value();
  EXPECT_THAT(params, IsEmpty());

  const std::optional<at::ScalarType> no_scalar_type = std::nullopt;
  auto params2 = (*OpParamCacheKeys::SetParam("foo", no_scalar_type)).value();
  EXPECT_THAT(params2, IsEmpty());

  const std::optional<int64_t> no_int64 = std::nullopt;
  auto params3 = (*OpParamCacheKeys::SetParam("foo", no_int64)).value();
  EXPECT_THAT(params3, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamNewOverwritesOld) {
  auto params = (*OpParamCacheKeys::SetParam("foo", at::Scalar(123))
                      .SetParam("foo", at::Scalar(456)))
                    .value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "456:Long")));
}

TEST(OpParamCacheKeys, SetParamNewNulloptRemovesOld) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  auto params = (*OpParamCacheKeys::SetParam("foo", at::Scalar(123))
                      .SetParam("foo", no_scalar))
                    .value();
  EXPECT_THAT(params, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamLayout) {
  auto params = (*OpParamCacheKeys::SetParam("foo", at::Layout::Strided)
                      .SetParam("bar", at::Layout::Sparse))
                    .value();
  EXPECT_THAT(params,
              ElementsAre(Pair("bar", "Sparse"), Pair("foo", "Strided")));
}

TEST(OpParamCacheKeys, SetParamMemoryFormat) {
  auto params =
      (*OpParamCacheKeys::SetParam("foo", at::MemoryFormat::Contiguous)
            .SetParam("bar", at::MemoryFormat::ChannelsLast3d))
          .value();
  EXPECT_THAT(params, ElementsAre(Pair("bar", "ChannelsLast3d"),
                                  Pair("foo", "Contiguous")));
}

TEST(OpParamCacheKeys, SetParamDimname) {
  auto params = (*OpParamCacheKeys::SetParam(
                     "foo", at::Dimname::fromSymbol(at::Symbol::dimname("N"))))
                    .value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "dimname::N")));
}

TEST(OpParamCacheKeys, SetParamSymInt) {
  auto params = (*OpParamCacheKeys::SetParam("foo", c10::SymInt(123))).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "123")));

  c10::SymNode sym_node =
      c10::make_intrusive<c10::ConstantSymNodeImpl<int64_t>>(456);
  auto params2 =
      (*OpParamCacheKeys::SetParam("foo", c10::SymInt(sym_node))).value();
  EXPECT_THAT(params2, ElementsAre(Pair("foo", "456")));
}

TEST(OpParamCacheKeys, SetParamSymIntArrayRef) {
  c10::SymNode sym_node =
      c10::make_intrusive<c10::ConstantSymNodeImpl<int64_t>>(456);
  c10::SymInt si[] = {c10::SymInt(123), c10::SymInt(sym_node)};
  c10::SymIntArrayRef sir(si);
  auto params = (*OpParamCacheKeys::SetParam("foo", sir)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "[123,456]")));

  c10::SymIntArrayRef empty_sym_int_array_ref;
  auto params2 =
      (*OpParamCacheKeys::SetParam("foo", empty_sym_int_array_ref)).value();
  EXPECT_THAT(params2, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamDevice) {
  auto params = (*OpParamCacheKeys::SetParam("foo", at::Device("cpu"))
                      .SetParam("bar", at::Device("cuda:1")))
                    .value();
  EXPECT_THAT(params, ElementsAre(Pair("bar", "cuda:1"), Pair("foo", "cpu")));
}

TEST(OpParamCacheKeys, SetParamAllreduceOptions) {
  c10d::AllreduceOptions options;
  options.reduceOp = c10d::ReduceOp::SUM;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "sum")));
}

TEST(OpParamCacheKeys, SetParamReduceScatterOptions) {
  c10d::ReduceScatterOptions options;
  options.reduceOp = c10d::ReduceOp::PRODUCT;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "product")));
}

TEST(OpParamCacheKeys, SetParamBroadcastOptions) {
  c10d::BroadcastOptions options;
  options.rootRank = 1;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "1")));
}

TEST(OpParamCacheKeys, SetParamScatterOptions) {
  c10d::ScatterOptions options;
  options.rootRank = 2;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, ElementsAre(Pair("foo", "2")));
}

TEST(OpParamCacheKeys, SetParamAllgatherOptions) {
  c10d::AllgatherOptions options;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamAllToAllOptions) {
  c10d::AllToAllOptions options;
  auto params = (*OpParamCacheKeys::SetParam("foo", options)).value();
  EXPECT_THAT(params, IsEmpty());
}

}  // namespace
}  // namespace torch_tpu
