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

#include "torch_tpu/common/fingerprint_utils.h"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

TEST(Fingerprint, WorksForIntegers) {
  EXPECT_EQ(Fingerprint(1), 1);
  EXPECT_EQ(Fingerprint(1234567890), 1234567890);
  EXPECT_EQ(Fingerprint(-1), static_cast<FingerprintType>(-1));
  EXPECT_EQ(Fingerprint(std::numeric_limits<int>::max()),
            std::numeric_limits<int>::max());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int>::min()),
            std::numeric_limits<int>::min());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int64_t>::max()),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int64_t>::min()),
            std::numeric_limits<int64_t>::min());
}

TEST(Fingerprint, WorksForEnums) {
  enum class Colors {
    kRed = 1,
    kGreen = 2,
    kBlue = 3,
  };
  EXPECT_EQ(Fingerprint(Colors::kRed), 1);
  EXPECT_EQ(Fingerprint(Colors::kGreen), 2);
  EXPECT_EQ(Fingerprint(Colors::kBlue), 3);

  EXPECT_EQ(Fingerprint(mlir::ElementType::F32),
            static_cast<FingerprintType>(mlir::ElementType::F32));
}

TEST(Fingerprint, WorksForSpan) {
  int a[] = {1, 2, 3};
  absl::Span<int> a_span(a, 3);
  auto fp = Fingerprint(a_span);
  // We deliberately hard-code the number here to ensure that the fingerprint
  // behavior doesn't change.
  EXPECT_EQ(fp, 4382234674942165662ULL);
}

TEST(Fingerprint, WorksForEmptySpan) {
  absl::Span<int> a_span;
  auto fp = Fingerprint(a_span);
  // We deliberately hard-code the number here to ensure that the fingerprint
  // behavior doesn't change.
  EXPECT_EQ(fp, 0);
}

TEST(Fingerprint, WorksForEnumSpan) {
  const mlir::ElementType a[] = {mlir::ElementType::F32, mlir::ElementType::F64,
                                 mlir::ElementType::BF16};
  absl::Span<const mlir::ElementType> span(a, 3);
  const auto fp = Fingerprint(span);
  // We deliberately hard-code the number here to ensure that the fingerprint
  // behavior doesn't change.
  EXPECT_EQ(fp, 11166452756956321874ULL);
}

TEST(Fingerprint, WorksForVector) {
  std::vector<mlir::ElementType> a = {
      mlir::ElementType::F32, mlir::ElementType::F64, mlir::ElementType::BF16};
  const auto fp = Fingerprint(a);
  // We deliberately hard-code the number here to ensure that the fingerprint
  // behavior doesn't change.
  EXPECT_EQ(fp, 11166452756956321874ULL);
}

TEST(Fingerprint, WorksForSpanSpan) {
  const mlir::ElementType a[] = {mlir::ElementType::F32, mlir::ElementType::F64,
                                 mlir::ElementType::BF16};
  const mlir::ElementType b[] = {mlir::ElementType::I16, mlir::ElementType::I2};
  absl::Span<const mlir::ElementType> span_a(a, 3), span_b(b, 2);
  std::vector<absl::Span<const mlir::ElementType>> span_span = {span_a, span_b};
  const auto fp = Fingerprint(span_span);
  // We deliberately hard-code the number here to ensure that the fingerprint
  // behavior doesn't change.
  EXPECT_EQ(fp, 5245874435612141896ULL);
}

TEST(Fingerprint, WorksForBool) {
  EXPECT_EQ(Fingerprint(true), 1);
  EXPECT_EQ(Fingerprint(false), 0);
}

TEST(Fingerprint, WorksForCString) {
  EXPECT_EQ(Fingerprint("hello"), 13009744463427800296ULL);
}

TEST(Fingerprint, WorksForStringView) {
  EXPECT_EQ(Fingerprint(std::string_view("hello")), 13009744463427800296ULL);
}

TEST(Fingerprint, WorksForStringObject) {
  std::string s = "hello";
  EXPECT_EQ(Fingerprint(s), 13009744463427800296ULL);
}

TEST(Finterprint, WorksForPair) {
  std::pair<int, std::string> p = {1, "hi"};
  EXPECT_EQ(Fingerprint(p), 2362095114178386943ULL);
}

TEST(Fingerprint, WorksForEmptyMap) {
  std::map<int, std::string> m;
  EXPECT_EQ(Fingerprint(m), 0);
}

TEST(Fingerprint, WorksForMap) {
  std::map<int, std::string> m = {{1, "hello"}, {2, "world"}};
  EXPECT_EQ(Fingerprint(m), 11808310844027338525ULL);
}

TEST(FingerprintCat, WorksForZeroArgs) { EXPECT_EQ(FingerprintCat(), 0); }

TEST(FingerprintCat, WorksForOneArg) {
  EXPECT_EQ(FingerprintCat(1), 1);
  EXPECT_EQ(FingerprintCat(1234567890), 1234567890);
}

TEST(FingerprintCat, WorksForMultipleArgs) {
  EXPECT_EQ(FingerprintCat(1, std::string("hi")), 2362095114178386943ULL);
  EXPECT_EQ(FingerprintCat(1, 2, 3), 10629069731271099820ULL);
  EXPECT_EQ(FingerprintCat(1, 2, 3, 4), 6144917672450084373ULL);
}

}  // namespace
}  // namespace torch_tpu
