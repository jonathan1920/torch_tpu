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

#include "torch_tpu/common/repeated_subsequence.h"

#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;

TEST(FindRepeatedSubsequences, EmptySpan) {
  std::string_view s = "";
  auto repeats = FindRepeatedSubsequences(absl::MakeSpan(s.begin(), s.end()));
  EXPECT_TRUE(repeats.empty());
}

TEST(FindRepeatedSubsequences, WithNoRepetition) {
  std::string_view s = "abcd";
  auto repeats = FindRepeatedSubsequences(absl::MakeSpan(s.begin(), s.end()));
  EXPECT_TRUE(repeats.empty());
}

TEST(FindRepeatedSubsequences, WithShortRepeatedSubstring) {
  std::string_view s = "abcbd";
  auto repeats = FindRepeatedSubsequences(absl::MakeSpan(s.begin(), s.end()),
                                          /*min_seq_len=*/5);
  EXPECT_TRUE(repeats.empty());
}

TEST(FindRepeatedSubsequences, WithRepetitions1) {
  std::string_view s = "banana";
  auto repeats = FindRepeatedSubsequences(absl::MakeSpan(s.begin(), s.end()));
  ASSERT_EQ(repeats.size(), 3);
  EXPECT_THAT(repeats[0], ElementsAre('a'));
  EXPECT_THAT(repeats[1], ElementsAre('a', 'n'));
  EXPECT_THAT(repeats[2], ElementsAre('n', 'a'));
}

TEST(FindRepeatedSubsequences, WithRepetitions2) {
  std::vector<int> v = {1, 2, 3, 4, 5, 2, 3, 6, 7, 2, 3, 8};
  auto repeats = FindRepeatedSubsequences(absl::Span<int>(v));
  ASSERT_EQ(repeats.size(), 2);
  EXPECT_THAT(repeats[0], ElementsAre(2, 3));
  EXPECT_THAT(repeats[1], ElementsAre(3));
}

TEST(FindRepetitionOffsets, WithEmptyString) {
  std::string_view s1 = "";
  std::string_view s2 = "a";
  auto offsets = FindRepetitionOffsets(absl::MakeSpan(s1.begin(), s1.end()),
                                       absl::MakeSpan(s2.begin(), s2.end()));
  EXPECT_TRUE(offsets.empty());
}

TEST(FindRepetitionOffsets, WithEmptySubstring) {
  std::string_view s1 = "abc";
  std::string_view s2 = "";
  auto offsets = FindRepetitionOffsets(absl::MakeSpan(s1.begin(), s1.end()),
                                       absl::MakeSpan(s2.begin(), s2.end()));
  EXPECT_TRUE(offsets.empty());
}

TEST(FindRepetitionOffsets, WithRepeatedSubstring1) {
  std::string_view s1 = "banana";
  std::string_view s2 = "an";
  auto offsets = FindRepetitionOffsets(absl::MakeSpan(s1.begin(), s1.end()),
                                       absl::MakeSpan(s2.begin(), s2.end()));
  EXPECT_THAT(offsets, ElementsAre(1, 3));
}

TEST(FindRepetitionOffsets, WithRepeatedSubstring2) {
  std::vector<int> v1 = {1, 2, 3, 4, 5, 2, 3, 6, 7, 2, 3, 8};
  std::vector<int> v2 = {2, 3};
  auto offsets =
      FindRepetitionOffsets(absl::Span<int>(v1), absl::Span<int>(v2));
  EXPECT_THAT(offsets, ElementsAre(1, 5, 9));
}

}  // namespace
}  // namespace torch_tpu
