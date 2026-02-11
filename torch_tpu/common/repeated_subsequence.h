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

#ifndef TORCH_TPU_COMMON_REPEATED_SUBSEQUENCE_H_
#define TORCH_TPU_COMMON_REPEATED_SUBSEQUENCE_H_

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {

namespace internal {

template <typename T>
absl::Span<T> MakeSuffix(absl::Span<T> input, size_t suffix_offset) {
  return absl::MakeSpan(input.begin() + suffix_offset, input.end());
}

}  // namespace internal

// Returns repeated, non-overlapping subsequences in sequence `input` having a
// minimal length of `min_seq_len`. Note this algorithm may not return a
// repeated substring that is prefix of a larger substring that is returned by
// the algorithm. As an example, consider input sequence "banana", the function
// will return "a", "an", and "na" as repeated subsequences.
template <typename T>
std::vector<absl::Span<T>> FindRepeatedSubsequences(absl::Span<T> input,
                                                    int min_seq_len = 1) {
  ABSL_VLOG(2) << "FindRepeatedSubsequence " << ToString(input);
  if (input.empty() || input.size() < 2 * min_seq_len) {
    return {};
  }

  // Step 1: Generate a Suffix Array, i.e., create an array of all possible
  // suffixes (endings) of your string and sort them lexicographically. In
  // practice, we need to track only the suffix offsets.
  //
  // Example:
  //   Input:  banana
  //   Suffixes: banana, anana, nana, ana, na, a
  //
  std::vector<size_t> suffix_offsets;
  suffix_offsets.reserve(input.size());
  for (auto i = 0; i < input.size(); ++i) {
    suffix_offsets.push_back(i);
  }
#ifndef NDEBUG
  ABSL_VLOG(2) << "Suffix Offsets: " << ToString(suffix_offsets);
  for (auto suffix_offset : suffix_offsets) {
    ABSL_VLOG(2) << ">>> "
                 << ToString(::torch_tpu::internal::MakeSuffix(input,
                                                               suffix_offset));
  }
#endif

  // Step 2: Sort the suffixes, lexicographically.
  //
  // Example:
  //   Sorted Suffixes: a, ana, anana, banana, na, nana
  //
  auto compare_suffix_offsets = [input](size_t offset1, size_t offset2) {
    return std::lexicographical_compare(input.begin() + offset1, input.end(),
                                        input.begin() + offset2, input.end());
  };
  absl::c_sort(suffix_offsets, compare_suffix_offsets);
#ifndef NDEBUG
  ABSL_VLOG(2) << "Sorted Suffixes: " << ToString(suffix_offsets);
  for (auto suffix_offset : suffix_offsets) {
    ABSL_VLOG(2) << ">>> "
                 << ToString(::torch_tpu::internal::MakeSuffix(input,
                                                               suffix_offset));
  }
#endif

  // Step 3: Compute an Longest Common Prefix (LCP) array. For each suffix in
  // the sorted array, the LCP value is the length of the common prefix it
  // shares with the previous suffix without overlap.
  //
  // Example:
  //   LCP(a, ana) = 1 (prefix "a" with length 1 repeats)
  //   LCP(ana, anana) = 2 (prefix "an" with length 2 repeats)
  //   LCP(anana, banana) = LCP(banana, na) = 0 (no common prefix)
  //   LCP(na, nana) = 2 (prefix "na" with length 2 repeats)
  //
  std::vector<absl::Span<T>> repeated_subsequences;
  for (auto i = 0; i < suffix_offsets.size() - 1; ++i) {
    int lcp = 0;
    auto iter1 = input.begin() + suffix_offsets[i];
    auto iter2 = input.begin() + suffix_offsets[i + 1];
    while (iter1 != input.end() && iter2 != input.end() && *iter1 == *iter2) {
      // Terminate if we detect an overlap.
      if ((iter1 == input.begin() + suffix_offsets[i + 1]) ||
          (iter2 == input.begin() + suffix_offsets[i])) {
        break;
      }
      lcp++;
      ++iter1;
      ++iter2;
    }
    if (lcp > 0 && lcp >= min_seq_len) {
      auto subseq = absl::MakeSpan(input.begin() + suffix_offsets[i],
                                   input.begin() + suffix_offsets[i] + lcp);
      ABSL_VLOG(2) << "Found repeated substring: " << ToString(subseq);
      repeated_subsequences.push_back(subseq);
    }
  }
  // Remove duplicates.
  absl::c_sort(repeated_subsequences);
  repeated_subsequences.erase(
      std::unique(repeated_subsequences.begin(), repeated_subsequences.end()),
      repeated_subsequences.end());

  return repeated_subsequences;
}

// Returns the list of offsets from the beginning of `sequence` where
// `subsequence` appears in `sequence`.
template <typename T>
std::vector<size_t> FindRepetitionOffsets(absl::Span<T> sequence,
                                          absl::Span<T> subsequence) {
  if (sequence.empty() || subsequence.empty()) {
    return {};
  }

  std::vector<size_t> offsets;
  auto iter = sequence.begin();
  while (true) {
    iter = std::search(iter, sequence.end(), subsequence.begin(),
                       subsequence.end());
    if (iter == sequence.end()) {
      break;
    }
    offsets.push_back(std::distance(sequence.begin(), iter));
    iter += subsequence.size();
  }
  return offsets;
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_REPEATED_SUBSEQUENCE_H_
