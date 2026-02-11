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

// Internally we use the latest version of abseil-cpp that has `ABSL_EXPECT_OK`
// and `ABSL_ASSERT_OK` but externally we are using a slightly older version
// that doesn't have it, so in the meantime we will re-implement the macro so
// that it is the same in both versions.
//
// TODO(b/458100122): Remove this file when abseil-cpp version is >= 2025-10-28
// We will also need to replace all instances of `TT_EXPECT_OK` with
// `ABSL_EXPECT_OK` and all instaneces of `TT_ASSERT_OK` with `ABSL_ASSERT_OK`.
#ifndef TORCH_TPU_COMMON_ABSL_TEST_SHIM_H_
#define TORCH_TPU_COMMON_ABSL_TEST_SHIM_H_

#include "gmock/gmock.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep

#define TT_EXPECT_OK(predicate) EXPECT_THAT(predicate, ::absl_testing::IsOk())
#define TT_ASSERT_OK(predicate) ASSERT_THAT(predicate, ::absl_testing::IsOk())

#endif  // TORCH_TPU_COMMON_ABSL_TEST_SHIM_H_
