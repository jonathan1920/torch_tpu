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

#include "torch_tpu/eager/device_buffer_utils.h"

#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

TEST(DeviceBufferUtilsTest,
     ComputeConstantDeviceBufferRefOpParamCacheKeysMatchesGoldenMap) {
  std::vector<char> data = {'a', 'b', 'c', 'd'};
  Dimensions dims = {2, 3};
  mlir::ElementType dtype = mlir::ElementType::F32;

  TF_ASSERT_OK_AND_ASSIGN(
      const OpParamCacheKeys keys,
      internal::ComputeConstantDeviceBufferRefOpParamCacheKeys(data, dims,
                                                               dtype));

  std::map<std::string, FingerprintType> actual_map(
      keys.begin(), keys.end());  // STD_PAIR_OK=test map.

  const std::map<std::string, FingerprintType> expected_map = {
      // STD_PAIR_OK=test map.
      {"data", 2026542488743870450ULL},
      {"dimensions", FingerprintCatLeft("", "2", "3")},
      {"element_type", Fingerprint("f32")},
  };

  EXPECT_EQ(actual_map, expected_map);
}

}  // namespace
}  // namespace torch_tpu
