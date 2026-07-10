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

#include "torch_tpu/ops/experimental/sparse_dense_matmul/input_preprocessing_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "torch/library.h"

namespace torch_tpu {

namespace {

inline int32_t FloorMod(int32_t a, int32_t b) {
  int32_t r = a % b;
  return (r < 0) ? r + b : r;
}

inline int32_t FloorDiv(int32_t a, int32_t b) {
  int32_t q = a / b;
  int32_t r = a % b;
  return (r < 0) ? q - 1 : q;
}

inline constexpr int64_t ALIGNMENT_SIZE = 16;

struct CooTuple {
  int32_t sample_id;
  int32_t emb_id;
  int32_t part_id;
};

// Note: TT_KERNEL macro bypass for standalone experimental op.
}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
SparseCoreInputTorch PreprocessToSparseCore(const at::Tensor& values,
                                            const at::Tensor& offsets,
                                            int64_t global_device_count,
                                            int64_t coo_buffer_size_per_device,
                                            int64_t num_sc_per_device,
                                            bool allow_id_dropping) {
  TORCH_CHECK(values.dim() == 1,  // TORCH_CHECK_OK
              "values tensor must be 1D, got dim ", values.dim());
  TORCH_CHECK(values.scalar_type() == at::kInt,  // TORCH_CHECK_OK
              "values tensor must be int32_t");
  TORCH_CHECK(offsets.dim() == 1,  // TORCH_CHECK_OK
              "offsets tensor must be 1D, got dim ", offsets.dim());
  TORCH_CHECK(offsets.size(0) >= 1,  // TORCH_CHECK_OK
              "offsets tensor must have at least 1 element");
  TORCH_CHECK(num_sc_per_device > 0,  // TORCH_CHECK_OK
              "num_sc_per_device must be positive");
  const int64_t num_rows = offsets.size(0) - 1;
  const int64_t num_scs = global_device_count * num_sc_per_device;
  const int64_t rows_per_sc =
      (num_rows + num_sc_per_device - 1) / num_sc_per_device;
  const int64_t bucket_size = std::max(num_scs, ALIGNMENT_SIZE);

  auto contiguous_offsets = offsets.contiguous();
  auto get_offset = [&](int64_t idx) -> int64_t {
    if (contiguous_offsets.scalar_type() == at::kLong) {
      return contiguous_offsets.data_ptr<int64_t>()[idx];
    }
    return contiguous_offsets.data_ptr<int32_t>()[idx];
  };

  if (coo_buffer_size_per_device <= 0) {
    int64_t max_raw_per_sc = 0;
    for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
      const int64_t rs = std::min(sc * rows_per_sc, num_rows);
      const int64_t re = std::min(rs + rows_per_sc, num_rows);
      max_raw_per_sc =
          std::max(max_raw_per_sc, get_offset(re) - get_offset(rs));
    }
    const int64_t aligned_per_part =
        ((max_raw_per_sc + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) *
        ALIGNMENT_SIZE;
    coo_buffer_size_per_device =
        aligned_per_part * bucket_size * num_sc_per_device;
  }

  auto row_pointers =
      at::zeros({num_sc_per_device * bucket_size}, values.options());
  auto embedding_ids =
      at::full({coo_buffer_size_per_device},
               std::numeric_limits<int32_t>::max(), values.options());
  auto sample_ids =
      at::full({coo_buffer_size_per_device},
               std::numeric_limits<int32_t>::max(), values.options());
  auto gains = at::full({coo_buffer_size_per_device},
                        std::numeric_limits<float>::quiet_NaN(),
                        values.options().dtype(at::kFloat));

  auto contiguous_values = values.contiguous();
  const int32_t* val_ptr = contiguous_values.data_ptr<int32_t>();
  int32_t* row_ptr = row_pointers.data_ptr<int32_t>();
  int32_t* emb_ptr = embedding_ids.data_ptr<int32_t>();
  int32_t* sam_ptr = sample_ids.data_ptr<int32_t>();
  float* gain_ptr = gains.data_ptr<float>();

  std::vector<CooTuple> coo_entries;
  coo_entries.reserve(values.numel());
  int64_t coo_idx = 0;

  for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
    coo_entries.clear();
    const int64_t sc_base_offset = coo_idx;
    const int64_t row_start = sc * rows_per_sc;
    const int64_t row_end = std::min(row_start + rows_per_sc, num_rows);
    coo_entries.reserve(get_offset(row_end) - get_offset(row_start));

    for (int64_t r = row_start; r < row_end; ++r) {
      const int32_t local_sam_id = static_cast<int32_t>(r - row_start);
      const int64_t start_idx = get_offset(r);
      const int64_t end_idx = get_offset(r + 1);
      for (int64_t i = start_idx; i < end_idx; ++i) {
        const int32_t val = val_ptr[i];
        coo_entries.push_back(
            {local_sam_id, val, FloorMod(val, static_cast<int32_t>(num_scs))});
      }
    }

    std::sort(coo_entries.begin(), coo_entries.end(),
              [](const CooTuple& a, const CooTuple& b) {
                if (a.part_id != b.part_id) {
                  return a.part_id < b.part_id;
                }
                return a.sample_id < b.sample_id;
              });

    int64_t cur_part = 0;
    const int64_t lhs_row_begin = sc * bucket_size;

    auto set_row_pointer = [&](int64_t part) {
      if (lhs_row_begin + part < num_sc_per_device * bucket_size) {
        row_ptr[lhs_row_begin + part] =
            static_cast<int32_t>(coo_idx - sc_base_offset);
      }
    };

    auto pad_to_alignment = [&]() {
      while (coo_idx % ALIGNMENT_SIZE != 0) {
        if (coo_idx < coo_buffer_size_per_device) {
          emb_ptr[coo_idx] = std::numeric_limits<int32_t>::max();
          sam_ptr[coo_idx] = std::numeric_limits<int32_t>::max();
          gain_ptr[coo_idx] = std::numeric_limits<float>::quiet_NaN();
          coo_idx++;
        } else {
          break;
        }
      }
    };

    auto advance_partition_to = [&](int64_t target_part) {
      while (cur_part < target_part) {
        set_row_pointer(cur_part);
        pad_to_alignment();
        cur_part++;
      }
    };

    int64_t idx = 0;
    while (idx < static_cast<int64_t>(coo_entries.size())) {
      const auto& item = coo_entries[idx];
      advance_partition_to(item.part_id);

      if (coo_idx < coo_buffer_size_per_device) {
        emb_ptr[coo_idx] = FloorDiv(item.emb_id, static_cast<int32_t>(num_scs));
        sam_ptr[coo_idx] = item.sample_id;
        gain_ptr[coo_idx] = 1.0f;
        coo_idx++;
      } else {
        TORCH_CHECK(allow_id_dropping,  // TORCH_CHECK_OK: capacity check
                    "SparseCore buffer overflow: COO index exceeded capacity "
                    "limit and allow_id_dropping is false.");
      }
      idx++;
    }

    advance_partition_to(num_scs);

    for (int64_t p = cur_part; p < bucket_size; ++p) {
      set_row_pointer(p);
    }
  }

  return {row_pointers, embedding_ids, sample_ids, gains};
}

TORCH_LIBRARY_FRAGMENT(tpu, m) {
  m.def(
      "preprocess_sparse_dense_matmul_input(Tensor values, Tensor offsets, "
      "int global_device_count, int coo_buffer_size_per_device, "
      "int num_sc_per_device=2, bool allow_id_dropping=True) -> "
      "(Tensor, Tensor, Tensor, Tensor)",
      [](const at::Tensor& values, const at::Tensor& offsets,
         int64_t global_device_count, int64_t coo_buffer_size_per_device,
         int64_t num_sc_per_device, bool allow_id_dropping) {
        auto res = PreprocessToSparseCore(values, offsets, global_device_count,
                                          coo_buffer_size_per_device,
                                          num_sc_per_device, allow_id_dropping);
        return std::make_tuple(res.row_pointers, res.embedding_ids,
                               res.sample_ids, res.gains);
      });
}

}  // namespace torch_tpu
