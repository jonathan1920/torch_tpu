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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "ATen/core/Dict.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/zeros.h"
#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch/library.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/experimental/sparse_dense_matmul/preprocessing_config.pb.h"

// JAX input preprocessing library
// Note: Despite the jax_* prefix in paths and namespaces, the core C++
// preprocessing library is independent of JAX and framework-agnostic.
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"

namespace torch_tpu {
namespace {

namespace sc_preprocessing = jax_sc_embedding;

}  // namespace

// TT_KERNEL is not required because this is a wrapper on the input
// preprocessing from jax_tpu_embedding (which is framework-independent).
TORCH_LIBRARY_FRAGMENT(tpu, m) {
  auto func = [](c10::Dict<std::string, c10::List<at::Tensor>> input_indices,
                 c10::Dict<std::string, c10::List<at::Tensor>> input_offsets,
                 std::string_view stacked_tables_config_proto,
                 int64_t local_device_count, int64_t global_device_count,
                 int64_t num_sc_per_device, bool allow_id_dropping)
      -> c10::Dict<std::string, c10::Dict<std::string, at::Tensor>> {
    torch_tpu::sparse_dense_matmul::StackedTablesConfig config_proto;
    TT_CHECK_THROW(config_proto.ParseFromString(stacked_tables_config_proto),
                   error::kInvalidArgument)
        << "failed to parse StackedTablesConfig proto";

    absl::flat_hash_map<std::string,
                        std::vector<sc_preprocessing::FeatureMetadataInStack>>
        sc_stacked_tables;
    std::vector<std::unique_ptr<sc_preprocessing::AbstractInputBatch>>
        input_batches;
    std::vector<at::Tensor> contiguous_tensors_holder;

    int total_features = 0;
    for (const auto& [table_name, feature_list] : config_proto.tables()) {
      total_features += feature_list.features_size();
    }
    contiguous_tensors_holder.reserve(total_features * 2);
    input_batches.reserve(total_features);

    int global_feat_idx = 0;
    for (const auto& [table_name, feature_list] : config_proto.tables()) {
      const auto& meta_list = feature_list.features();

      auto indices_it = input_indices.find(table_name);
      TT_CHECK_THROW(indices_it != input_indices.end(), error::kInvalidArgument)
          << "input_indices missing table " << table_name;

      auto offsets_it = input_offsets.find(table_name);
      TT_CHECK_THROW(offsets_it != input_offsets.end(), error::kInvalidArgument)
          << "input_offsets missing table " << table_name;

      const auto& indices_list = indices_it->value();
      const auto& offsets_list = offsets_it->value();

      TT_CHECK_THROW(indices_list.size() == meta_list.size(),
                     error::kInvalidArgument)
          << "indices list size mismatch for table " << table_name;
      TT_CHECK_THROW(offsets_list.size() == meta_list.size(),
                     error::kInvalidArgument)
          << "offsets list size mismatch for table " << table_name;

      sc_stacked_tables[table_name].reserve(meta_list.size());

      for (size_t i = 0; i < meta_list.size(); ++i) {
        const auto& f = meta_list[i];
        sc_stacked_tables[table_name].push_back(
            sc_preprocessing::FeatureMetadataInStack(
                f.name(), global_feat_idx++,
                static_cast<int>(f.max_ids_per_partition()),
                static_cast<int>(f.max_unique_ids_per_partition()),
                static_cast<int>(f.row_offset()),
                static_cast<int>(f.col_offset()),
                static_cast<int>(f.col_shift()), f.batch_size(),
                f.suggested_coo_buffer_size_per_device() > 0
                    ? std::make_optional(static_cast<int>(
                          f.suggested_coo_buffer_size_per_device()))
                    : std::nullopt,
                sc_preprocessing::GetRowCombiner(f.combiner())));

        const at::Tensor& indices = indices_list[i];
        const at::Tensor& offsets = offsets_list[i];

        TT_CHECK_THROW(indices.dtype() == at::kInt, error::kInvalidArgument)
            << "indices must be int32";
        TT_CHECK_THROW(offsets.dtype() == at::kInt, error::kInvalidArgument)
            << "offsets must be int32";

        auto contiguous_indices = indices.contiguous();
        auto contiguous_offsets = offsets.contiguous();
        contiguous_tensors_holder.push_back(contiguous_indices);
        contiguous_tensors_holder.push_back(contiguous_offsets);

        absl::Span<const int32_t> val_span(
            contiguous_indices.data_ptr<int32_t>(), contiguous_indices.numel());
        absl::Span<const int32_t> off_span(
            contiguous_offsets.data_ptr<int32_t>(), contiguous_offsets.numel());

        input_batches.push_back(
            std::make_unique<sc_preprocessing::RaggedTensorInputBatch<
                absl::Span<const int32_t>, absl::Span<const int32_t>>>(
                val_span, off_span, table_name));
      }
    }

    c10::Dict<std::string, c10::Dict<std::string, at::Tensor>> outputs;
    sc_preprocessing::OutputCsrArrays sc_output_buffers;

    sc_preprocessing::PreprocessSparseDenseMatmulInputOptions sc_options{
        .local_device_count = static_cast<int>(local_device_count),
        .global_device_count = static_cast<int>(global_device_count),
        .num_sc_per_device = static_cast<int>(num_sc_per_device),
        .sharding_strategy = sc_preprocessing::ShardingStrategy::kMod,
        .allow_id_dropping = allow_id_dropping,
    };
    const int row_pointers_size_per_device =
        sc_options.GetRowPointersSizePerDevice();

    for (const auto& [table_name, feature_list] : config_proto.tables()) {
      const auto& meta_list = feature_list.features();
      if (meta_list.empty()) continue;

      int32_t required_buffer_size =
          meta_list[0].suggested_coo_buffer_size_per_device();
      if (required_buffer_size <= 0) {
        required_buffer_size = 128;  // Default fallback
      }

      int total_row_pointers_size =
          local_device_count * row_pointers_size_per_device;
      int total_coo_buffer_size = local_device_count * required_buffer_size;

      at::Tensor row_pointers = at::zeros({total_row_pointers_size}, at::kInt);
      at::Tensor embedding_ids = at::empty({total_coo_buffer_size}, at::kInt);
      at::Tensor sample_ids = at::empty({total_coo_buffer_size}, at::kInt);
      at::Tensor gains = at::empty({total_coo_buffer_size}, at::kFloat);

      // Wrap PyTorch data tensors DIRECTLY
      Eigen::Map<sc_preprocessing::MatrixXi> row_pointers_map(
          row_pointers.data_ptr<int32_t>(), local_device_count,
          row_pointers_size_per_device);
      Eigen::Map<sc_preprocessing::MatrixXi> embedding_ids_map(
          embedding_ids.data_ptr<int32_t>(), local_device_count,
          required_buffer_size);
      Eigen::Map<sc_preprocessing::MatrixXi> sample_ids_map(
          sample_ids.data_ptr<int32_t>(), local_device_count,
          required_buffer_size);
      Eigen::Map<sc_preprocessing::MatrixXf> gains_map(
          gains.data_ptr<float>(), local_device_count, required_buffer_size);

      sc_output_buffers.lhs_row_pointers.emplace(table_name, row_pointers_map);
      sc_output_buffers.lhs_embedding_ids.emplace(table_name,
                                                  embedding_ids_map);
      sc_output_buffers.lhs_sample_ids.emplace(table_name, sample_ids_map);
      sc_output_buffers.lhs_gains.emplace(table_name, gains_map);

      c10::Dict<std::string, at::Tensor> table_dict;
      table_dict.insert("row_pointers", row_pointers);
      table_dict.insert("embedding_ids", embedding_ids);
      table_dict.insert("sample_ids", sample_ids);
      table_dict.insert("gains", gains);

      outputs.insert(table_name, table_dict);
    }

    auto result = sc_preprocessing::PreprocessSparseDenseMatmulInput(
        absl::MakeSpan(input_batches), sc_stacked_tables, sc_options,
        &sc_output_buffers);

    TT_CHECK_THROW(result.ok(), result.status().code())
        << result.status().message();

    return outputs;
  };

  m.def(
      "preprocess_sparse_dense_matmul_input("
      "    Dict(str, Tensor[]) input_indices, "
      "    Dict(str, Tensor[]) input_offsets, "
      "    str stacked_tables_config_proto, "
      "    int local_device_count, "
      "    int global_device_count, "
      "    int num_sc_per_device, "
      "    bool allow_id_dropping"
      ") -> Dict(str, Dict(str, Tensor))",
      func);
}

}  // namespace torch_tpu
