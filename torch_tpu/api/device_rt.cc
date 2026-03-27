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

#include <cstdint>
#include <optional>
#include <string>

#include "absl/time/time.h"
#include "ATen/core/Generator.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep, needed for at::Tensor mapping
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/discovery.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/pjrt/pjrt_init.h"
#include "torch_tpu/pjrt/pjrt_shutdown.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "pybind11/attr.h"
#include "pybind11/chrono.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torch_tpu {

namespace py = pybind11;

namespace {
void SetRngStatePy(at::Tensor state, int device_index) {
  const at::Generator& gen = GetDefaultDeviceGenerator(device_index);
  if (state.dtype() == at::kByte) {
    state = state.view(at::kUInt64);
  }
  if (state.device() != gen.device()) {
    state = state.to(gen.device());
  }
  TT_THROW_IF_ERROR(UpdateRngState(gen, state)).SetPrepend()
      << "failed to set RNG state: ";
}
}  // namespace

PYBIND11_MODULE(_device_ops_backend, m) {
  py::module::import("torch");
  m.doc() =
      "PjRt backend utilities for PyTorch PrivateUse1 integration. "
      "Core ops are registered via C++ TORCH_LIBRARY_IMPL.";

  m.def(
      "_init_runtime",
      [](const std::string& device_type) -> PjRtInitializationResult {
        TT_ASSIGN_OR_THROW(
            PjRtInitializationResult result,
            InitializePjRt(
                {.device_type = device_type,
                 // TODO: what's the right default here?
                 // Single-device, or "all available devices"?
                 // Should distributed mode be opt-in or opt-out?
                 .world_size = GetWorldSizeFromEnvOnce().value_or(1),
                 // TODO(@lukeboyer): Determine what a safe default
                 // is here.
                 .premapped_buffer_size =
                     GetPremappedBufferSizeFromEnvOnce().value_or(0)}),
            _.SetPrepend() << "failed to initialize PjRt: ");
        if (device_type == "tpu" || device_type == "xla_cuda") {
          TT_THROW_IF_ERROR(AddTpuHooks()) << "failed to initialize TpuHooks.";
          RegisterTpuAllocator();
        }
        CompilationCache::Initialize({});
        return result;
      },
      py::arg("device_type") = "tpu",
      "Initializes the PjRt runtime for the specified device type.");

  m.def(
      "_shutdown_runtime", []() { ShutdownPjRt(); },
      "Shuts down the PjRt runtime.");

  m.def(
      "_synchronize",
      [](std::optional<int> device_index) {
        c10::DeviceIndex index;
        if (device_index.has_value()) {
          index = static_cast<c10::DeviceIndex>(*device_index);
        } else {
          index = c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType())
                      ->getDevice()
                      .index();
        }
        SynchronizeStream(index);
      },
      py::arg("device_index") = py::none(),
      py::call_guard<py::gil_scoped_release>(),
      "Blocks until all async d2h copies on the specified device have "
      "completed.");

  py::class_<PjRtInitializationResult>(m, "PjRtInitializationResult")
      .def_readonly("device_count", &PjRtInitializationResult::device_count)
      .def_readonly("device_id", &PjRtInitializationResult::global_device_id);

  m.def(
      "_set_allow_cache",
      [](bool allow) {
        CompilationCache::GetInstance().SetAllowCacheMode(allow);
      },
      py::arg("allow"),
      "If True (default), the compilation cache will be enabled. If False, "
      "the compilation cache will be disabled and every computation graph "
      "will be compiled from scratch (this is useful for debugging and perf "
      "analysis).");
  m.def(
      "_set_cache_only",
      [](bool cache_only) {
        CompilationCache::GetInstance().SetCacheOnlyMode(cache_only);
      },
      py::arg("cache_only"),
      "If True, the compilation cache will only lookup and not compile. Any "
      "cache miss will result in an error.");
  m.def(
      "_get_cache_requests",
      []() { return CompilationCache::GetInstance().GetCacheRequests(); },
      "Get compilation cache requests.");
  m.def(
      "_get_cache_hits",
      []() { return CompilationCache::GetInstance().GetCacheHits(); },
      "Get compilation cache hits.");
  m.def(
      "_get_cache_misses",
      []() { return CompilationCache::GetInstance().GetCacheMisses(); },
      "Get compilation cache misses.");
  m.def(
      "_clear_cache", []() { CompilationCache::GetInstance().EvictAll(); },
      "Evict all existing entries in the compilation cache. The function "
      "waits for all in-flight compilations to complete.");

  m.def(
      "_hbm_usage_summary",
      []() { return CompilationCache::GetInstance().HbmUsageSummary(); },
      "Return the total HBM usage of the compilation cache.");
  m.def(
      "manual_seed", [](uint64_t seed) { SetManualSeed(seed); },
      py::arg("seed"),
      "Manually set the seed for the generator of the current device.");
  m.def(
      "manual_seed_all", [](uint64_t seed) { SetManualSeedAll(seed); },
      py::arg("seed"),
      "Manually set the seed for all generators (one per device).");
  m.def(
      "get_rng_state",
      [](int device_index) -> at::Tensor {
        TT_ASSIGN_OR_THROW(at::Tensor state,
                           GetRngState(GetDefaultDeviceGenerator(device_index)),
                           _.SetPrepend() << "failed to get RNG state: ");
        return state.view(at::kByte);
      },
      py::arg("device_index"), "Get RNG state for the given device index.");
  m.def("set_rng_state", &SetRngStatePy, py::arg("state"),
        py::arg("device_index"), "Set RNG state for the given device index.");

  py::class_<CacheEntryStats>(m, "CacheEntryStats")
      .def_property_readonly(
          "compilation_duration",
          [](const CacheEntryStats& stats) {
            // Pybind doesn't support absl::Duration, so we convert to
            // microseconds.
            return absl::ToChronoMicroseconds(stats.compilation_duration);
          })
      .def_property_readonly("last_read",
                             [](const CacheEntryStats& stats) {
                               // Pybind doesn't support absl::Time, so we
                               // convert to chrono time.
                               return absl::ToChronoTime(stats.last_read);
                             })
      .def_readonly("read_count", &CacheEntryStats::read_count);

  // Suppress the clang-tidy warning that the variable is destroyed immediately
  // after the definition. We are just using the ctor's side effect to register
  // the class with pybind11.
  py::class_<PerfStats::EntryStats, CacheEntryStats>(  // NOLINT
      m, "EntryStats");

  py::class_<PerfStats>(m, "PerfStats")
      .def_readonly("num_cache_reqs", &PerfStats::num_cache_reqs)
      .def_readonly("num_cache_hits", &PerfStats::num_cache_hits)
      .def_readonly("per_entry_stats", &PerfStats::per_entry_stats);

  m.def(
      "_get_cache_stats",
      []() { return CompilationCache::GetInstance().GetCacheStats(); },
      "Get compilation cache statistics.");
}

}  // namespace torch_tpu
