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

#ifndef TORCH_TPU_PJRT_PJRT_STATE_H_
#define TORCH_TPU_PJRT_PJRT_STATE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "torch_tpu/common/device_type.h"
#include "xla/future.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// PjRt initialization options. Defaults to TPU, no pre-mapped buffer.
struct PjRtInitializationOptions {
  std::string device_type = "tpu";
  int64_t premapped_buffer_size_bytes = 0;
};

// PjrtBackend manages the lifecycle and state of the PjRt runtime.
//
// It provides a singleton interface to initialize and access the PjRt client
// and devices. Initialization is lazy and thread-safe.
class PjrtBackend {
 public:
  // PjrtBackend is neither copyable nor movable.
  PjrtBackend(const PjrtBackend&) = delete;
  PjrtBackend& operator=(const PjrtBackend&) = delete;
  PjrtBackend(PjrtBackend&&) = delete;
  PjrtBackend& operator=(PjrtBackend&&) = delete;

  // Returns the singleton PjrtBackend instance.
  [[nodiscard]] static PjrtBackend& GetInstance();

  // Sets the global PjRt initialization options. These options will be used
  // for lazy initialization when the first device or client is requested.
  void SetPjRtInitializationOptions(const PjRtInitializationOptions& options)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Ensures that the PjRt singleton state is initialized. This method is
  // idempotent and is triggered by any operation that requires the PjRt
  // infrastructure (e.g., GetClient, GetDevice).
  absl::Status EnsureInitialized() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns whether the PjRt runtime is initialized.
  [[nodiscard]] bool IsInitialized() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRtClient singleton. Triggers lazy initialization if needed.
  [[nodiscard]] xla::PjRtClient* absl_nullable GetClient()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRtDevice singleton. Triggers lazy initialization if needed.
  [[nodiscard]] xla::PjRtDevice* absl_nullable GetDevice()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global device count. Triggers lazy initialization if needed.
  absl::StatusOr<int> GetGlobalDeviceCount() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global ID of the current device. Triggers lazy initialization
  // if needed.
  absl::StatusOr<int> GetGlobalDeviceId() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRt device type.
  [[nodiscard]] PjRtDeviceType GetDeviceType() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns allocator stats for the current PjRt device.
  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRt host allocator from the PjRt client.
  absl::StatusOr<xla::HostMemoryAllocator*> GetHostAllocator()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Shuts down the PjRt runtime and resets the global state.
  void Shutdown() ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  friend class absl::NoDestructor<PjrtBackend>;

  PjrtBackend() = default;
  ~PjrtBackend() = default;

  // Internal initialization logic that returns an absl::Status.
  absl::Status InitializeInternal() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::unique_ptr<xla::PjRtClient> absl_nullable client_
      ABSL_GUARDED_BY(mutex_);
  xla::PjRtDevice* absl_nullable device_ ABSL_GUARDED_BY(mutex_) = nullptr;
  PjRtDeviceType device_type_ ABSL_GUARDED_BY(mutex_) =
      PjRtDeviceType::kUnknown;
  int global_device_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int global_device_id_ ABSL_GUARDED_BY(mutex_) = -1;
  PjRtInitializationOptions options_ ABSL_GUARDED_BY(mutex_) = {};
  bool init_attempted_ ABSL_GUARDED_BY(mutex_) = false;
  absl::Status init_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
};

// Updates the tracked future for the given device and stream.
void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      xla::Future<void> future);

// Updates the tracked future for the current stream.
void MarkStreamActive(xla::Future<void> future);

// Blocks until all pending operations on the given device and stream have
// completed.
void SynchronizeStream(c10::DeviceIndex device_index, int64_t stream_id);

// Blocks until all pending operations on ALL streams of the given device have
// completed.
void SynchronizeDevice(c10::DeviceIndex device_index);

// An EventSnapshot is a collection of XLA futures that represents the state of
// a stream at a particular point in time.
class EventSnapshot {
 public:
  ~EventSnapshot();

  // Records an event snapshot for the given device and stream.
  // This is an awaitable and queryable checkpoint; when it is reached, all
  // prior async host-to-device and device-to-host operations on the stream
  // are complete, as well as other futures recorded using MarkStreamActive().
  // TODO(bawilson): also include deferred ops in the snapshot
  static std::shared_ptr<EventSnapshot> Record(c10::DeviceIndex device_index,
                                               c10::StreamId stream_id);

  // Wait for the event snapshot to complete.
  absl::Status Wait() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

 private:
  // The event ID of the event snapshot.
  // Private; must use Record() so that the snapshot is tracked on the stream.
  explicit EventSnapshot(int64_t event_id) : event_id_(event_id) {}

  // The event ID of the event snapshot.
  const int64_t event_id_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_PJRT_PJRT_STATE_H_
