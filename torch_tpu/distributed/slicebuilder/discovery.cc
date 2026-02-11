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

#include "torch_tpu/distributed/slicebuilder/discovery.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "torch/csrc/distributed/c10d/TCPStore.hpp"
#include "torch_tpu/common/unique_file_descriptor.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/environment.h"

namespace torch_tpu {

namespace {

// Simple wrapper around a socket that can be used to listen on a port.
// Necessary for RAII scope to close when this goes out of scope.
struct SimpleSocket {
  UniqueFileDescriptor fd;
  int port_num;

  SimpleSocket(UniqueFileDescriptor f, int p) : fd(std::move(f)), port_num(p) {}

  SimpleSocket(const SimpleSocket&) = delete;
  SimpleSocket& operator=(const SimpleSocket&) = delete;
  SimpleSocket(SimpleSocket&& other) = default;
  SimpleSocket& operator=(SimpleSocket&& other) = default;
};

// Helper function to find a free port.
absl::StatusOr<SimpleSocket> GetFreePort() {
  // Try IPv6 first.
  UniqueFileDescriptor fd(socket(AF_INET6, SOCK_STREAM, 0));
  if (fd.valid()) {
    // IPv6 is supported.
    struct sockaddr_in6 addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_any;
    addr.sin6_port = 0;

    if (bind(fd.get(), reinterpret_cast<struct sockaddr*>(&addr),
             sizeof(addr)) == -1) {
      return TT_ERROR(error::kInternal)
             << "bind(ipv6) failed: " << std::strerror(errno);
    }

    socklen_t len = sizeof(addr);
    if (getsockname(fd.get(), reinterpret_cast<struct sockaddr*>(&addr),
                    &len) == -1) {
      return TT_ERROR(error::kInternal)
             << "getsockname(ipv6) failed: " << std::strerror(errno);
    }

    if (listen(fd.get(), 1) == -1) {
      return TT_ERROR(error::kInternal)
             << "listen(ipv6) failed: " << std::strerror(errno);
    }

    return SimpleSocket(std::move(fd), ntohs(addr.sin6_port));
  }

  // Fallback to IPv4 if IPv6 socket creation failed.
  fd.reset(socket(AF_INET, SOCK_STREAM, 0));
  if (!fd.valid()) {
    return TT_ERROR(error::kInternal)
           << "socket(ipv4) failed: " << std::strerror(errno);
  }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;

  if (bind(fd.get(), reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) ==
      -1) {
    return TT_ERROR(error::kInternal)
           << "bind(ipv4) failed: " << std::strerror(errno);
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd.get(), reinterpret_cast<struct sockaddr*>(&addr), &len) ==
      -1) {
    return TT_ERROR(error::kInternal)
           << "getsockname(ipv4) failed: " << std::strerror(errno);
  }

  if (listen(fd.get(), 1) == -1) {
    return TT_ERROR(error::kInternal)
           << "listen(ipv4) failed: " << std::strerror(errno);
  }

  return SimpleSocket(std::move(fd), ntohs(addr.sin_port));
}

// Helper function to get a required environment variable or return an error.
absl::StatusOr<std::string> GetRequiredEnv(const char* env_name) {
  const char* env_val = getenv(env_name);
  if (env_val == nullptr) {
    return TT_ERROR(error::kFailedPrecondition)
           << env_name << " environment variable is not set.";
  }
  return std::string(env_val);
}

absl::StatusOr<std::pair<std::string, int64_t>> InitializeWorker(
    int rank, int world_size, c10d::TCPStore& store) {
  std::string sb_addrs;
  int64_t sb_port = -1;

  TT_ASSIGN_OR_RETURN(auto sock, GetFreePort());
  sb_port = sock.port_num;

  // TODO(jparkerh): Once we go multi-host, this cannot be localhost anymore.
  std::string my_addr = absl::StrCat("localhost:", sb_port);
  std::string key = absl::StrCat("worker_addr/", rank);
  store.set(key, std::vector<uint8_t>(my_addr.begin(), my_addr.end()));

  std::vector<std::string> keys;
  keys.reserve(world_size);
  for (int i = 0; i < world_size; ++i) {
    keys.push_back(absl::StrCat("worker_addr/", i));
  }

  std::vector<std::vector<uint8_t>> blobs = store.multiGet(keys);
  std::vector<std::string> addrs;
  addrs.reserve(blobs.size());
  for (const auto& blob : blobs) {
    addrs.emplace_back(blob.begin(), blob.end());
  }

  sb_addrs = absl::StrJoin(addrs, ",");

  // Rank 0 must hold the distributed store alive while other ranks are still
  // working with it, hence this extra completion sync.
  if (rank == 0) {
    std::vector<std::string> done_keys;
    done_keys.reserve(world_size - 1);
    for (int i = 1; i < world_size; ++i) {
      done_keys.push_back(absl::StrCat("init_complete/", i));
    }
    store.wait(done_keys);
  } else {
    std::string done_key = absl::StrCat("init_complete/", rank);
    store.set(done_key, std::vector<uint8_t>{});
  }

  return std::make_pair(sb_addrs, sb_port);
}

}  // namespace

absl::StatusOr<std::pair<std::string, int64_t>> GetSlicebuilderMeshConfig(
    int rank, int world_size, std::string master_addr, int master_port) {
  c10d::TCPStoreOptions opts;
  opts.port = master_port;
  opts.isServer = (rank == 0);
  opts.numWorkers = world_size;
  // Setting a generous timeout since this involves multiple workers.
  opts.timeout = std::chrono::milliseconds(30000);
  auto store = std::make_unique<c10d::TCPStore>(master_addr, opts);
  store->waitForWorkers();

  return InitializeWorker(rank, world_size, *store);
}

absl::Status InitializeAsDistributedWorker(
    const DistributedWorkerConfiguration& config) {
  ABSL_LOG(INFO) << "Configuring distributed TPU for rank: " << config.rank
                 << " and world size: " << config.world_size;

  TT_ASSIGN_OR_RETURN(
      (std::pair<std::string, int64_t> mesh_config),
      GetSlicebuilderMeshConfig(config.rank, config.world_size,
                                config.master_addr, config.master_port));

  return InitializeDistributedEnvironment(config.rank, config.world_size,
                                          config.local_rank, mesh_config.first,
                                          mesh_config.second);
}

absl::StatusOr<DistributedWorkerConfiguration>
GetDistributedWorkerConfiguration() {
  int rank = -1;
  TT_ASSIGN_OR_RETURN(std::string env_rank, GetRequiredEnv("RANK"));
  if (!absl::SimpleAtoi(env_rank, &rank)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse RANK: " << env_rank;
  }

  int local_rank = -1;
  TT_ASSIGN_OR_RETURN(std::string env_local_rank, GetRequiredEnv("LOCAL_RANK"));
  if (!absl::SimpleAtoi(env_local_rank, &local_rank)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse LOCAL_RANK: " << env_local_rank;
  }

  // Get the master address and port from the environment variables.
  std::string master_addr;
  int master_port;
  TT_ASSIGN_OR_RETURN(master_addr, GetRequiredEnv("MASTER_ADDR"));
  TT_ASSIGN_OR_RETURN(std::string env_master_port,
                      GetRequiredEnv("MASTER_PORT"));
  if (!absl::SimpleAtoi(env_master_port, &master_port)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse MASTER_PORT: " << env_master_port;
  }

  // Get the world size from the environment variables.
  int world_size;
  TT_ASSIGN_OR_RETURN(std::string env_world_size, GetRequiredEnv("WORLD_SIZE"));
  if (!absl::SimpleAtoi(env_world_size, &world_size)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse WORLD_SIZE: " << env_world_size;
  }

  auto distributed_worker_config = DistributedWorkerConfiguration{
      .rank = rank,
      .local_rank = local_rank,
      .world_size = world_size,
      .master_addr = master_addr,
      .master_port = master_port,
  };
  ABSL_LOG(INFO) << "DistributedWorkerConfiguration: "
                 << distributed_worker_config.rank << " "
                 << distributed_worker_config.local_rank << " "
                 << distributed_worker_config.world_size << " "
                 << distributed_worker_config.master_addr << " "
                 << distributed_worker_config.master_port;

  return distributed_worker_config;
}

}  // namespace torch_tpu
