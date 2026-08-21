# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed handshake protocol implementation using ZMQ and asyncio."""

import asyncio
import collections.abc
from collections.abc import Collection
import dataclasses
import functools
import json
import os
import struct
from typing import Any

from absl import logging
import portpicker
import torch.distributed as dist
import zmq
import zmq.asyncio


def _get_handshake_timeout_ms() -> int:
  """Returns the handshake socket timeout in milliseconds.

  The timeout specifies the maximum duration (in milliseconds) for send and
  receive operations on the handshake sockets.

  The timeout can be overridden using the
  `TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_MS` environment variable. Defaults to
  60000 ms (60 seconds).

  Returns:
    The timeout in milliseconds.

  Raises:
    ValueError: If `TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_MS` cannot be parsed as
      an integer or is negative.
  """
  env_val = os.environ.get("TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_MS", "60000")
  try:
    timeout_ms = int(env_val)
  except ValueError as e:
    raise ValueError(
        "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_MS must be an integer, got"
        f" {env_val!r}."
    ) from e
  if timeout_ms < 0:
    raise ValueError(
        "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_MS must be non-negative, got"
        f" {timeout_ms}."
    )
  return timeout_ms


class ProcessGroupCollectiveCount:
  """Holds collective counts for a process group.

  Attributes:
    collective_count_before: Number of collectives that occurred on the rank so
      far for this process group.
    num_collectives_in_graph: Number of collectives in the FX graph for the
      compiled region for this process group. Must be strictly greater than 0.
  """

  def __init__(
      self,
      collective_count_before: int,
      num_collectives_in_graph: int,
  ) -> None:
    if collective_count_before < 0:
      raise ValueError(
          "collective_count_before must be non-negative, got"
          f" {collective_count_before}."
      )

    if num_collectives_in_graph <= 0:
      raise ValueError(
          "num_collectives_in_graph must be strictly positive, got"
          f" {num_collectives_in_graph}."
      )

    self.collective_count_before = collective_count_before
    self.num_collectives_in_graph = num_collectives_in_graph

  def increment_num_collectives_in_graph(
      self, number_of_collectives: int
  ) -> None:
    """Updates collective counts.

    Adds the current `num_collectives_in_graph` to `collective_count_before`
    and sets `num_collectives_in_graph` to `number_of_collectives`.

    Args:
      number_of_collectives: Number of collectives in the new compiled region.
        Must be greater or equal to 0.

    Raises:
      ValueError: If number_of_collectives < 0.
    """
    if number_of_collectives < 0:
      raise ValueError(
          "number_of_collectives must be greater or equal to 0, got"
          f" {number_of_collectives}."
      )
    self.collective_count_before += self.num_collectives_in_graph
    self.num_collectives_in_graph = number_of_collectives

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ProcessGroupCollectiveCount):
      return False
    return (
        self.collective_count_before == other.collective_count_before
        and self.num_collectives_in_graph == other.num_collectives_in_graph
    )


class ProcessGroupId:
  """Identifier for a process group consisting of sorted, unique ranks.

  Attributes:
    ranks: A tuple of non-negative rank integers representing the process group
      members, sorted in ascending order with no duplicates.
    world_size: Optional world size. If provided, all ranks must be in [0,
      world_size).
  """

  def __init__(
      self,
      ranks: collections.abc.Sequence[int],
      world_size: int | None = None,
  ) -> None:
    if not ranks:
      raise ValueError("ranks cannot be empty.")

    if any(r < 0 for r in ranks):
      raise ValueError(f"All ranks must be non-negative, got {ranks}.")

    if world_size is not None:
      if world_size <= 0:
        raise ValueError(
            f"world_size must be strictly positive, got {world_size}."
        )
      if any(r >= world_size for r in ranks):
        raise ValueError(
            f"All ranks must be in [0, {world_size}), got {ranks}."
        )

    if len(ranks) != len(set(ranks)):
      raise ValueError(f"ranks must not contain duplicates, got {ranks}.")

    if list(ranks) != sorted(ranks):
      raise ValueError(f"ranks must be sorted in ascending order, got {ranks}.")

    self.ranks = tuple(ranks)
    self.world_size = world_size

  def __hash__(self) -> int:
    return hash(self.ranks)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, ProcessGroupId):
      return self.ranks == other.ranks
    return False

  def __iter__(self) -> collections.abc.Iterator[int]:
    return iter(self.ranks)

  def __len__(self) -> int:
    return len(self.ranks)

  def __getitem__(self, index: int) -> int:
    return self.ranks[index]


class RankCollectiveCounts:
  """Holds collective counts on this rank, mapped by process group ranks."""

  def __init__(
      self,
      pg_to_count: (
          dict[ProcessGroupId, ProcessGroupCollectiveCount] | None
      ) = None,
  ) -> None:
    """Initializes RankCollectiveCounts.

    Args:
      pg_to_count: Optional dictionary mapping ProcessGroupId to
        ProcessGroupCollectiveCount. If None, an empty RankCollectiveCounts is
        created.
    """
    self._pg_to_count: dict[ProcessGroupId, ProcessGroupCollectiveCount] = {}
    if pg_to_count is not None:
      for pg, count in pg_to_count.items():
        self._pg_to_count[pg] = count

  def _increment_pg_collective_count(
      self,
      key: ProcessGroupId,
      number_of_collectives: int,
  ) -> None:
    """Increments the collective count for a process group.

    If the process group is already present, its collective counts are updated
    by adding the previous `num_collectives_in_graph` to
    `collective_count_before` and setting `num_collectives_in_graph` to
    `number_of_collectives`. If not present, a new
    `ProcessGroupCollectiveCount` is inserted with `collective_count_before=0`
    and `num_collectives_in_graph=number_of_collectives`.

    Args:
      key: ProcessGroupId identifying the process group.
      number_of_collectives: Number of collectives in the compiled graph for
        this process group. Must be non-negative.
    """
    if key in self._pg_to_count:
      self._pg_to_count[key].increment_num_collectives_in_graph(
          number_of_collectives
      )
    else:
      self._pg_to_count[key] = ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=number_of_collectives,
      )

  def increment_pg_collective_counts(
      self,
      pg_to_num_collectives: dict[ProcessGroupId, int],
  ) -> None:
    """Increments collective counts for multiple process groups.

    For each process group in `pg_to_num_collectives`, updates its collective
    counts with the corresponding number of collectives. For process groups
    that are currently in `_pg_to_count` but not in `pg_to_num_collectives`,
    updates their counts with `number_of_collectives=0`.

    Args:
      pg_to_num_collectives: A mapping of ProcessGroupId to the number of
        collectives for that process group.
    """
    for pg in list(self._pg_to_count.keys()):
      # We have to ensure that we progress the collective count even if the
      # process group is not in the new compiled region.
      if pg not in pg_to_num_collectives:
        self._pg_to_count[pg].increment_num_collectives_in_graph(0)

    # We do this loop to ensure new process groups are added to the
    # `_pg_to_count` map if not already present.
    for pg, num_collectives in pg_to_num_collectives.items():
      self._increment_pg_collective_count(pg, num_collectives)

  def to_json(self) -> list[dict[str, Any]]:
    """Serializes RankCollectiveCounts to a JSON-serializable list of dicts.

    Example:
      >>> counts = RankCollectiveCounts()
      >>> counts.increment_pg_collective_counts({ProcessGroupId([0, 1]): 2})
      >>> counts.to_json()
      [{'process_group': [0, 1], 'collective_count_before': 0,
      'num_collectives_in_graph': 2}]

    Returns:
      A list of dictionaries representing the process group collective counts.
    """
    return [
        {
            "process_group": list(pg.ranks),
            "collective_count_before": count.collective_count_before,
            "num_collectives_in_graph": count.num_collectives_in_graph,
        }
        for pg, count in self._pg_to_count.items()
    ]

  @classmethod
  def from_json(
      cls, data: list[dict[str, Any]] | str
  ) -> "RankCollectiveCounts":
    """Deserializes RankCollectiveCounts from a list of dicts or a JSON string.

    Args:
      data: A JSON string or a list of dictionaries representing the counts.

    Returns:
      A RankCollectiveCounts instance.
    """
    if isinstance(data, str):
      items = json.loads(data)
    else:
      items = data
    counts = cls()
    for item in items:
      pg_id = ProcessGroupId(item["process_group"])
      counts._pg_to_count[pg_id] = ProcessGroupCollectiveCount(
          collective_count_before=item["collective_count_before"],
          num_collectives_in_graph=item["num_collectives_in_graph"],
      )
    return counts

  def items(
      self,
  ) -> collections.abc.ItemsView[ProcessGroupId, ProcessGroupCollectiveCount]:
    """Returns an iterator over (key, value) pairs of _pg_to_count."""
    return self._pg_to_count.items()

  def keys(self) -> collections.abc.KeysView[ProcessGroupId]:
    """Returns an iterator over keys of _pg_to_count."""
    return self._pg_to_count.keys()

  def values(
      self,
  ) -> collections.abc.ValuesView[ProcessGroupCollectiveCount]:
    """Returns an iterator over values of _pg_to_count."""
    return self._pg_to_count.values()

  def __getitem__(
      self,
      key: ProcessGroupId,
  ) -> ProcessGroupCollectiveCount:
    """Gets the collective count for the given process group ID.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      ProcessGroupCollectiveCount for the process group.

    Raises:
      KeyError: If key is not present in _pg_to_count.
    """
    return self._pg_to_count[key]

  def __contains__(self, key: ProcessGroupId) -> bool:
    """Checks if key is in collective counts.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      True if key is present, False otherwise.
    """
    return key in self._pg_to_count

  def __len__(self) -> int:
    return len(self._pg_to_count)

  def __iter__(self) -> collections.abc.Iterator[ProcessGroupId]:
    return iter(self._pg_to_count)

  def __eq__(self, other: object) -> bool:
    """Checks whether two RankCollectiveCounts instances match."""
    if not isinstance(other, RankCollectiveCounts):
      return False
    return self._pg_to_count == other._pg_to_count

  def collective_count(self, key: ProcessGroupId) -> int:
    """Returns total collective count for the given process group.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      The sum of collective_count_before and num_collectives_in_graph for the
      given process group.

    Raises:
      KeyError: If key is not present in _pg_to_count.
    """
    count = self._pg_to_count[key]
    return count.collective_count_before + count.num_collectives_in_graph


class CollectiveHandshakeRequest:
  """Request payload for collective handshake consensus.

  Attributes:
    pg_collective_counts: RankCollectiveCounts tracking collective counts per
      process group of the entire rank.
    executable_fingerprint: Unique identifier for the compiled executable.
    rank: The rank sending this handshake request. Has to be a member of
      participating_ranks.
  """

  def __init__(
      self,
      pg_collective_counts: RankCollectiveCounts,
      executable_fingerprint: str,
      rank: int,
  ) -> None:
    # TODO(b/542976786): We currently Handshake only on global process
    # groups.
    if rank < 0:
      raise ValueError(f"rank must be non-negative, got {rank}.")

    self.pg_collective_counts = pg_collective_counts
    self.executable_fingerprint = executable_fingerprint
    self.rank = rank

    if not self.participating_ranks:
      raise ValueError("participating_ranks cannot be empty.")

    if any(r < 0 for r in self.participating_ranks):
      raise ValueError(
          "All elements in participating_ranks must be non-negative, got"
          f" {self.participating_ranks}."
      )

    if rank not in self.participating_ranks:
      raise ValueError(
          f"rank ({rank}) must be present in participating_ranks"
          f" ({self.participating_ranks})."
      )

  @functools.cached_property
  def participating_ranks(self) -> list[int]:
    """Ranks participating in the handshake.

    This is a sorted list containing unique rank integers derived from the
    process groups in pg_collective_counts.
    """
    return sorted(set().union(*self.pg_collective_counts.keys()))

  def to_bytes(self) -> bytes:
    """Serializes CollectiveHandshakeRequest to JSON UTF-8 encoded bytes.

    Note: The serialized format is not guaranteed to be stable across different
    versions of TorchTPU and should not be persisted or used across different
    TorchTPU versions.

    Returns:
      JSON UTF-8 encoded bytes representing the request.
    """
    return json.dumps({
        "executable_fingerprint": self.executable_fingerprint,
        "participating_ranks": self.participating_ranks,
        "pg_collective_counts": self.pg_collective_counts.to_json(),
        "rank": self.rank,
    }).encode("utf-8")

  @classmethod
  def from_bytes(cls, data: bytes) -> "CollectiveHandshakeRequest":
    """Deserializes CollectiveHandshakeRequest from JSON UTF-8 encoded bytes.

    Args:
      data: JSON UTF-8 encoded bytes representing a CollectiveHandshakeRequest.

    Returns:
      A deserialized CollectiveHandshakeRequest instance.

    Raises:
      UnicodeDecodeError: If data is not valid UTF-8.
      json.JSONDecodeError: If data is not valid JSON.
      KeyError: If required fields are missing from the deserialized JSON
        object.
      ValueError: If deserialized values fail parameter validation in
        __init__.
    """
    obj = json.loads(data.decode("utf-8"))
    return cls(
        pg_collective_counts=RankCollectiveCounts.from_json(
            obj["pg_collective_counts"]
        ),
        executable_fingerprint=obj["executable_fingerprint"],
        rank=obj["rank"],
    )


class CollectiveHandshakeResponse:
  """Response payload for collective handshake consensus.

  Attributes:
    success: Whether the collective handshake succeeded (consensus achieved).
  """

  _TRUE_BYTES = b"\x01"
  _FALSE_BYTES = b"\x00"

  def __init__(self, success: bool) -> None:
    self.success = success

  def to_bytes(self) -> bytes:
    """Serializes CollectiveHandshakeResponse to bytes.

    Note: The serialized format is not guaranteed to be stable across different
    versions of TorchTPU and should not be persisted or used across different
    TorchTPU versions.


    Returns:
      Bytes representing the response.
    """
    return (
        CollectiveHandshakeResponse._TRUE_BYTES
        if self.success
        else CollectiveHandshakeResponse._FALSE_BYTES
    )

  @classmethod
  def from_bytes(cls, data: bytes) -> "CollectiveHandshakeResponse":
    """Deserializes CollectiveHandshakeResponse from bytes.

    Args:
      data: Bytes representing a CollectiveHandshakeResponse.

    Returns:
      A deserialized CollectiveHandshakeResponse instance.

    Raises:
      ValueError: If data is not valid CollectiveHandshakeResponse bytes.
    """
    if data not in [
        CollectiveHandshakeResponse._TRUE_BYTES,
        CollectiveHandshakeResponse._FALSE_BYTES,
    ]:
      raise ValueError(
          f"Invalid CollectiveHandshakeResponse bytes: {data} (expected"
          f" {CollectiveHandshakeResponse._TRUE_BYTES} or"
          f" {CollectiveHandshakeResponse._FALSE_BYTES})."
      )
    return cls(success=(data == CollectiveHandshakeResponse._TRUE_BYTES))


def _select_handshake_port(current_rank: int, coordinator_rank: int) -> int:
  """Selects the port for the Handshake server.

  Port selection strategy:
  1. If the default process group is initialized, the coordinator rank
     dynamically finds a free local port using portpicker. The coordinator rank
     broadcasts the chosen port.
  2. Otherwise, we raise a `RuntimeError`.

  Args:
    current_rank: The rank of the current process.
    coordinator_rank: The rank of the coordinator process.

  Returns:
    The selected port for the Handshake server.

  Raises:
    RuntimeError: If no port can be selected.
  """
  if not dist.is_initialized():
    raise RuntimeError(
        "the default process group is not initialized when a collective"
        " operation was invoked, please initialize the distributed process"
        " group by calling `torch.distributed.dist.init_process_group()`."
    )

  if current_rank == coordinator_rank:
    port = portpicker.pick_unused_port()
    port_list = [port]
  else:
    port_list = [0]
  dist.broadcast_object_list(port_list, src=coordinator_rank)
  return port_list[0]


@dataclasses.dataclass
class _CollectiveHandshakeRequestZMQEnvelope:
  """Envelope containing PyZMQ client routing ID and deserialized request.

  Attributes:
    client_id: Opaque routing identifier bytes automatically assigned by PyZMQ
      ROUTER socket representing the client connection.
    request: The deserialized CollectiveHandshakeRequest instance.
  """

  client_id: bytes
  request: CollectiveHandshakeRequest


class _CollectiveHandshakeConsensus:
  """Stores and manages incoming handshake requests from participating ranks.

  This does not have to include the coordinator rank if it isn't participating
  in the handshake.

  Attributes:
    _request_queues: A list of asyncio queues, one per rank, storing incoming
      handshake requests.
    _request_available: An asyncio event set whenever a new request is put into
      any queue.
  """

  def __init__(self, num_queues: int) -> None:
    """Initializes _CollectiveHandshakeConsensus.

    Args:
      num_queues: Total number of rank queues to initialize. Must be strictly
        positive.

    Raises:
      ValueError: If num_queues is not strictly positive.
    """
    if num_queues <= 0:
      raise ValueError(
          f"num_queues must be strictly positive, got {num_queues}."
      )
    self._request_queues: list[
        asyncio.Queue[_CollectiveHandshakeRequestZMQEnvelope]
    ] = [asyncio.Queue() for _ in range(num_queues)]
    self._request_available = asyncio.Event()

  async def put_request(
      self, rank: int, request: _CollectiveHandshakeRequestZMQEnvelope
  ) -> None:
    """Puts a request into the queue corresponding to the given rank.

    Args:
      rank: The rank index from which the request was received. Must be in range
        [0, num_queues).
      request: The request payload to enqueue.

    Raises:
      ValueError: If rank is out of range [0, num_queues).
    """
    if rank < 0 or rank >= len(self._request_queues):
      raise ValueError(
          f"rank ({rank}) must be in range [0, {len(self._request_queues)})."
      )
    await self._request_queues[rank].put(request)
    self._request_available.set()

  async def get_first_request(self) -> _CollectiveHandshakeRequestZMQEnvelope:
    """Waits for and returns the first available request across all queues.

    Returns:
      The request payload from the first non-empty rank queue.
    """
    # We wait until there is a request available on any of the queues.
    while not any(not q.empty() for q in self._request_queues):
      self._request_available.clear()
      await self._request_available.wait()

    # We find which queue has the request available and get only that request.
    first_rank = next(
        r for r, q in enumerate(self._request_queues) if not q.empty()
    )
    return self._request_queues[first_rank].get_nowait()

  async def get_from_ranks(
      self, ranks: Collection[int]
  ) -> list[_CollectiveHandshakeRequestZMQEnvelope]:
    """Concurrently awaits and returns requests from the specified ranks.

    Args:
      ranks: A collection of unique rank indices to await requests from. Each
        rank must be in the range [0, num_queues).

    Returns:
      A list of request payloads corresponding to each rank in `ranks`, where
      the i-th element in the returned list is the request received from the
      i-th rank in `ranks`.

    Raises:
      ValueError: If any rank in `ranks` is out of range [0, num_queues) or if
        `ranks` contains duplicate values.
    """
    if not ranks:
      return []
    if len(ranks) != len(set(ranks)):
      raise ValueError(f"ranks must not contain duplicates, got {ranks}.")
    for r in ranks:
      if r < 0 or r >= len(self._request_queues):
        raise ValueError(
            f"rank ({r}) in ranks must be in range [0,"
            f" {len(self._request_queues)})."
        )
    return await asyncio.gather(*[self._request_queues[r].get() for r in ranks])


def _validate_port_number(port: int) -> None:
  """Validates that the port number is in the valid 16 bit range.

  Args:
    port: The port number to validate.

  Raises:
    ValueError: If the port number is not in the valid 16 bit range.
  """
  try:
    struct.pack("!H", port)
  except struct.error as e:
    raise ValueError(
        f"port number {port} is not a valid 16 bit unsigned integer."
    ) from e


class _ZMQServer:
  """Handles PyZMQ ROUTER socket network communication for _HandshakeServer.

  Example usage:
    # Coordinator server (async):
    server = _ZMQServer(port=12345)
    envelope = await server.recv_request()
    # Process envelope.request...
    await server.send_reply(
        envelope.client_id, CollectiveHandshakeResponse(success=True)
    )

    # Client (synchronous):
    client = _ZMQClient(port=12345)
    client.send(request)
    response = client.recv()  # CollectiveHandshakeResponse(success=True)
    client.close()
    server.close()

  Attributes:
    _port: The TCP port number the server binds to.
    _context: The zmq.asyncio.Context managing socket lifecycle.
    _socket: The PyZMQ ROUTER socket listening for incoming client requests and
      sending replies.
  """

  def __init__(self, port: int) -> None:
    """Initializes _ZMQServer and binds the ROUTER socket.

    Args:
      port: TCP port number to bind to.
    """
    _validate_port_number(port)
    self._port = port
    self._context = zmq.asyncio.Context()
    self._socket = self._context.socket(zmq.ROUTER)
    # Server should not timeout on recv as we might want it to be idle
    # sometimes.
    self._socket.setsockopt(zmq.SNDTIMEO, _get_handshake_timeout_ms())
    self._socket.bind(f"tcp://*:{self._port}")

  async def recv_request(self) -> _CollectiveHandshakeRequestZMQEnvelope | None:
    """Receives a raw PyZMQ request and returns a _CollectiveHandshakeRequestZMQEnvelope."""
    frames = await self._socket.recv_multipart()
    if len(frames) != 2:
      logging.warning(
          "ZMQServer expected 2 frames [client_id, payload], got %d frames."
          " Frames: [%s]",
          len(frames),
          frames,
      )
      return None
    client_id = frames[0]
    payload = frames[1]
    req = CollectiveHandshakeRequest.from_bytes(payload)
    return _CollectiveHandshakeRequestZMQEnvelope(
        client_id=client_id, request=req
    )

  async def send_reply(
      self, client_id: bytes, response: CollectiveHandshakeResponse
  ) -> None:
    """Sends a PyZMQ multipart reply [client_id, serialized_response] to a client rank."""
    await self._socket.send_multipart([client_id, response.to_bytes()])

  def close(self) -> None:
    """Closes the server socket and terminates the ZMQ context.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops). Calling network operations (`recv_request`, `send_reply`) after
    `close` will raise a `zmq.ZMQError` because the underlying socket is closed.
    """
    if not self._socket.closed:
      self._socket.close(linger=0)
    if not self._context.closed:
      self._context.term()


class _ZMQClient:
  """Handles PyZMQ DEALER socket network communication for _HandshakeClient.

  Attributes:
    _port: TCP port number of the coordinator handshake server.
    _master_addr: Hostname or IP address of the server to connect to. (defaults
      to `MASTER_ADDR` environment variable or 'localhost').
    _context: The zmq.Context managing socket lifecycle.
    _socket: The PyZMQ DEALER socket connected to the coordinator server.
  """

  def __init__(self, port: int) -> None:
    """Initializes _ZMQClient and connects the DEALER socket.

    Args:
      port: TCP port number of the coordinator server.
    """
    _validate_port_number(port)
    self._port = port
    self._master_addr = os.environ.get("MASTER_ADDR", "localhost")

    self._context = zmq.Context()  # pyrefly: ignore[missing-attribute]
    self._socket = self._context.socket(zmq.DEALER)
    timeout_ms = _get_handshake_timeout_ms()
    self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    self._socket.connect(f"tcp://{self._master_addr}:{self._port}")

  def send(self, request: CollectiveHandshakeRequest) -> None:
    """Sends serialized request synchronously to the server via PyZMQ socket."""
    self._socket.send(request.to_bytes())

  def recv(self) -> CollectiveHandshakeResponse:
    """Receives response synchronously from the server via PyZMQ socket."""
    return CollectiveHandshakeResponse.from_bytes(self._socket.recv())

  def close(self) -> None:
    """Closes the client socket and terminates the ZMQ context.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops). Calling network operations (`send`, `recv`) after `close` will
    raise
    a `zmq.ZMQError` because the underlying socket is closed.
    """
    if not self._socket.closed:
      self._socket.close(linger=0)
    if not self._context.closed:
      self._context.term()
