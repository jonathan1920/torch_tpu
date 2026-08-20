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
import functools
import json
from typing import Any


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
    self._request_queues: list[asyncio.Queue[Any]] = [
        asyncio.Queue() for _ in range(num_queues)
    ]
    self._request_available = asyncio.Event()

  async def put_request(self, rank: int, request: Any) -> None:
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

  async def get_first_request(self) -> Any:
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

  async def get_from_ranks(self, ranks: Collection[int]) -> list[Any]:
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
