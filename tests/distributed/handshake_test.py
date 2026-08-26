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

"""Tests for distributed handshake protocol using PyZMQ and asyncio."""

import asyncio
import json
import os
import threading
import time
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import handshake
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils
import zmq

_LoopRunner = handshake._LoopRunner

_CollectiveHandshakeConsensus = handshake._CollectiveHandshakeConsensus
_CollectiveHandshakeRequestZMQEnvelope = (
    handshake._CollectiveHandshakeRequestZMQEnvelope
)
_ZMQClient = handshake._ZMQClient
_ZMQServer = handshake._ZMQServer
_HandshakeClient = handshake._HandshakeClient
_HandshakeServer = handshake._HandshakeServer
Handshake = handshake.Handshake
CollectiveHandshakeRequest = handshake.CollectiveHandshakeRequest
CollectiveHandshakeResponse = handshake.CollectiveHandshakeResponse
ProcessGroupId = handshake.ProcessGroupId
ProcessGroupCollectiveCount = handshake.ProcessGroupCollectiveCount
_get_handshake_timeout_s = handshake._get_handshake_timeout_s
RankCollectiveCounts = handshake.RankCollectiveCounts
should_handshake = handshake.should_handshake


def _make_request(
    rank: int,
    participating_ranks: list[int] | None = None,
    fingerprint: str = "fp",
    num_collectives_in_graph: int = 2,
    collective_count_before: int = 0,
) -> CollectiveHandshakeRequest:
  if participating_ranks is None:
    participating_ranks = [rank]
  pg_counts = RankCollectiveCounts({
      ProcessGroupId(participating_ranks): ProcessGroupCollectiveCount(
          collective_count_before=collective_count_before,
          num_collectives_in_graph=num_collectives_in_graph,
      )
  })
  return CollectiveHandshakeRequest(
      pg_collective_counts=pg_counts,
      executable_fingerprint=fingerprint,
      rank=rank,
  )


class ProcessGroupCollectiveCountTest(seed_test_utils.RepeatableTest):
  """Unit tests for ProcessGroupCollectiveCount."""

  def test_valid_counts(self) -> None:
    count = ProcessGroupCollectiveCount(
        collective_count_before=0,
        num_collectives_in_graph=2,
    )
    self.assertEqual(count.collective_count_before, 0)
    self.assertEqual(count.num_collectives_in_graph, 2)

  def test_invalid_collective_count_before(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "collective_count_before must be non-negative"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=-1,
          num_collectives_in_graph=1,
      )

  def test_invalid_num_collectives_in_graph(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "num_collectives_in_graph must be strictly positive"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=0,
      )
    with self.assertRaisesRegex(
        ValueError, "num_collectives_in_graph must be strictly positive"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=-2,
      )

  def test_increment_num_collectives_in_graph(self) -> None:
    count = ProcessGroupCollectiveCount(
        collective_count_before=0,
        num_collectives_in_graph=2,
    )
    count.increment_num_collectives_in_graph(3)
    self.assertEqual(count.collective_count_before, 2)
    self.assertEqual(count.num_collectives_in_graph, 3)

    count.increment_num_collectives_in_graph(0)
    self.assertEqual(count.collective_count_before, 5)
    self.assertEqual(count.num_collectives_in_graph, 0)

  def test_increment_num_collectives_in_graph_invalid(self) -> None:
    count = ProcessGroupCollectiveCount(
        collective_count_before=0,
        num_collectives_in_graph=2,
    )
    with self.assertRaisesRegex(
        ValueError, "number_of_collectives must be greater or equal to 0"
    ):
      count.increment_num_collectives_in_graph(-1)


class ProcessGroupIdTest(seed_test_utils.RepeatableTest):
  """Unit tests for ProcessGroupId."""

  def test_valid_process_group_id(self) -> None:
    pg = ProcessGroupId([0, 1, 2])
    self.assertEqual(pg.ranks, (0, 1, 2))
    self.assertIsNone(pg.world_size)

  def test_valid_with_world_size(self) -> None:
    pg = ProcessGroupId([0, 2, 3], world_size=4)
    self.assertEqual(pg.ranks, (0, 2, 3))
    self.assertEqual(pg.world_size, 4)

  def test_empty_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, "ranks cannot be empty"):
      ProcessGroupId([])

  def test_negative_rank_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, "All ranks must be non-negative"):
      ProcessGroupId([-1, 0])

  def test_invalid_world_size_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "world_size must be strictly positive"
    ):
      ProcessGroupId([0, 1], world_size=0)
    with self.assertRaisesRegex(
        ValueError, "world_size must be strictly positive"
    ):
      ProcessGroupId([0, 1], world_size=-2)

  def test_rank_out_of_world_size_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, r"All ranks must be in \[0, 4\)"):
      ProcessGroupId([0, 4], world_size=4)
    with self.assertRaisesRegex(ValueError, r"All ranks must be in \[0, 2\)"):
      ProcessGroupId([0, 2], world_size=2)

  def test_duplicate_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "ranks must not contain duplicates"
    ):
      ProcessGroupId([0, 1, 1])

  def test_unsorted_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "ranks must be sorted in ascending order"
    ):
      ProcessGroupId([1, 0])
    with self.assertRaisesRegex(
        ValueError, "ranks must be sorted in ascending order"
    ):
      ProcessGroupId([0, 3, 2])

  def test_hash_and_equality(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1])
    pg3 = ProcessGroupId([0, 2])
    self.assertEqual(pg1, pg2)
    self.assertNotEqual(pg1, pg3)
    self.assertEqual(hash(pg1), hash(pg2))
    self.assertEqual({pg1: "val"}[pg2], "val")

  def test_iteration_len_indexing(self) -> None:
    pg = ProcessGroupId([0, 1, 3])
    self.assertLen(pg, 3)
    self.assertEqual(list(pg), [0, 1, 3])
    self.assertEqual(pg[0], 0)
    self.assertEqual(pg[1], 1)
    self.assertEqual(pg[2], 3)


class RankCollectiveCountsTest(seed_test_utils.RepeatableTest):
  """Unit tests for RankCollectiveCounts."""

  def test_increment_pg_collective_counts_new_and_existing(self) -> None:
    counts = RankCollectiveCounts()
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1, 2])

    # Initial insertion of multiple process groups
    counts.increment_pg_collective_counts({pg1: 2, pg2: 3})
    self.assertEqual(counts[pg1].collective_count_before, 0)
    self.assertEqual(counts[pg1].num_collectives_in_graph, 2)
    self.assertEqual(counts[pg2].collective_count_before, 0)
    self.assertEqual(counts[pg2].num_collectives_in_graph, 3)
    self.assertIn(pg1, counts)
    self.assertIn(pg2, counts)
    self.assertLen(counts, 2)

    # Next region only includes pg1: pg2 should be incremented with 0
    counts.increment_pg_collective_counts({pg1: 4})
    self.assertEqual(counts[pg1].collective_count_before, 2)
    self.assertEqual(counts[pg1].num_collectives_in_graph, 4)
    self.assertEqual(counts[pg2].collective_count_before, 3)
    self.assertEqual(counts[pg2].num_collectives_in_graph, 0)

    # Next region only includes pg2: pg1 should be incremented with 0
    counts.increment_pg_collective_counts({pg2: 1})
    self.assertEqual(counts[pg1].collective_count_before, 6)
    self.assertEqual(counts[pg1].num_collectives_in_graph, 0)
    self.assertEqual(counts[pg2].collective_count_before, 3)
    self.assertEqual(counts[pg2].num_collectives_in_graph, 1)

  def test_init_with_dict(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1, 2])
    count1 = ProcessGroupCollectiveCount(
        collective_count_before=0, num_collectives_in_graph=1
    )
    count2 = ProcessGroupCollectiveCount(
        collective_count_before=2, num_collectives_in_graph=3
    )
    counts = RankCollectiveCounts({
        pg1: count1,
        pg2: count2,
    })
    self.assertIs(counts[pg1], count1)
    self.assertIs(counts[pg2], count2)

  def test_items_keys_values(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1, 2])
    counts = RankCollectiveCounts()
    counts.increment_pg_collective_counts({pg1: 1, pg2: 3})

    self.assertEqual(list(counts.keys()), [pg1, pg2])
    self.assertEqual(
        [c.num_collectives_in_graph for c in counts.values()], [1, 3]
    )
    self.assertEqual(
        [(k, v.num_collectives_in_graph) for k, v in counts.items()],
        [(pg1, 1), (pg2, 3)],
    )

  def test_to_json_and_from_json(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1, 2])
    counts = RankCollectiveCounts()
    counts.increment_pg_collective_counts({pg1: 2, pg2: 1})

    json_data = counts.to_json()
    self.assertEqual(
        json_data,
        [
            {
                "process_group": [0, 1],
                "collective_count_before": 0,
                "num_collectives_in_graph": 2,
            },
            {
                "process_group": [0, 1, 2],
                "collective_count_before": 0,
                "num_collectives_in_graph": 1,
            },
        ],
    )

    deserialized = RankCollectiveCounts.from_json(json_data)
    self.assertLen(deserialized, 2)
    self.assertEqual(deserialized[pg1].collective_count_before, 0)
    self.assertEqual(deserialized[pg1].num_collectives_in_graph, 2)
    self.assertEqual(deserialized[pg2].collective_count_before, 0)
    self.assertEqual(deserialized[pg2].num_collectives_in_graph, 1)

  def test_from_json_with_json_string(self) -> None:
    json_str = (
        '[{"process_group": [0, 1], "collective_count_before": 2,'
        ' "num_collectives_in_graph": 4}]'
    )
    deserialized = RankCollectiveCounts.from_json(json_str)
    self.assertLen(deserialized, 1)
    pg = ProcessGroupId([0, 1])
    self.assertEqual(deserialized[pg].collective_count_before, 2)
    self.assertEqual(deserialized[pg].num_collectives_in_graph, 4)

  def test_missing_key_raises_key_error(self) -> None:
    counts = RankCollectiveCounts()
    pg = ProcessGroupId([0, 1])
    with self.assertRaises(KeyError):
      _ = counts[pg]

  def test_eq(self) -> None:
    counts1 = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=2
        ),
    })
    counts2 = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=2
        ),
    })
    self.assertEqual(counts1, counts2)

    # Different before count
    counts3 = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=1, num_collectives_in_graph=2
        ),
    })
    self.assertNotEqual(counts1, counts3)

    # Different in graph count
    counts4 = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=3
        ),
    })
    self.assertNotEqual(counts1, counts4)

    # Different process group key
    counts5 = RankCollectiveCounts({
        ProcessGroupId([0, 2]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=2
        ),
    })
    self.assertNotEqual(counts1, counts5)

  def test_collective_count_valid(self) -> None:
    pg = ProcessGroupId([0, 1, 2, 3])
    counts = RankCollectiveCounts({
        pg: ProcessGroupCollectiveCount(
            collective_count_before=3, num_collectives_in_graph=2
        ),
    })
    self.assertEqual(counts.collective_count(pg), 5)

  def test_collective_count_multiple_and_non_global_pgs(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1, 2, 3])
    counts = RankCollectiveCounts({
        pg1: ProcessGroupCollectiveCount(
            collective_count_before=1, num_collectives_in_graph=4
        ),
        pg2: ProcessGroupCollectiveCount(
            collective_count_before=2, num_collectives_in_graph=1
        ),
    })
    self.assertEqual(counts.collective_count(pg1), 5)
    self.assertEqual(counts.collective_count(pg2), 3)

  def test_collective_count_empty_raises(self) -> None:
    counts = RankCollectiveCounts()
    with self.assertRaises(KeyError):
      counts.collective_count(ProcessGroupId([0, 1]))

  def test_collective_count_missing_key_raises(self) -> None:
    counts = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=1
        ),
    })
    with self.assertRaises(KeyError):
      counts.collective_count(ProcessGroupId([0, 1, 2, 3]))


class ShouldHandshakeTest(seed_test_utils.RepeatableTest):
  """Unit tests for should_handshake function."""

  def setUp(self) -> None:
    super().setUp()
    self.world_size = 4
    self.enter_context(
        mock.patch.object(dist, "get_world_size", return_value=self.world_size)
    )

  def test_empty_dict(self) -> None:
    self.assertFalse(should_handshake({}))

  def test_global_process_group_returns_true(self) -> None:
    global_pg = ProcessGroupId(range(4), self.world_size)
    self.assertTrue(should_handshake({global_pg: 1}))
    self.assertTrue(should_handshake({global_pg: 3}))

  def test_sub_process_group_returns_false(self) -> None:
    sub_pg = ProcessGroupId((0,), self.world_size)
    self.assertFalse(should_handshake({sub_pg: 1}))

  def test_multiple_process_groups_returns_false(self) -> None:
    global_pg = ProcessGroupId(range(4), self.world_size)
    sub_pg = ProcessGroupId((0,), self.world_size)
    self.assertFalse(should_handshake({global_pg: 1, sub_pg: 1}))


class CollectiveHandshakeRequestTest(seed_test_utils.RepeatableTest):
  """Unit tests for CollectiveHandshakeRequest class and serialization."""

  def test_to_bytes_and_from_bytes(self) -> None:
    pg_counts = RankCollectiveCounts()
    pg_counts.increment_pg_collective_counts({
        ProcessGroupId([0, 1, 2, 3]): 2,
        ProcessGroupId([0, 1]): 3,
    })

    request = CollectiveHandshakeRequest(
        pg_collective_counts=pg_counts,
        executable_fingerprint="fp_12345",
        rank=0,
    )
    serialized = request.to_bytes()
    self.assertIsInstance(serialized, bytes)

    deserialized = CollectiveHandshakeRequest.from_bytes(serialized)
    self.assertEqual(deserialized.executable_fingerprint, "fp_12345")
    self.assertEqual(deserialized.participating_ranks, [0, 1, 2, 3])
    self.assertEqual(deserialized.rank, 0)
    pg_0123 = ProcessGroupId([0, 1, 2, 3])
    pg_01 = ProcessGroupId([0, 1])
    self.assertEqual(
        deserialized.pg_collective_counts[pg_0123].collective_count_before,
        0,
    )
    self.assertEqual(
        deserialized.pg_collective_counts[pg_0123].num_collectives_in_graph,
        2,
    )
    self.assertEqual(
        deserialized.pg_collective_counts[pg_01].collective_count_before,
        0,
    )
    self.assertEqual(
        deserialized.pg_collective_counts[pg_01].num_collectives_in_graph,
        3,
    )

  def test_participating_ranks_derived_as_cached_property(self) -> None:
    pg_counts = RankCollectiveCounts()
    pg_counts.increment_pg_collective_counts({
        ProcessGroupId([0, 2]): 1,
        ProcessGroupId([1, 2]): 2,
    })

    request = CollectiveHandshakeRequest(
        pg_collective_counts=pg_counts,
        executable_fingerprint="fp",
        rank=1,
    )
    self.assertEqual(request.participating_ranks, [0, 1, 2])

  def test_invalid_participating_ranks_empty(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "participating_ranks cannot be empty"
    ):
      CollectiveHandshakeRequest(
          pg_collective_counts=RankCollectiveCounts(),
          executable_fingerprint="fp",
          rank=0,
      )

  def test_invalid_rank(self) -> None:
    pg_counts = RankCollectiveCounts()
    pg_counts.increment_pg_collective_counts({
        ProcessGroupId([0, 1, 2, 3]): 1,
    })
    with self.assertRaisesRegex(
        ValueError, r"rank \(4\) must be present in participating_ranks"
    ):
      CollectiveHandshakeRequest(
          pg_collective_counts=pg_counts,
          executable_fingerprint="fp",
          rank=4,
      )
    with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
      CollectiveHandshakeRequest(
          pg_collective_counts=pg_counts,
          executable_fingerprint="fp",
          rank=-1,
      )

  def test_from_bytes_invalid_payloads(self) -> None:
    with self.assertRaises(json.JSONDecodeError):
      CollectiveHandshakeRequest.from_bytes(b"invalid json")
    with self.assertRaises(UnicodeDecodeError):
      CollectiveHandshakeRequest.from_bytes(b"\xff\xfe")
    with self.assertRaises(KeyError):
      CollectiveHandshakeRequest.from_bytes(b"{}")


class CollectiveHandshakeResponseTest(seed_test_utils.RepeatableTest):
  """Unit tests for CollectiveHandshakeResponse class and serialization."""

  def test_valid_response(self) -> None:
    resp_true = CollectiveHandshakeResponse(success=True)
    self.assertTrue(resp_true.success)
    resp_false = CollectiveHandshakeResponse(success=False)
    self.assertFalse(resp_false.success)

  def test_to_bytes_and_from_bytes(self) -> None:
    resp = CollectiveHandshakeResponse(success=True)
    serialized = resp.to_bytes()
    self.assertEqual(serialized, b"\x01")
    deserialized = CollectiveHandshakeResponse.from_bytes(serialized)
    self.assertTrue(deserialized.success)

    resp_f = CollectiveHandshakeResponse(success=False)
    serialized_f = resp_f.to_bytes()
    self.assertEqual(serialized_f, b"\x00")
    deserialized_f = CollectiveHandshakeResponse.from_bytes(serialized_f)
    self.assertFalse(deserialized_f.success)

  def test_from_bytes_invalid_payloads(self) -> None:
    with self.assertRaises(ValueError):
      CollectiveHandshakeResponse.from_bytes(b"invalid")


class GetHandshakeTimeoutMsTest(seed_test_utils.RepeatableTest):
  """Unit tests for _get_handshake_timeout_s environment variable parser."""

  def test_default_timeout(self) -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(_get_handshake_timeout_s(), 60)

  def test_custom_valid_timeout(self) -> None:
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S": "5"}
    ):
      self.assertEqual(_get_handshake_timeout_s(), 5)

  def test_invalid_string_timeout(self) -> None:
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S": "invalid"}
    ):
      with self.assertRaisesRegex(
          ValueError,
          "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be an integer",
      ):
        _get_handshake_timeout_s()

  def test_negative_timeout(self) -> None:
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S": "-100"}
    ):
      with self.assertRaisesRegex(
          ValueError,
          "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be non-negative",
      ):
        _get_handshake_timeout_s()


class CollectiveHandshakeConsensusTest(seed_test_utils.RepeatableTest):
  """Unit tests for _CollectiveHandshakeConsensus request queuing and gathering."""

  def test_invalid_num_queues(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "num_queues must be strictly positive"
    ):
      _CollectiveHandshakeConsensus(num_queues=0)
    with self.assertRaisesRegex(
        ValueError, "num_queues must be strictly positive"
    ):
      _CollectiveHandshakeConsensus(num_queues=-1)

  def test_invalid_put_request_rank(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      msg = _make_request(rank=0)
      env = _CollectiveHandshakeRequestZMQEnvelope(client_id=b"c", request=msg)
      with self.assertRaisesRegex(ValueError, r"rank \(-1\) must be in range"):
        await consensus.put_request(-1, env)
      with self.assertRaisesRegex(ValueError, r"rank \(4\) must be in range"):
        await consensus.put_request(4, env)

    asyncio.run(_test())

  def test_invalid_get_from_ranks(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      with self.assertRaisesRegex(
          ValueError, "ranks must not contain duplicates"
      ):
        await consensus.get_from_ranks([0, 1, 0])
      with self.assertRaisesRegex(
          ValueError, r"rank \(-1\) in ranks must be in range"
      ):
        await consensus.get_from_ranks([-1, 0])
      with self.assertRaisesRegex(
          ValueError, r"rank \(4\) in ranks must be in range"
      ):
        await consensus.get_from_ranks([0, 4])

    asyncio.run(_test())

  def test_put_and_get_first_request(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      msg = _make_request(rank=2)
      envelope = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"client_2", request=msg
      )
      await consensus.put_request(2, envelope)
      result = await consensus.get_first_request()
      self.assertEqual(result, envelope)

    asyncio.run(_test())

  def test_get_from_ranks(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      msg1 = _make_request(rank=1, fingerprint="fp1")
      msg3 = _make_request(rank=3, fingerprint="fp3")
      env1 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c1", request=msg1
      )
      env3 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c3", request=msg3
      )

      await consensus.put_request(1, env1)
      await consensus.put_request(3, env3)

      results = await consensus.get_from_ranks([1, 3])
      self.assertEqual(results, [env1, env3])

    asyncio.run(_test())

  def test_get_from_ranks_order(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      env0 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c0", request=_make_request(rank=0)
      )
      env1 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c1", request=_make_request(rank=1)
      )
      env2 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c2", request=_make_request(rank=2)
      )
      env3 = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c3", request=_make_request(rank=3)
      )
      await consensus.put_request(0, env0)
      await consensus.put_request(1, env1)
      await consensus.put_request(2, env2)
      await consensus.put_request(3, env3)

      results = await consensus.get_from_ranks([3, 1, 0])
      self.assertEqual(results, [env3, env1, env0])

    asyncio.run(_test())

  def test_get_from_ranks_empty(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      results = await consensus.get_from_ranks([])
      self.assertEqual(results, [])

    asyncio.run(_test())

  def test_concurrent_put_and_get(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      msg = _make_request(rank=1, fingerprint="fp_delayed")
      env = _CollectiveHandshakeRequestZMQEnvelope(
          client_id=b"c_delayed", request=msg
      )

      async def _delayed_put() -> None:
        await asyncio.sleep(0.01)
        await consensus.put_request(1, env)

      task = asyncio.create_task(_delayed_put())
      result = await consensus.get_first_request()
      self.assertEqual(result, env)
      await task

    asyncio.run(_test())


class SelectHandshakePortTest(seed_test_utils.RepeatableTest):
  """Unit tests for _select_handshake_port."""

  def test_select_handshake_port_exception(self) -> None:
    """Tests that RuntimeError is raised when process group is not initialized."""
    with mock.patch.object(dist, "is_initialized", return_value=False):
      with self.assertRaisesRegex(
          RuntimeError, "the default process group is not initialized"
      ):
        handshake._select_handshake_port(current_rank=0, coordinator_rank=0)

  @mock.patch("portpicker.pick_unused_port", return_value=12345)
  @mock.patch.object(dist, "broadcast_object_list")
  @mock.patch.object(dist, "is_initialized", return_value=True)
  def test_select_handshake_port_coordinator(
      self, mock_init: Any, mock_broadcast: Any, mock_pick: Any
  ) -> None:
    del mock_init  # Unused.
    port = handshake._select_handshake_port(current_rank=0, coordinator_rank=0)
    self.assertEqual(port, 12345)
    mock_pick.assert_called_once()
    mock_broadcast.assert_called_once_with([12345], src=0)

  @mock.patch.object(dist, "is_initialized", return_value=True)
  def test_select_handshake_port_worker(self, mock_init: Any) -> None:
    del mock_init  # Unused.

    def fake_broadcast(obj_list: list[Any], src: int = 0) -> None:
      del src  # Unused.
      obj_list[0] = 54321

    with mock.patch.object(
        dist, "broadcast_object_list", side_effect=fake_broadcast
    ):
      port = handshake._select_handshake_port(
          current_rank=1, coordinator_rank=0
      )
      self.assertEqual(port, 54321)


def _msg_eq(
    msg1: CollectiveHandshakeRequest, msg2: CollectiveHandshakeRequest
) -> bool:
  return (
      msg1.executable_fingerprint == msg2.executable_fingerprint
      and msg1.participating_ranks == msg2.participating_ranks
      and msg1.rank == msg2.rank
      and list(msg1.pg_collective_counts.keys())
      == list(msg2.pg_collective_counts.keys())
  )


class ZMQServerClientTest(seed_test_utils.RepeatableTest):
  """Unit tests for _ZMQServer and _ZMQClient communication."""

  def test_zmq_server_invalid_port_number(self) -> None:
    with self.assertRaises(ValueError):
      _ZMQServer(port=-1)

    with self.assertRaises(ValueError):
      _ZMQServer(port=65536)

  def test_zmq_client_invalid_port_number(self) -> None:
    with self.assertRaises(ValueError):
      _ZMQClient(port=-1)
    with self.assertRaises(ValueError):
      _ZMQClient(port=65536)

  def test_zmq_server_invalid_timeout(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "send_timeout_s must be non-negative"
    ):
      _ZMQServer(port=12345, send_timeout_s=-1)
    with self.assertRaisesRegex(
        ValueError, "recv_timeout_s must be non-negative"
    ):
      _ZMQServer(port=12345, recv_timeout_s=-1)

  def test_zmq_client_invalid_timeout(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "send_timeout_s must be non-negative"
    ):
      _ZMQClient(port=12345, send_timeout_s=-1)
    with self.assertRaisesRegex(
        ValueError, "recv_timeout_s must be non-negative"
    ):
      _ZMQClient(port=12345, recv_timeout_s=-1)

  def test_zmq_server_client_custom_timeouts(self) -> None:
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port, send_timeout_s=5, recv_timeout_s=10)
    client = _ZMQClient(port=port, send_timeout_s=2, recv_timeout_s=3)
    try:
      self.assertEqual(server._socket.getsockopt(zmq.SNDTIMEO), 5000)
      self.assertEqual(server._socket.getsockopt(zmq.RCVTIMEO), 10000)
      self.assertEqual(client._socket.getsockopt(zmq.SNDTIMEO), 2000)
      self.assertEqual(client._socket.getsockopt(zmq.RCVTIMEO), 3000)
    finally:
      client.close()
      server.close()

  def test_zmq_envelope(self) -> None:
    msg = _make_request(rank=0)
    env = _CollectiveHandshakeRequestZMQEnvelope(client_id=b"c0", request=msg)
    self.assertEqual(env.client_id, b"c0")
    self.assertEqual(env.request, msg)

  def test_zmq_server_client_send_recv(self) -> None:
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    client = _ZMQClient(port=port)

    msg = _make_request(
        rank=1, participating_ranks=[0, 1], fingerprint="fp_comm"
    )

    async def _server_flow() -> None:
      envelope = await server.recv_request()
      self.assertIsNotNone(envelope)
      assert envelope is not None
      self.assertTrue(_msg_eq(envelope.request, msg))
      await server.send_reply(
          envelope.client_id, CollectiveHandshakeResponse(success=True)
      )

    async def _run() -> None:
      server_task = asyncio.create_task(_server_flow())
      # Send message from client
      client.send(msg)
      await server_task
      # Recv reply on client
      reply = client.recv()
      self.assertTrue(reply.success)

    try:
      asyncio.run(_run())
    finally:
      client.close()
      server.close()

  def test_zmq_server_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    server.close()
    # Multiple close calls must be safe and idempotent
    server.close()

    async def _test() -> None:
      with self.assertRaises(zmq.ZMQError):
        await server.recv_request()
      with self.assertRaises(zmq.ZMQError):
        await server.send_reply(
            b"c0", CollectiveHandshakeResponse(success=True)
        )

    asyncio.run(_test())

  def test_zmq_client_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    client = _ZMQClient(port=port)
    client.close()
    # Multiple close calls must be safe and idempotent
    client.close()

    msg = _make_request(rank=0)
    with self.assertRaises(zmq.ZMQError):
      client.send(msg)
    with self.assertRaises(zmq.ZMQError):
      client.recv()


class LoopRunnerTest(seed_test_utils.RepeatableTest):
  """Unit tests for _LoopRunner."""

  def test_loop_runner_shutdown_cancels_pending_tasks_and_closes_loop(
      self,
  ) -> None:
    """Tests that _LoopRunner.close cancels pending tasks, joins the thread, and closes the loop."""
    runner = _LoopRunner(thread_name="TestLoopRunnerShutdown")
    task_started = threading.Event()
    task_cancelled = False

    async def long_running_task() -> None:
      nonlocal task_cancelled
      task_started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        task_cancelled = True
        raise

    runner.run_coroutine_async(long_running_task())
    self.assertTrue(task_started.wait(timeout=2.0))

    runner.close()
    self.assertTrue(task_cancelled)
    self.assertTrue(runner._loop.is_closed())
    self.assertFalse(runner._thread.is_alive())


class HandshakeServerClientBackendTest(seed_test_utils.RepeatableTest):
  """Unit tests for _HandshakeServer and _HandshakeClient."""

  def setUp(self) -> None:
    super().setUp()
    _HandshakeServer._reset_instance()
    _HandshakeClient._reset_instance()

  def tearDown(self) -> None:
    super().tearDown()
    _HandshakeServer._reset_instance()
    _HandshakeClient._reset_instance()

  def test_handshake_server_singleton(self) -> None:
    """Tests that _HandshakeServer acts as a singleton per process."""
    server1 = _HandshakeServer(current_rank=0, port=31234, world_size=1)
    server2 = _HandshakeServer(current_rank=0, port=31234, world_size=1)
    self.assertIs(server1, server2)

  def test_handshake_server_close_cancels_loop_tasks_and_shuts_down_loop(
      self,
  ) -> None:
    """Tests that _HandshakeServer.close cleanly cancels loop tasks and shuts down loop."""
    port = portpicker.pick_unused_port()
    server = _HandshakeServer(current_rank=0, port=port, world_size=1)
    loop_tasks = server._get_loop_tasks()
    loop_runner = server._loop_runner

    self.assertNotEmpty(loop_tasks)
    for task in loop_tasks:
      self.assertFalse(task.done())
    self.assertTrue(loop_runner._thread.is_alive())
    self.assertFalse(loop_runner._loop.is_closed())

    server.close()

    for task in loop_tasks:
      self.assertTrue(task.done())
      self.assertTrue(task.cancelled())
    self.assertTrue(loop_runner._loop.is_closed())
    self.assertFalse(loop_runner._thread.is_alive())
    self.assertTrue(server._zmq_server._socket.closed)
    self.assertTrue(server._zmq_server._context.closed)
    self.assertTrue(server._closed)
    self.assertFalse(server._initialized)

  def test_zmq_ipv6_sockopt(self) -> None:
    """Tests that _ZMQServer and _ZMQClient enable zmq.IPV6 on their sockets."""
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    self.assertEqual(server._socket.getsockopt(zmq.IPV6), 1)
    client = _ZMQClient(port=port)
    self.assertEqual(client._socket.getsockopt(zmq.IPV6), 1)
    client.close()
    server.close()

  @parameterized.parameters("::1", "[::1]", "127.0.0.1", "localhost")
  def test_zmq_communication_address_formats(self, master_addr: str) -> None:
    """Tests communication between _ZMQServer and _ZMQClient with various address formats including IPv6."""
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    with mock.patch.dict(os.environ, {"MASTER_ADDR": master_addr}):
      client = _ZMQClient(port=port)

    req = _make_request(rank=1, participating_ranks=[0, 1])
    client.send(req)

    async def server_roundtrip():
      envelope = await server.recv_request()
      self.assertIsNotNone(envelope)
      self.assertEqual(envelope.request.rank, 1)
      await server.send_reply(
          envelope.client_id, CollectiveHandshakeResponse(success=True)
      )

    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(server_roundtrip())
      resp = client.recv()
      self.assertTrue(resp.success)
    finally:
      loop.close()
      client.close()
      server.close()


def test_wrapper(
    target_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
  """Wrapper for distributed test worker functions.

  Initializes the PyTorch process group before invoking target_fn and tears it
  down afterwards.

  Args:
    target_fn: Target worker function to invoke.
    *args: Positional arguments for `target_fn`.
    **kwargs: Keyword arguments for `target_fn`.

  Returns:
    Result of calling `target_fn(*args, **kwargs)`.
  """
  dist.init_process_group(
      backend="gloo",
      init_method="env://",
      rank=int(os.environ.get("RANK", "0")),
      world_size=int(os.environ.get("WORLD_SIZE", "1")),
  )
  try:
    return target_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def _run_handshake_request_matching(coordinator_rank: int) -> None:
  """Worker function to test matching CollectiveHandshakeRequest across ranks."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  msg = _make_request(
      rank=rank,
      participating_ranks=list(range(world_size)),
      fingerprint="matching_fingerprint_123",
      num_collectives_in_graph=2,
      collective_count_before=0,
  )
  res = hs.submit(msg)
  assert res, f"Rank {rank} expected True, got {res}"


def _run_handshake_request_mismatching(coordinator_rank: int) -> None:
  """Worker function to test mismatching CollectiveHandshakeRequest fingerprints across ranks."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  # Rank 0 has a different fingerprint than other ranks
  fingerprint = "fp_coordinator" if rank == 0 else f"fp_worker_{rank}"
  msg = _make_request(
      rank=rank,
      participating_ranks=list(range(world_size)),
      fingerprint=fingerprint,
      num_collectives_in_graph=2,
      collective_count_before=0,
  )
  res = hs.submit(msg)
  assert not res, f"Rank {rank} expected False for mismatch, got {res}"


def _run_multiple_consecutive_submissions(coordinator_rank: int) -> None:
  """Worker function to test multiple consecutive step submissions."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)

  for step in range(5):
    # Step where all ranks match
    msg_match = _make_request(
        rank=rank,
        participating_ranks=list(range(world_size)),
        fingerprint=f"step_{step}_match",
        num_collectives_in_graph=2,
        collective_count_before=step * 2,
    )
    assert hs.submit(msg_match)

    # Step where ranks mismatch
    msg_mismatch = _make_request(
        rank=rank,
        participating_ranks=list(range(world_size)),
        fingerprint=f"step_{step}_rank_{rank}",
        num_collectives_in_graph=1,
        collective_count_before=(step + 1) * 2,
    )
    assert not hs.submit(msg_mismatch)


def _run_select_handshake_port_pg_broadcast(coordinator_rank: int) -> None:
  """Verifies that non-coordinator ranks receive the broadcasted port."""
  rank = int(os.environ["RANK"])

  port = handshake._select_handshake_port(
      current_rank=rank, coordinator_rank=coordinator_rank
  )

  world_size = dist.get_world_size()
  gathered_ports = [None for _ in range(world_size)] if rank == 0 else None
  dist.gather_object(port, gathered_ports, dst=0)

  if rank == 0:
    assert gathered_ports is not None
    unique_ports = set(gathered_ports)
    assert (
        len(unique_ports) == 1
    ), f"All ranks must see identical port, got {gathered_ports}"
    assert (
        gathered_ports[0] > 0
    ), f"Port must be valid positive int, got {gathered_ports[0]}"


def _run_handshake_with_initialized_process_group(
    coordinator_rank: int,
) -> None:
  """Worker function to test Handshake port exchange over process group."""
  Handshake._reset_instances()
  hs = Handshake(coordinator_rank=coordinator_rank)
  msg = _make_request(
      rank=int(os.environ["RANK"]),
      participating_ranks=list(range(int(os.environ["WORLD_SIZE"]))),
      fingerprint="pg_fingerprint_test",
      num_collectives_in_graph=2,
      collective_count_before=0,
  )
  res = hs.submit(msg)
  assert res, f"Rank {os.environ['RANK']} expected True, got {res}"


def _run_subset_handshake_without_coordinator(coordinator_rank: int) -> None:
  """Worker function testing subset handshake raises RuntimeError."""
  rank = int(os.environ["RANK"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  # Ranks 1, 2, 3 participate (subset of world_size 4), coordinator rank 0 does
  # not call submit().
  if rank != coordinator_rank:
    msg = _make_request(
        rank=rank,
        participating_ranks=[1, 2, 3],
        fingerprint="subset_fingerprint_456",
        num_collectives_in_graph=2,
        collective_count_before=0,
    )
    try:
      hs.submit(msg)
      assert (
          False
      ), f"Rank {rank} expected RuntimeError for subgroup communication"
    except RuntimeError as e:
      assert "do not support subgroup communication" in str(e)
  else:
    time.sleep(2)


def _run_mismatched_frame_count() -> None:
  """Worker function testing mismatched frame count between rank 0 and 1-7."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=0)
  participating_ranks = list(range(world_size))

  if rank == 0:
    # Rank 0 sends 2 handshakes:
    # 1st handshake: fingerprint_A, before=0, num_collectives=1
    msg1 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fingerprint_A",
        num_collectives_in_graph=1,
        collective_count_before=0,
    )
    res1 = hs.submit(msg1)
    assert not res1, f"Rank 0 expected False for 1st submit, got {res1}"
    # 2nd handshake: fingerprint_B, before=1, num_collectives=1
    msg2 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fingerprint_B",
        num_collectives_in_graph=1,
        collective_count_before=1,
    )
    res2 = hs.submit(msg2)
    assert not res2, f"Rank 0 expected False for 2nd submit, got {res2}"
  else:
    # Ranks 1-7 send 1 handshake: fingerprint_C, before=0, num_collectives=2
    msg = _make_request(
        rank=rank,
        participating_ranks=participating_ranks,
        fingerprint="fingerprint_C",
        num_collectives_in_graph=2,
        collective_count_before=0,
    )
    res = hs.submit(msg)
    assert not res, f"Rank {rank} expected False, got {res}"


def _run_mismatched_multi_frame_count_exhaustion() -> None:
  """Worker function testing multi-frame exhaustion before matching step."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=0)
  participating_ranks = list(range(world_size))

  if rank == 0:
    # Rank 0 sends 3 handshakes with 1 collective each (after=1, after=2,
    # after=3).
    msg1 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fp_rank0_1",
        num_collectives_in_graph=1,
        collective_count_before=0,
    )
    assert not hs.submit(msg1)

    msg2 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fp_rank0_2",
        num_collectives_in_graph=1,
        collective_count_before=1,
    )
    assert not hs.submit(msg2)

    msg3 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fp_rank0_3",
        num_collectives_in_graph=1,
        collective_count_before=2,
    )
    assert not hs.submit(msg3)
  else:
    # Ranks 1-7 send 1 handshake with 3 collectives (after=3)
    msg = _make_request(
        rank=rank,
        participating_ranks=participating_ranks,
        fingerprint="fp_other",
        num_collectives_in_graph=3,
        collective_count_before=0,
    )
    assert not hs.submit(msg)

  # Next step: all ranks submit matching messages
  msg_match = _make_request(
      rank=rank,
      participating_ranks=participating_ranks,
      fingerprint="matching_after_multi_exhaust",
      num_collectives_in_graph=2,
      collective_count_before=3,
  )
  assert hs.submit(msg_match)


def _run_target_count_increases_during_exhaustion() -> None:
  """Worker function testing target count increases during exhaustion."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=0)
  participating_ranks = list(range(world_size))

  if rank == 0:
    # Rank 0 starts with 1 collective (after=1), then in exhaust round 1 runs
    # ahead with after=5 (num_collectives_in_graph=4)
    msg1 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fp_rank0_1",
        num_collectives_in_graph=1,
        collective_count_before=0,
    )
    assert not hs.submit(msg1)

    msg2 = _make_request(
        rank=0,
        participating_ranks=participating_ranks,
        fingerprint="fp_rank0_2",
        num_collectives_in_graph=4,
        collective_count_before=1,
    )
    assert not hs.submit(msg2)
  else:
    # Ranks 1-3 start with 2 collectives (after=2).
    # Initially target_count=2, so only rank 0 is in ranks_to_exhaust.
    # When rank 0 sends after=5 in exhaust round 1, ranks 1-3 must be added to
    # ranks_to_exhaust in round 2 (num_collectives_in_graph=3).
    msg1 = _make_request(
        rank=rank,
        participating_ranks=participating_ranks,
        fingerprint=f"fp_rank_{rank}_1",
        num_collectives_in_graph=2,
        collective_count_before=0,
    )
    assert not hs.submit(msg1)

    msg2 = _make_request(
        rank=rank,
        participating_ranks=participating_ranks,
        fingerprint=f"fp_rank_{rank}_2",
        num_collectives_in_graph=3,
        collective_count_before=2,
    )
    assert not hs.submit(msg2)

  # Step 2: Next matching step where all ranks match.
  msg_match = _make_request(
      rank=rank,
      participating_ranks=participating_ranks,
      fingerprint="matching_step2",
      num_collectives_in_graph=2,
      collective_count_before=5,
  )
  assert hs.submit(msg_match)


class HandshakeTest(seed_test_utils.RepeatableTest):
  """Unit tests for distributed Handshake protocol on CPU."""

  def setUp(self) -> None:
    super().setUp()
    Handshake._reset_instances()

  def tearDown(self) -> None:
    super().tearDown()
    Handshake._reset_instances()

  def test_handshake_port_cache(self) -> None:
    """Tests that Handshake caches port selection per coordinator rank."""
    with mock.patch.object(
        handshake, "_select_handshake_port", return_value=31234
    ):
      with mock.patch.object(dist, "get_world_size", return_value=1):
        _ = Handshake(current_rank=0, coordinator_rank=0)
        self.assertEqual(Handshake._port_cache[0], 31234)

  def test_submit_invalid_message_type(self) -> None:
    """Tests that submitting non-CollectiveHandshakeRequest raises TypeError."""
    with mock.patch.object(
        handshake, "_select_handshake_port", return_value=31234
    ):
      with mock.patch.object(dist, "get_world_size", return_value=1):
        hs = Handshake(current_rank=0, coordinator_rank=0)
        with self.assertRaises(TypeError):
          hs.submit("invalid_string")  # pyrefly: ignore[bad-argument-type]

  def test_submit_subgroup_not_supported(self) -> None:
    """Tests that submitting participating_ranks != world_size raises RuntimeError."""
    with mock.patch.object(
        handshake, "_select_handshake_port", return_value=31234
    ):
      with mock.patch.object(dist, "get_world_size", return_value=4):
        hs = Handshake(current_rank=0, coordinator_rank=0)
        msg = _make_request(
            rank=0,
            participating_ranks=[0, 1],  # len 2 != world_size 4
            fingerprint="fp",
            num_collectives_in_graph=2,
            collective_count_before=0,
        )
        with self.assertRaisesRegex(
            RuntimeError, "do not support subgroup communication"
        ):
          hs.submit(msg)

  def test_select_handshake_port_pg_broadcast(self) -> None:
    """Tests that worker ranks receive the broadcasted port from coordinator."""
    distributed_utils.dist_run(
        3, test_wrapper, _run_select_handshake_port_pg_broadcast, 0
    )

  @parameterized.parameters(0, 1)
  def test_handshake_request_matching(self, coordinator_rank: int) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_handshake_request_matching, coordinator_rank
    )

  @parameterized.parameters(0, 1)
  def test_handshake_request_mismatching(self, coordinator_rank: int) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_handshake_request_mismatching, coordinator_rank
    )

  def test_multiple_consecutive_submissions(self) -> None:
    distributed_utils.dist_run(
        3, test_wrapper, _run_multiple_consecutive_submissions, 0
    )

  def test_handshake_with_initialized_process_group(self) -> None:
    distributed_utils.dist_run(
        3, test_wrapper, _run_handshake_with_initialized_process_group, 0
    )

  def test_subset_handshake_without_coordinator(self) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_subset_handshake_without_coordinator, 0
    )

  def test_mismatched_frame_count(self) -> None:
    """Tests that mismatched frame count across ranks is exhausted cleanly without deadlocking."""
    distributed_utils.dist_run(8, test_wrapper, _run_mismatched_frame_count)

  def test_mismatched_multi_frame_count_exhaustion(self) -> None:
    """Tests that multi-frame count mismatch across ranks is exhausted cleanly before a subsequent matching step."""
    distributed_utils.dist_run(
        8, test_wrapper, _run_mismatched_multi_frame_count_exhaustion
    )

  def test_target_count_increases_during_exhaustion(self) -> None:
    """Tests that when target_count increases during an exhaust round, lagging ranks are exhausted to the higher target count."""
    distributed_utils.dist_run(
        4, test_wrapper, _run_target_count_increases_during_exhaustion
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
