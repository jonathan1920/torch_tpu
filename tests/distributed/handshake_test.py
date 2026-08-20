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

"""Tests for distributed handshake protocol."""

import asyncio
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch.distributed as dist
from torch_tpu._internal.distributed import handshake
from tests import seed_test_utils

_CollectiveHandshakeConsensus = handshake._CollectiveHandshakeConsensus
CollectiveHandshakeRequest = handshake.CollectiveHandshakeRequest
ProcessGroupId = handshake.ProcessGroupId
ProcessGroupCollectiveCount = handshake.ProcessGroupCollectiveCount
RankCollectiveCounts = handshake.RankCollectiveCounts

# TODO(b/542976786): Distributed tests will be added in a follow up CL.


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
      with self.assertRaisesRegex(ValueError, r"rank \(-1\) must be in range"):
        await consensus.put_request(-1, "req")
      with self.assertRaisesRegex(ValueError, r"rank \(4\) must be in range"):
        await consensus.put_request(4, "req")

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
      await consensus.put_request(2, "req_rank_2")
      result = await consensus.get_first_request()
      self.assertEqual(result, "req_rank_2")

    asyncio.run(_test())

  def test_get_from_ranks(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      await consensus.put_request(0, "req_0")
      await consensus.put_request(1, "req_1")
      await consensus.put_request(2, "req_2")
      await consensus.put_request(3, "req_3")

      results = await consensus.get_from_ranks([1, 3])
      self.assertEqual(results, ["req_1", "req_3"])

    asyncio.run(_test())

  def test_get_from_ranks_order(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      await consensus.put_request(0, "req_0")
      await consensus.put_request(1, "req_1")
      await consensus.put_request(2, "req_2")
      await consensus.put_request(3, "req_3")

      results = await consensus.get_from_ranks([3, 1, 0])
      self.assertEqual(results, ["req_3", "req_1", "req_0"])

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

      async def _delayed_put() -> None:
        await asyncio.sleep(0.01)
        await consensus.put_request(1, "delayed_req")

      task = asyncio.create_task(_delayed_put())
      result = await consensus.get_first_request()
      await task
      self.assertEqual(result, "delayed_req")

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


if __name__ == "__main__":
  absltest.main()
