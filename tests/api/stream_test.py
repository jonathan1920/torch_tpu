# Copyright 2025 Google LLC
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

from absl.testing import absltest
import torch
from torch_tpu import api


class StreamsTest(absltest.TestCase):

  def test_stream(self):
    """Tests basic usage of streams and events.

    Because they are dummy methods no functionality is tested here.
    This test only checks that the APIs can be used without throwing an error.
    """
    _ = api.tpu_device()
    default_stream = torch.tpu.default_stream()
    new_stream = torch.tpu.Stream(priority=-1)

    torch.tpu.set_stream(new_stream)
    x = torch.ones((8, 8), device='tpu')
    event = new_stream.record_event()
    torch.tpu.set_stream(default_stream)

    with torch.tpu.stream(new_stream):
      y = x + x

    default_stream.wait_event(event)
    default_stream.wait_stream(new_stream)
    z = y + y
    z.to('cpu')

  def test_stream_unimplemented_method(self):
    _ = api.tpu_device()
    expected_msg = (
        'Streams and Events are not implemented in TorchTPU. Please file a'
        ' feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      default_stream = torch.tpu.default_stream()
      default_stream.query()


if __name__ == '__main__':
  absltest.main()
