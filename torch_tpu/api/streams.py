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

"""Python interface for streams and events on TPU."""

from typing import Optional, Self

_NOT_IMPLEMENTED_STREAMS_MSG = (
    'Streams and Events are not implemented in TorchTPU. Please file a feature'
    ' request describing your use case.'
)


class TpuStream:
  """A stream of operations on a TPU device.

  Streams are a CUDA concept with no clear analog in XLA.
  They refer to a linear sequence of executions on a specific device.
  Along with events they can be used to schedule and synchronize parallel work
  streams. For example, one stream might write to a buffer and record an event
  while a second stream waits for that event before reading from the buffer.
  Because XLA is functional it can schedule operations based on data
  dependencies and does not need streams, at least for the use cases we have
  considered so far around distributed algorithms.

  This class implements dummy methods that are needed to enable
  PyTorch's Fully Sharded Data Parallel. Other methods are left unimplemented
  to help identify new use cases. The CUDA Stream class on which this is
  based can be found in torch/cuda/streams.py.

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __init__(self, device=None, priority: int = 0, **kwargs):
    pass

  def wait_event(self, event: 'TpuEvent') -> None:  # pylint: disable=unused-argument
    """Makes all future work submitted to the stream wait for an event."""
    event.wait()

  def wait_stream(self, stream: Self) -> None:  # pylint: disable=unused-argument
    """Synchronizes with another stream."""
    self.wait_event(stream.record_event())

  def record_event(self, event: Optional['TpuEvent'] = None) -> 'TpuEvent':
    """Records an event on this stream."""
    if event is None:
      event = TpuEvent()
    event.record(self)
    return event

  def query(self) -> bool:
    """Checks if all work submitted on this stream has been completed."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def synchronize(self) -> None:
    """Waits for all work submitted on this stream to complete."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self):
    return '<torch.tpu.TpuStream>'


class TpuEvent:
  """A synchronization primitive for TPU operations.

  This class implements dummy methods that are needed to enable
  PyTorch's Fully Sharded Data Parallel. Other methods are left unimplemented
  to help identify new use cases. The CUDA Event class on which this is
  based can be found in torch/cuda/streams.py.

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __init__(
      self,
      enable_timing: bool = False,
      blocking: bool = False,
      interprocess: bool = False,
      external: bool = False,
  ):
    pass

  def record(self, stream: TpuStream | None = None):  # pylint: disable=unused-argument
    """Records this event on the given stream."""
    return

  def wait(self, stream: TpuStream | None = None) -> None:  # pylint: disable=unused-argument
    """Makes all future work submitted to the given stream wait on this event."""
    return

  def query(self) -> bool:
    """Checks if all work currently captpured by this event has completed."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def elapsed_time(self, end_event: Self) -> float:  # pylint: disable=unused-argument
    """Returns the time elapsed between recording and completing this event."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def synchronize(self) -> None:
    """Waits for this event to complete."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  @classmethod
  def from_ipc_handle(cls, device, handle):
    """Reconstructs an event from an IPC handle on the given device."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def ipc_handle(self):
    """Returns an IPC handle of this event."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self) -> str:
    return '<torch.tpu.TpuEvent>'
