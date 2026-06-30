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

"""Utilities for OSS/GCP environment detection and testing."""

import functools


def running_in_cloud() -> bool:
  """Returns True if the codebase is running in GCP."""
  try:
    import libtpu  # pylint: disable=g-import-not-at-top,unused-import # pytype: disable=import-error

    return True
  except ImportError:
    return False


def libtpu_version() -> str:
  """Returns the libtpu version."""
  try:
    import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    return libtpu.__version__
  except ImportError as e:
    raise ValueError("libtpu is not available") from e


def skip_if_cloud_and_libtpu_older_than(min_version: str):
  """Decorator to skip test in cloud if libtpu version < min_version."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if running_in_cloud():
        version = libtpu_version()
        if version < min_version:
          self.skipTest(
              f"{func.__name__} requires libtpu >= {min_version} in cloud, but"
              f" got {version}"
          )
      return func(self, *args, **kwargs)

    return wrapper

  return decorator
