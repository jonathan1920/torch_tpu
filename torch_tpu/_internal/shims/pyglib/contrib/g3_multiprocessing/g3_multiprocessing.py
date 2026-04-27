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

"""Bridge for internal<->external multiprocessing setup.

See internal version of multiprocessing_setup
"""

import multiprocessing
from absl import app


def handle_main(main, *args, **kwargs):
  return app.run(main, *args, **kwargs)


def handle_test_main(main, *args, **kwargs):
  return main(*args, **kwargs)


# g3_multiprocessing constants
ABSL_SPAWN = "spawn"
ABSL_FORKSERVER = "forkserver"


def get_context(method=None):
  if method == ABSL_SPAWN:
    method = "spawn"
  elif method == ABSL_FORKSERVER:
    method = "forkserver"
  return multiprocessing.get_context(method)
