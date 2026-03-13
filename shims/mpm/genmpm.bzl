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

"""Shim for genmpm."""

def genmpm(name, deps, testonly = False, **kwargs):
    """Shim for the internal genmpm rule.

    Internally this rule generates a specific binary packaging format;
    in OSS it does nothing as this packaging format is not used.

    Args:
      name: The name of the target.
      deps: The dependencies of the target.
      testonly: Whether the target is testonly.
      **kwargs: Extra arguments.
    """
    _ = name  # @unused
    _ = deps  # @unused
    _ = testonly  # @unused
    _ = kwargs  # @unused
    pass
