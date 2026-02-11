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

"""Define the rules_python repo."""

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

def rules_python_repo():
    """Define the rule_python repo."""

    # NOTE: XLA defines various patches. If a different rules_python version
    # needs to be used (or a local one for debugging), those patches have to
    # also being applied. See `python_init_rules()` implementation for
    # details on the patches.
    python_init_rules()
