#!/usr/bin/env python3
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

"""Strict two-party ping-pong issue turn tracking for TorchTPU.

Rules:
1. Only two parties participate: original poster (creator) and assignee(s).
2. Comments from any other party are ignored.
3. When the original poster comments, status becomes 'stat:awaiting
torchtpu-eng'.
4. When the assignee comments, status becomes 'stat:awaiting response from
contributor'.
5. New, reopened, or transferred issues default to 'stat:awaiting torchtpu-eng'.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Final, Mapping, NamedTuple, Sequence
import urllib.error
import urllib.parse
import urllib.request

TEAM_LABEL: Final[str] = "stat:awaiting torchtpu-eng"
USER_LABEL: Final[str] = "stat:awaiting response from contributor"
STALE_LABEL: Final[str] = "stale"


class LabelChanges(NamedTuple):
  add: Sequence[str] = ()
  remove: Sequence[str] = ()


def determine_label_changes(
    event_name: str, payload: Mapping[str, Any]
) -> LabelChanges:
  """Determines labels to add and remove based on the event."""
  issue = payload.get("issue")
  if not issue:
    return LabelChanges()

  if issue.get("pull_request"):
    return LabelChanges()

  if issue.get("state") == "closed" and event_name != "issues":
    return LabelChanges()

  existing_labels: dict[str, str] = {}
  for item in issue.get("labels", []):
    name = item if isinstance(item, str) else item.get("name")
    if name:
      existing_labels[name.lower()] = name

  if event_name == "issues":
    action = payload.get("action")
    if action in ("opened", "reopened", "transferred"):
      to_add = []
      to_remove = []
      if TEAM_LABEL.lower() not in existing_labels:
        to_add.append(TEAM_LABEL)
      if USER_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[USER_LABEL.lower()])
      if STALE_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[STALE_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))
    return LabelChanges()

  if event_name == "issue_comment":
    action = payload.get("action")
    if action != "created":
      return LabelChanges()

    sender = payload.get("sender") or {}
    if sender.get("type") == "Bot" or (sender.get("login") or "").endswith(
        "[bot]"
    ):
      return LabelChanges()

    sender_login = (sender.get("login") or "").lower()
    if not sender_login:
      return LabelChanges()

    op_login = ((issue.get("user") or {}).get("login") or "").lower()

    raw_assignees = [issue.get("assignee"), *(issue.get("assignees") or [])]
    assignees: frozenset[str] = frozenset(
        a["login"].lower() for a in raw_assignees if a and a.get("login")
    )

    is_op = bool(op_login and sender_login == op_login)
    is_assignee = sender_login in assignees

    if is_op and is_assignee:
      return LabelChanges()

    to_add = []
    to_remove = []

    if is_assignee:
      if USER_LABEL.lower() not in existing_labels:
        to_add.append(USER_LABEL)
      if TEAM_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[TEAM_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

    if is_op:
      if TEAM_LABEL.lower() not in existing_labels:
        to_add.append(TEAM_LABEL)
      if USER_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[USER_LABEL.lower()])
      if STALE_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[STALE_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

    return LabelChanges()

  return LabelChanges()


def apply_label_changes(
    api_url: str,
    repo: str,
    token: str,
    issue_number: int,
    to_add: Sequence[str],
    to_remove: Sequence[str],
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
  """Executes GitHub REST API requests to update labels."""
  headers = {
      "Authorization": f"Bearer {token}",
      "Accept": "application/vnd.github+json",
      "User-Agent": "TorchTPU-Issue-Turn-Tracker",
      "X-GitHub-Api-Version": "2022-11-28",
  }

  for label in to_remove:
    encoded_name = urllib.parse.quote(label)
    url = f"{api_url}/repos/{repo}/issues/{issue_number}/labels/{encoded_name}"
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
      with opener(req, timeout=timeout):
        pass
    except urllib.error.HTTPError as err:
      if err.code != 404:
        raise

  if to_add:
    url = f"{api_url}/repos/{repo}/issues/{issue_number}/labels"
    data = json.dumps({"labels": list(to_add)}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with opener(req, timeout=timeout):
      pass


def main() -> None:
  token = os.environ.get("GITHUB_TOKEN")  # PY_ENVIRON_OK=ci
  event_path = os.environ.get("GITHUB_EVENT_PATH")  # PY_ENVIRON_OK=ci
  repo = os.environ.get("GITHUB_REPOSITORY")  # PY_ENVIRON_OK=ci
  event_name = os.environ.get("GITHUB_EVENT_NAME")  # PY_ENVIRON_OK=ci
  api_url = os.environ.get(  # PY_ENVIRON_OK=ci
      "GITHUB_API_URL", "https://api.github.com"
  )

  if not (token and event_path and repo and event_name):
    print("Missing required environment variables", file=sys.stderr)
    sys.exit(1)

  event_payload_path = Path(event_path)
  payload = json.loads(event_payload_path.read_text(encoding="utf-8"))

  issue_number = (payload.get("issue") or {}).get("number")
  if not issue_number:
    return

  changes = determine_label_changes(event_name, payload)
  if not changes.add and not changes.remove:
    return

  apply_label_changes(
      api_url, repo, token, issue_number, changes.add, changes.remove
  )


if __name__ == "__main__":
  main()
