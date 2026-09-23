"""Thin client for the daemon.

Deliberately stdlib-only: the Claude Code hook that will call this must start
fast and must not drag in a dependency tree.
"""

from __future__ import annotations

import json
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8778"


def decide(
    state: str,
    questions: list[dict],
    base_url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict:
    payload = json.dumps({"state": state, "questions": questions}).encode()
    request = urllib.request.Request(
        f"{base_url}/decide",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def health(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as response:
        return json.loads(response.read())
