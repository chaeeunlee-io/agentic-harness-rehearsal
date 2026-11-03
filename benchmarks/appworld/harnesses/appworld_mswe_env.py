"""mini-swe-agent environment for executing Python in a persistent AppWorld world.

Each action is sent directly to the AppWorld environment server's `/execute`
endpoint instead of being run through a shell. This avoids shell quoting issues
for multi-line Python while preserving AppWorld state across actions.

The environment runs inside mini-swe-agent's isolated uv-tool environment and
communicates with AppWorld over HTTP using only the standard library.

Implements the expected mini-swe-agent environment interface and raises
`Submitted` when AppWorld reports task completion.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted


class AppWorldEnvironmentConfig(BaseModel):
    env_url: str = "http://localhost:9000"
    timeout: int = 600


def _post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AppWorldEnvironment:
    def __init__(self, *, config_class: type = AppWorldEnvironmentConfig, **kwargs: Any):
        self.config = config_class(**kwargs)

    def _task_id(self) -> str:
        tid = os.environ.get("APPWORLD_TASK_ID")
        if not tid:
            raise RuntimeError("APPWORLD_TASK_ID not set in sandbox")
        return tid

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        code = action.get("command", "")
        url = self.config.env_url.rstrip("/")
        tid = self._task_id()
        try:
            out = _post(f"{url}/execute", {"task_id": tid, "code": code}, timeout or self.config.timeout)
            text = out.get("output", "")
            result = {"output": text, "returncode": 0, "exception_info": ""}
        except Exception as e:
            return {
                "output": f"AppWorld env-server /execute failed: {e}",
                "returncode": -1,
                "exception_info": str(e),
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        # AppWorld's own completion (agent called apis.supervisor.complete_task)
        # is the episode's stop signal -- surface it as mini-swe's Submitted.
        try:
            done = bool(_post(f"{url}/task_completed", {"task_id": tid}, 60).get("output"))
        except Exception:
            done = False
        if done:
            raise Submitted(
                {
                    "role": "exit",
                    "content": text,
                    "extra": {"exit_status": "Submitted", "submission": text},
                }
            )
        return result

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {**self.config.model_dump(), **kwargs}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
