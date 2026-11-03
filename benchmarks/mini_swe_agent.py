"""Pinned mini-swe-agent setup, for the benchmarks that use it (Reasoning Gym, AppWorld).

The original runs used mini-swe-agent 2.4.6. Newer LiteLLM releases can fail
because they import `mcp` without installing it.

This class pins mini-swe-agent to 2.4.6 and LiteLLM to 1.83.14. The original
runs' LiteLLM version was not recorded; 1.83.14 is compatible with
mini-swe-agent 2.4.6 and matches Harbor's own environment.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment

MINI_SWE_AGENT_VERSION = "2.4.6"
LITELLM_VERSION = "1.83.14"

# The Python of the environment `uv tool install` creates for mini-swe-agent
TOOL_PYTHON = "$HOME/.local/share/uv/tools/mini-swe-agent/bin/python"


class PinnedMiniSweAgent(MiniSweAgent):
    def __init__(self, *args: Any, version: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, version=version or MINI_SWE_AGENT_VERSION, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await self.exec_as_agent(
            environment,
            command=f'. "$HOME/.local/bin/env"; uv pip install -q --python {TOOL_PYTHON} litellm=={LITELLM_VERSION}',
        )
