"""BaseInstalledAgent wrapper for BFCL multi-turn harnesses.

Uses `run_bfclmt_agent.py` with BFCL's environment and checker bundled as
`bfcl_env_bundle.tgz`, both uploaded into the sandbox. `harness` is one of `fc`,
`json`, `code` (the `pycall` harness) or `xml`.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNNER_FILENAME = "run_bfclmt_agent.py"
RUNNER_TARGET_PATH = f"/{RUNNER_FILENAME}"
INSTRUCTION_TARGET_PATH = "/tmp/bfclmt_agent_instruction.md"
VALID_HARNESSES = ("fc", "json", "code", "xml")


class BfclMtAgent(BaseInstalledAgent):
    def __init__(self, *args: Any, harness: str, temperature: float = 0.0,
                 max_tokens: int = 4096, **kwargs: Any) -> None:
        if harness not in VALID_HARNESSES:
            raise ValueError(f"harness must be one of {VALID_HARNESSES}, got {harness!r}")
        super().__init__(*args, **kwargs)
        self._harness = harness
        self._temperature = temperature
        self._max_tokens = max_tokens

    @staticmethod
    def name() -> str:
        return "bfclmt-agent"

    def version(self) -> str | None:
        return self._version or "1.0"

    @property
    def install_agent_template_path(self) -> Path | None:
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment, command="python3 -m pip install --no-cache-dir litellm==1.83.0 mpmath"
        )
        here = Path(__file__).parent
        local_copy = self.logs_dir / RUNNER_FILENAME
        local_copy.write_text((here / RUNNER_FILENAME).read_text(encoding="utf-8"), encoding="utf-8")
        await environment.upload_file(local_copy, RUNNER_TARGET_PATH)
        bundle_copy = self.logs_dir / "bfcl_env_bundle.tgz"
        bundle_copy.write_bytes((here / "bfcl_env_bundle.tgz").read_bytes())
        await environment.upload_file(bundle_copy, "/bfcl_env_bundle.tgz")
        await self.exec_as_root(
            environment, command="mkdir -p /agent_lib && tar xzf /bfcl_env_bundle.tgz -C /agent_lib"
        )

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("model_name is required for BfclMtAgent")
        instruction_copy = self.logs_dir / "bfclmt-agent.instruction.md"
        instruction_copy.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_copy, INSTRUCTION_TARGET_PATH)
        cmd = " ".join([
            "python3", RUNNER_TARGET_PATH,
            "--model", shlex.quote(self.model_name),
            "--harness", self._harness,
            "--instruction-file", shlex.quote(INSTRUCTION_TARGET_PATH),
            "--temperature", str(self._temperature),
            "--max-tokens", str(self._max_tokens),
        ]) + " 2>&1 | tee /logs/agent/bfclmt-agent.txt"
        await self.exec_as_agent(environment, command=cmd, env=self._build_runner_env())

    def populate_context_post_run(self, context: AgentContext) -> None:
        context.metadata = {"harness": f"bfclmt-{self._harness}"}

    def _build_runner_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
            value = self._get_env(key)
            if value:
                env[key] = value
        for source in (os.environ, self._extra_env):
            for key, value in source.items():
                if key.startswith("LITELLM_") and value:
                    env[key] = value
        return env


class DispatchAgent(BfclMtAgent):
    """Dispatch shim so job yamls can use kwargs {harness: bfclmt-<h>} like other lines."""
    def __init__(self, *args: Any, harness: str, **kwargs: Any) -> None:
        h = harness.replace("bfclmt-", "")
        super().__init__(*args, harness=h, **kwargs)
