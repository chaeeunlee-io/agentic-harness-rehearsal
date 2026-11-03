"""BaseInstalledAgent wrapper for the BFCL single-turn fc, json, and xml harnesses
(`t2` is Harbor's Terminus-2, registered in `harness_dispatch.py`).

Handles installation, execution, environment passthrough, and post-run context
collection for single-turn BFCL tasks.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNNER_FILENAME = "run_bfcl_agent.py"
DEPENDENCY_FILENAMES = ["bfcl_protocols.py"]
RUNNER_TARGET_PATH = f"/{RUNNER_FILENAME}"
INSTRUCTION_TARGET_PATH = "/tmp/bfcl_agent_instruction.md"
SUMMARY_FILENAME = "bfcl-agent-summary.json"
DEFAULT_VERSION = "1.0"
VALID_HARNESSES = ("fc", "json", "xml")


class BfclAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args: Any,
        harness: str,
        temperature: float = 0.0,
        seed: int | None = 300,
        llm_args_json: str | None = None,
        **kwargs: Any,
    ) -> None:
        if harness not in VALID_HARNESSES:
            raise ValueError(
                f"BfclAgent harness must be one of {VALID_HARNESSES}, got {harness!r}"
            )
        super().__init__(*args, **kwargs)
        self._harness = harness
        self._temperature = temperature
        self._seed = seed
        self._llm_args_json = llm_args_json

    @staticmethod
    def name() -> str:
        return "bfcl-agent"

    def version(self) -> str | None:
        return self._version or DEFAULT_VERSION

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment, command="python3 -m pip install --no-cache-dir litellm==1.83.0"
        )
        here = Path(__file__).parent
        for filename in [RUNNER_FILENAME, *DEPENDENCY_FILENAMES]:
            local_copy = self.logs_dir / filename
            local_copy.write_text(
                (here / filename).read_text(encoding="utf-8"), encoding="utf-8"
            )
            await environment.upload_file(local_copy, f"/{filename}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("model_name is required for BfclAgent")

        instruction_copy = self.logs_dir / "bfcl-agent.instruction.md"
        instruction_copy.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_copy, INSTRUCTION_TARGET_PATH)

        await self.exec_as_agent(
            environment,
            command=self._build_run_command(),
            env=self._build_runner_env(),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        summary_path = self.logs_dir / SUMMARY_FILENAME
        if not summary_path.exists():
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        context.n_input_tokens = summary.get("n_input_tokens")
        context.n_output_tokens = summary.get("n_output_tokens")
        context.cost_usd = summary.get("cost_usd")
        metadata = summary.get("metadata")
        if isinstance(metadata, dict):
            context.metadata = metadata

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

    def _build_run_command(self) -> str:
        parts = [
            "python3",
            RUNNER_TARGET_PATH,
            "--model",
            shlex.quote(self.model_name or ""),
            "--harness",
            self._harness,
            "--instruction-file",
            shlex.quote(INSTRUCTION_TARGET_PATH),
            "--temperature",
            str(self._temperature),
            "--seed",
            "none" if self._seed is None else str(self._seed),
            "--summary-path",
            shlex.quote(f"/logs/agent/{SUMMARY_FILENAME}"),
            "--debug-path",
            shlex.quote("/logs/agent/bfcl-agent-transcript.json"),
        ]
        if self._llm_args_json:
            parts.extend(["--llm-args-json", shlex.quote(self._llm_args_json)])
        return " ".join(parts) + " 2>&1 | tee /logs/agent/bfcl-agent.txt"
