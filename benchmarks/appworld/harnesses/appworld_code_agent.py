"""BaseInstalledAgent wrapper for the AppWorld code, fc, and xml harnesses
(chosen with `harness`).

Per-task values such as `APPWORLD_TASK_ID` and `APPWORLD_EXPERIMENT_NAME`
are passed to `exec()` through `task.toml`'s `[environment].env`, so `run()`
only needs to pass the model name.
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

RUNNER_FILENAME = "run_appworld_code_agent.py"
RUNNER_TARGET_PATH = f"/{RUNNER_FILENAME}"
PROTOCOLS_FILENAME = "appworld_protocols.py"
PROTOCOLS_TARGET_PATH = f"/{PROTOCOLS_FILENAME}"
INSTRUCTION_TARGET_PATH = "/tmp/appworld_code_agent_instruction.md"
SUMMARY_FILENAME = "appworld-code-agent-summary.json"
DEFAULT_VERSION = "1.0"
DEFAULT_MAX_STEPS = 40
DEFAULT_MAX_ERRORS = 8
DEFAULT_SEED = 300


class AppWorldCodeAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args: Any,
        temperature: float = 0.0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_errors: int = DEFAULT_MAX_ERRORS,
        seed: int | None = DEFAULT_SEED,
        llm_args_json: str | None = None,
        api_base: str | None = None,
        harness: str = "code",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._temperature = temperature
        self._max_steps = max_steps
        self._max_errors = max_errors
        self._seed = seed
        self._llm_args_json = llm_args_json
        self._api_base = api_base
        self._harness = harness

    @staticmethod
    def name() -> str:
        return "appworld-code-agent"

    def version(self) -> str | None:
        return self._version or DEFAULT_VERSION

    async def install(self, environment: BaseEnvironment) -> None:
        # No pip install here on purpose. AppWorld 0.1.3 pins pydantic V1
        # (image ships pydantic 1.10.26 + sqlmodel 0.0.10); litellm requires
        # pydantic V2, and installing it upgrades pydantic in-place, which
        # breaks `from appworld import ...` (sqlmodel 0.0.10 imports
        # pydantic.errors:ConfigError, removed in V2 -- the two cannot coexist
        # in one process, and this runner must import appworld). The runner
        # instead talks to the vLLM server over its OpenAI-compatible HTTP API
        # using `requests`, which is already in the image and needs no pydantic.
        runner_src = Path(__file__).with_name(RUNNER_FILENAME)
        local_copy = self.logs_dir / RUNNER_FILENAME
        local_copy.write_text(runner_src.read_text(encoding="utf-8"), encoding="utf-8")
        await environment.upload_file(local_copy, RUNNER_TARGET_PATH)
        proto_src = Path(__file__).with_name(PROTOCOLS_FILENAME)
        proto_copy = self.logs_dir / PROTOCOLS_FILENAME
        proto_copy.write_text(proto_src.read_text(encoding="utf-8"), encoding="utf-8")
        await environment.upload_file(proto_copy, PROTOCOLS_TARGET_PATH)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("model_name is required for AppWorldCodeAgent")

        instruction_copy = self.logs_dir / "appworld-code-agent.instruction.md"
        instruction_copy.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_copy, INSTRUCTION_TARGET_PATH)

        env = self._build_runner_env()
        run_command = self._build_run_command()
        await self.exec_as_agent(environment, command=run_command, env=env)

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
        context.n_cache_tokens = summary.get("n_cache_tokens")
        context.cost_usd = summary.get("cost_usd")
        metadata = summary.get("metadata")
        if isinstance(metadata, dict):
            context.metadata = metadata

    def _build_runner_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        passthrough_keys = {
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        }
        for key in passthrough_keys:
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
            shlex.quote(self._harness),
            "--instruction-file",
            shlex.quote(INSTRUCTION_TARGET_PATH),
            "--max-steps",
            str(self._max_steps),
            "--max-errors",
            str(self._max_errors),
            "--temperature",
            str(self._temperature),
            "--seed",
            "none" if self._seed is None else str(self._seed),
            "--summary-path",
            shlex.quote(f"/logs/agent/{SUMMARY_FILENAME}"),
            "--debug-path",
            shlex.quote("/logs/agent/appworld-code-agent-transcript.json"),
        ]
        if self._api_base:
            parts.extend(["--api-base", shlex.quote(self._api_base)])
        if self._llm_args_json:
            parts.extend(["--llm-args-json", shlex.quote(self._llm_args_json)])
        command = " ".join(parts)
        return f"{command} 2>&1 | tee /logs/agent/appworld-code-agent.txt"
