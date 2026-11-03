"""AppWorld mini-swe-agent harness.

Wraps Harbor's MiniSweAgent with AppWorld-specific integration:

1. AppWorld's environment server keeps the world state persistent across
   mini-swe-agent's separate action processes. `run()` starts the server and
   initializes the task world.
2. `AppWorldEnvironment` (`appworld_mswe_env.py`) sends Python directly to the
   server's `/execute` endpoint, avoiding shell quoting. `appworld_mswe.yaml`
   provides the matching Python-action prompt.
3. mini-swe-agent runs in its isolated pydantic-v2 environment, while AppWorld
   runs with pydantic v1 behind the HTTP boundary.
4. mini-swe-agent reaches vLLM through LiteLLM's `openai/` provider using
   `OPENAI_API_BASE` and `OPENAI_API_KEY`.

`/save_logs` writes the final AppWorld state to the experiment output directory
for the verifier.

`run()` sets the OpenAI endpoint variables process-wide, so concurrent trials in
one process must use the same endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmarks.mini_swe_agent import PinnedMiniSweAgent   # mini-swe-agent 2.4.6, litellm pinned
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# uploaded-into-sandbox helpers (bare name -> /<name>)
_UPLOADS = {
    "aw_env_init.py": "/aw_env_init.py",       # health-check + /initialize (in-pod)
    "aw_env_save.py": "/aw_env_save.py",       # /save_logs (in-pod)
    "appworld_mswe_env.py": "/appworld_mswe_env.py",  # custom env; cp'd into venv below
}
CONFIG_FILENAME = "appworld_mswe.yaml"
ENV_MODULE = "appworld_mswe_env.py"
ENV_SERVER_PORT = 9000
APPWORLD_ROOT = "/opt/appworld"


class AppWorldMiniSweAgent(PinnedMiniSweAgent):
    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        config_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        if config_file is None:
            config_file = str(Path(__file__).with_name(CONFIG_FILENAME))
        super().__init__(*args, config_file=config_file, **kwargs)
        self._api_base = api_base
        # mini-swe-agent talks to the model via litellm; route to our vLLM
        # through the openai provider so OPENAI_API_BASE takes effect.
        if self.model_name and self.model_name.startswith("hosted_vllm/"):
            self.model_name = "openai/" + self.model_name[len("hosted_vllm/"):]

    @staticmethod
    def name() -> str:
        return "appworld-mini-swe-agent"

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)  # mini-swe-agent into isolated uv-tool venv
        for filename, target in _UPLOADS.items():
            src = Path(__file__).with_name(filename)
            local = self.logs_dir / filename
            local.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            await environment.upload_file(local, target)
        # place the custom environment module ON mini-swe's venv path so its
        # importlib.import_module("appworld_mswe_env") resolves.
        await self.exec_as_agent(
            environment,
            command=(
                'SP=$(ls -d "$HOME"/.local/share/uv/tools/mini-swe-agent/lib/python*/'
                'site-packages 2>/dev/null | head -1); '
                'test -n "$SP" && cp /appworld_mswe_env.py "$SP/" '
                '&& echo "placed env module in $SP" '
                '|| { echo "could not find mini-swe venv site-packages"; exit 1; }'
            ),
        )

    async def _start_env_server_and_init(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(
            environment,
            command=(
                f"nohup appworld serve environment --port {ENV_SERVER_PORT} "
                f"--root {APPWORLD_ROOT} --no-show-usage "
                f">/tmp/aw_env_server.log 2>&1 & "
                f"python3 /aw_env_init.py"
            ),
        )

    async def _save_logs(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, command="python3 /aw_env_save.py")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._start_env_server_and_init(environment)

        # point mini-swe-agent's litellm at our vLLM (openai provider). Set ONLY
        # OPENAI_API_KEY (+ base): MiniSweAgent.run forwards MSWEA_API_KEY if
        # present INSTEAD of the provider key, starving litellm of a key.
        if self._api_base:
            os.environ["OPENAI_API_BASE"] = self._api_base
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        os.environ.pop("MSWEA_API_KEY", None)

        # AppWorld rules + Python-emitting format live in appworld_mswe.yaml's
        # system prompt; pass the task through unmodified.
        await super().run(instruction, environment, context)

        await self._save_logs(environment)
        context.metadata = context.metadata or {}
        context.metadata["harness"] = "appworld-mini"
