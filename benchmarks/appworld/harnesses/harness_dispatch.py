"""Dispatch AppWorld rows to the selected harness.

Supported harnesses:
- `appworld-code`: fenced Python blocks.
- `appworld-fc`: a single `execute_python(code)` tool call.
- `appworld-mini`: mini-swe-agent (`appworld_mini_swe_agent.py`).
- `appworld-xml`: `<reasoning>` / `<code>` tags; evaluation only.

Unknown harness names fail at construction.

Example:
    agents:
      - import_path: benchmarks.appworld.harnesses.harness_dispatch:DispatchAgent
        model_name: hosted_vllm/Qwen3-4B
        kwargs:
          harness: appworld-fc
          api_base: http://<model server>:8000/v1
"""

from __future__ import annotations

import logging
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

try:  # harbor repo root convention
    import importlib

    AppWorldCodeAgent = importlib.import_module(
        "benchmarks.appworld.harnesses.appworld_code_agent"
    ).AppWorldCodeAgent
except ImportError:  # SkyRL cwd -- sibling-file fallback
    import importlib.util
    import sys
    from pathlib import Path as _Path

    def _load_sibling(module_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(
            module_name, _Path(__file__).parent / filename
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, mod)
        spec.loader.exec_module(mod)
        return mod

    # appworld_code_agent imports appworld_protocols by bare name inside the
    # sandbox, but here (orchestrator side) it does not import it at module
    # load; make the adapter dir importable anyway for parity with bfcl.
    sys.path.insert(0, str(_Path(__file__).parent))
    AppWorldCodeAgent = _load_sibling(
        "appworld_code_agent", "appworld_code_agent.py"
    ).AppWorldCodeAgent

logger = logging.getLogger(__name__)

HARNESS_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "appworld-code": (AppWorldCodeAgent, {"harness": "code"}),
    "appworld-fc": (AppWorldCodeAgent, {"harness": "fc"}),
    "appworld-xml": (AppWorldCodeAgent, {"harness": "xml"}),  # held out, eval-only
}

# mini_swe coding-agent rung -- registered behind try/except so an environment
# without harbor.agents.installed.mini_swe_agent still gets a working envelope
# registry (same defensive pattern as bfcl/rg use for their coding-agent rung).
try:
    try:
        _mswe_mod = importlib.import_module("benchmarks.appworld.harnesses.appworld_mini_swe_agent")
    except ImportError:
        _mswe_mod = _load_sibling("appworld_mini_swe_agent", "appworld_mini_swe_agent.py")
    HARNESS_REGISTRY["appworld-mini"] = (_mswe_mod.AppWorldMiniSweAgent, {})
except Exception as _e:  # pragma: no cover
    logger.debug("mini_swe unavailable; appworld-mini rung not registered: %s", _e)

class DispatchAgent:
    """BaseAgent-shaped wrapper delegating to a registry-selected backend."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir=None,
        model_name: str | None = None,
        *,
        harness: str,
        **kwargs: Any,
    ):
        if harness not in HARNESS_REGISTRY:
            raise ValueError(
                f"DispatchAgent: unknown harness {harness!r}. "
                f"Known: {sorted(HARNESS_REGISTRY)}. Refusing to guess."
            )
        backend_cls, presets = HARNESS_REGISTRY[harness]
        merged: dict[str, Any] = {**presets, **kwargs}  # row kwargs win
        self._harness = harness
        self._backend = backend_cls(logs_dir=logs_dir, model_name=model_name, **merged)
        self.SUPPORTS_ATIF = getattr(type(self._backend), "SUPPORTS_ATIF", False)
        self.SUPPORTS_WINDOWS = getattr(type(self._backend), "SUPPORTS_WINDOWS", False)
        logger.info(
            "DispatchAgent: harness=%s backend=%s", harness, backend_cls.__name__
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._backend, item)

    @staticmethod
    def name() -> str:
        return "appworld-harness-dispatch"

    def version(self) -> str | None:
        return f"{self._harness}:{self._backend.version()}"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._backend.run(instruction, environment, context)
        context.metadata = context.metadata or {}
        context.metadata["harness"] = self._harness
        context.metadata["harness_backend"] = type(self._backend).__name__
