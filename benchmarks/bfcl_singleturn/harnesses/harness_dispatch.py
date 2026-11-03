"""Dispatch BFCL rows to the selected harness.

Supported harnesses:
- `bfcl-fc`: native OpenAI tool calls.
- `bfcl-json`: text JSON with `analysis` and `calls`.
- `bfcl-xml`: XML-style reasoning and calls; evaluation only.
- `bfcl-t2`: Terminus-2 terminal scaffold.

`bfcl-t2` differs from the other harnesses in both output format and use of a
multi-turn scaffold, so it is not a format-only comparison.

Unknown harness names fail at construction.

Example:
    agents:
      - import_path: benchmarks.bfcl_singleturn.harnesses.harness_dispatch:DispatchAgent
        model_name: hosted_vllm/Qwen3-4B
        kwargs:
          harness: bfcl-fc
          api_base: http://<model server>:8000/v1

`bfcl-t2` also requires `model_info` for hosted vLLM models:
    kwargs:
      harness: bfcl-t2
      model_info:
        max_input_tokens: 32768
        max_output_tokens: 4096
        input_cost_per_token: 0.0
        output_cost_per_token: 0.0
"""

from __future__ import annotations

import logging
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

try:  # harbor repo root convention
    import importlib

    BfclAgent = importlib.import_module("benchmarks.bfcl_singleturn.harnesses.bfcl_agent").BfclAgent
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

    # bfcl_agent imports bfcl_protocols by bare name, so make the adapter dir
    # importable before loading it.
    sys.path.insert(0, str(_Path(__file__).parent))
    BfclAgent = _load_sibling("bfcl_agent", "bfcl_agent.py").BfclAgent

logger = logging.getLogger(__name__)

HARNESS_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "bfcl-fc": (BfclAgent, {"harness": "fc"}),
    "bfcl-json": (BfclAgent, {"harness": "json"}),
    "bfcl-xml": (BfclAgent, {"harness": "xml"}),  # held out, eval-only
}

# terminus-2 rung: the adapter's own shell-based contract, and the bridge to
# the RG harness family. Registered behind try/except so an environment
# without terminus-2 available still gets a working fc/json/xml registry
# (same defensive pattern as rg/medagentbench use for goose).
try:
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    HARNESS_REGISTRY["bfcl-t2"] = (
        Terminus2,
        {"parser": "json", "max_episodes": 8},
    )
except ImportError:  # pragma: no cover
    logger.debug("terminus-2 unavailable; bfcl-t2 rung not registered")


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
        return "bfcl-harness-dispatch"

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
