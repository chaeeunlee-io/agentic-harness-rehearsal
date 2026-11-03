"""Dispatch MedAgentBench rows to the selected harness.

Harbor uses one agent class per run, while agent kwargs may vary by row.
`DispatchAgent` uses `harness="<registry key>"` to select the backend agent and
its preset configuration.

Unknown harness names fail at construction. The selected harness is stored in
`context.metadata["harness"]` for each trial.

Registry entries are `(backend_agent, preset_kwargs)`. Per-row kwargs override
preset values, while `harness` itself is consumed by `DispatchAgent`.

Example:
    agents:
      - import_path: benchmarks.medagentbench.harnesses.harness_dispatch:DispatchAgent
        model_name: hosted_vllm/Qwen3-4B
        kwargs:
          harness: t2  # parity, fc, or xml
          api_base: http://...
"""

from __future__ import annotations

import logging
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

try:  # harbor repo root convention
    from benchmarks.medagentbench.harnesses.medagentbench_trainable_agent import (
        MedAgentBenchTrainableAgent,
    )
except ImportError:  # SkyRL cwd — sibling-file fallback (same pattern as trainable)
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _spec = importlib.util.spec_from_file_location(
        "medagentbench_trainable_agent",
        _Path(__file__).parent / "medagentbench_trainable_agent.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("medagentbench_trainable_agent", _mod)
    _spec.loader.exec_module(_mod)
    MedAgentBenchTrainableAgent = _mod.MedAgentBenchTrainableAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: harness id -> (backend class, preset kwargs).
# Presets are backend-appropriate; row kwargs override presets.
# Add variants freely (context budgets, thinking caps, tool sets); NEVER add
# a fallback/default entry — unknown ids must raise.
# ---------------------------------------------------------------------------
HARNESS_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    # Benchmark-native minimal protocol harness (GET/POST/FINISH).
    "parity": (
        MedAgentBenchTrainableAgent,
        {
            "max_rounds": 8,
            "max_tokens": 8192,
        },
    ),
    # Generic terminal harness, budget-honest defaults for our 64k serving.
    "t2": (
        Terminus2,
        {
            "model_info": {
                "max_input_tokens": 57344,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        },
    ),
}

# FC harness (2026-09-04): native tool-calling dialect harness — same info and
# action surface as parity, wire format = OpenAI tool_calls (hermes parser).
# Harness-side; added after the OpenHands in-sandbox-install saga.
try:
    from benchmarks.medagentbench.harnesses.medagentbench_fc_agent import MedAgentBenchFCAgent
except ImportError:
    import importlib.util as _ilu
    from pathlib import Path as _P

    _fspec = _ilu.spec_from_file_location(
        "medagentbench_fc_agent", _P(__file__).parent / "medagentbench_fc_agent.py"
    )
    _fmod = _ilu.module_from_spec(_fspec)
    import sys as _sys

    _sys.modules.setdefault("medagentbench_fc_agent", _fmod)
    _fspec.loader.exec_module(_fmod)
    MedAgentBenchFCAgent = _fmod.MedAgentBenchFCAgent

HARNESS_REGISTRY["fc"] = (
    MedAgentBenchFCAgent,
    {"max_rounds": 8, "max_tokens": 8192},
)


class DispatchAgent:
    """BaseAgent-shaped wrapper delegating to a registry-selected backend.

    Harbor's factory constructs import-path agents as
    ``Agent(logs_dir=..., model_name=..., **kwargs)`` with no subclass check
    (factory.py:131), so a duck-typed wrapper suffices; capability flags are
    mirrored from the constructed backend.
    """

    SUPPORTS_ATIF: bool = False  # mirrored per-instance after construction
    SUPPORTS_WINDOWS: bool = False

    def __init__(self, logs_dir=None, model_name: str | None = None,
                 *, harness: str, **kwargs: Any):
        if harness not in HARNESS_REGISTRY:
            raise ValueError(
                f"DispatchAgent: unknown harness {harness!r}. "
                f"Known: {sorted(HARNESS_REGISTRY)}. Refusing to guess."
            )
        backend_cls, presets = HARNESS_REGISTRY[harness]
        merged: dict[str, Any] = {**presets, **kwargs}  # row kwargs win
        self._harness = harness
        self._backend = backend_cls(
            logs_dir=logs_dir, model_name=model_name, **merged
        )
        self.SUPPORTS_ATIF = getattr(type(self._backend), "SUPPORTS_ATIF", False)
        self.SUPPORTS_WINDOWS = getattr(type(self._backend), "SUPPORTS_WINDOWS", False)
        logger.info(
            "DispatchAgent: harness=%s backend=%s", harness, backend_cls.__name__
        )

    # -- BaseAgent surface ---------------------------------------------------
    # Everything not defined here forwards to the backend (to_agent_info,
    # setup, logger plumbing, and whatever the trial layer grows next).
    def __getattr__(self, item: str) -> Any:
        return getattr(self._backend, item)

    @staticmethod
    def name() -> str:
        return "harness-dispatch"

    def version(self) -> str | None:
        return f"{self._harness}:{self._backend.version()}"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._backend.run(instruction, environment, context)
        # Identity audit trail: every trial records which harness produced it.
        context.metadata = context.metadata or {}
        context.metadata["harness"] = self._harness
        context.metadata["harness_backend"] = type(self._backend).__name__

# XML held-out rung (2026-09-21): pure FORMAT variant of parity — same info,
# same action surface, same budget, XML-tagged wire format. EVALUATION ONLY:
# never used for teacher traces, never in an RL rollout set, so it stays a genuine held-out
# interface (MAB's counterpart to RG's aider and AppWorld's/BFCL's xml).
try:
    from benchmarks.medagentbench.harnesses.medagentbench_xml_agent import MedAgentBenchXMLAgent
except ImportError:
    import importlib.util as _xlu
    import sys as _xsys
    from pathlib import Path as _XP

    _xspec = _xlu.spec_from_file_location(
        "medagentbench_xml_agent", _XP(__file__).parent / "medagentbench_xml_agent.py"
    )
    _xmod = _xlu.module_from_spec(_xspec)
    _xsys.modules.setdefault("medagentbench_xml_agent", _xmod)
    _xspec.loader.exec_module(_xmod)
    MedAgentBenchXMLAgent = _xmod.MedAgentBenchXMLAgent

HARNESS_REGISTRY["xml"] = (
    MedAgentBenchXMLAgent,
    {"max_rounds": 8, "max_tokens": 8192},
)
