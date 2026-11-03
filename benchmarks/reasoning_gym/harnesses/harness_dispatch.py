"""Dispatch Reasoning Gym rows to the selected harness.

Used for GRPO rollouts; evaluation runs Harbor's agents directly (see the
benchmark config). Supported harnesses: `t2`, `miniswe`, and `goose`. The
harness is selected per row through `kwargs["harness"]`.

Installed agents read model endpoints from the environment; wrappers translate
the `api_base` argument as needed.
"""
from typing import Any

HARNESS_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {}

from harbor.agents.terminus_2.terminus_2 import Terminus2  # noqa: E402
HARNESS_REGISTRY["t2"] = (Terminus2, {"parser": "json"})

try:
    from benchmarks.mini_swe_agent import PinnedMiniSweAgent as _Mini   # mini-swe-agent 2.4.6, litellm pinned
    class MiniSweDispatch(_Mini):
        def __init__(self, *args, api_base: str | None = None, **kwargs):
            kwargs.pop("harness", None)
            extra = kwargs.setdefault("extra_env", {})
            if api_base:
                extra.setdefault("OPENAI_API_BASE", api_base)
            super().__init__(*args, **kwargs)
    HARNESS_REGISTRY["miniswe"] = (MiniSweDispatch, {})
except ImportError:
    pass

try:
    from harbor.agents.installed.goose import Goose as _Goose
    class GooseDispatch(_Goose):
        def __init__(self, *args, api_base: str | None = None, **kwargs):
            kwargs.pop("harness", None)
            extra = kwargs.setdefault("extra_env", {})
            if api_base:
                host = api_base.rsplit("/v1", 1)[0]
                extra.setdefault("OPENAI_HOST", host)
                extra.setdefault("OPENAI_BASE_PATH", "v1/chat/completions")
            super().__init__(*args, **kwargs)
    HARNESS_REGISTRY["goose"] = (GooseDispatch, {})
except ImportError:
    pass

class DispatchAgent:
    """Instantiates the registered backend for kwargs['harness'] and proxies to it."""

    def __init__(self, logs_dir=None, model_name: str | None = None, *, harness: str, **kwargs):
        if harness not in HARNESS_REGISTRY:
            raise KeyError(f"Unknown rg harness {harness!r}. Known: {sorted(HARNESS_REGISTRY)}.")
        backend_cls, presets = HARNESS_REGISTRY[harness]
        merged = {**presets, **kwargs}
        self._backend = backend_cls(logs_dir=logs_dir, model_name=model_name, **merged)
        self._harness = harness

    def __getattr__(self, item: str) -> Any:
        return getattr(self._backend, item)

    @staticmethod
    def name() -> str:
        return "rg-dispatch"

    def version(self) -> str | None:
        return getattr(self._backend, "version", lambda: None)()

    async def run(self, *args, **kwargs):
        return await self._backend.run(*args, **kwargs)

    async def setup(self, *args, **kwargs):
        return await self._backend.setup(*args, **kwargs)
