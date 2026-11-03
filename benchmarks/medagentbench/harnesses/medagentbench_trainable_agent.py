"""Trainable version of the MedAgentBench parity agent.

Uses the same prompt, GET/POST/FINISH protocol, FHIR handling, and `answer.json`
format as `MedAgentBenchParityAgent`, but routes model calls through Harbor's
instrumented `LiteLLM` and `Chat` stack so SkyRL can collect rollout details.

The parity agent exposes `_call_model` as the model-call boundary. This subclass
overrides that method and adds rollout details to the `AgentContext`; all other
behavior is inherited.

Harbor:
    harbor run -p <task> -e <backend> \
      --agent-import-path benchmarks.medagentbench.harnesses.medagentbench_trainable_agent:MedAgentBenchTrainableAgent \
      -m hosted_vllm/Qwen3-4B --ak api_base=http://<vllm>/v1

SkyRL uses the same import path in `harbor_trial_config.agent.import_path`.
"""

from __future__ import annotations

from typing import Any

from harbor.llms.base import OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

try:  # dotted import when run from the harbor repo root (harbor CLI convention)
    from benchmarks.medagentbench.harnesses.medagentbench_parity_agent import (
        MedAgentBenchParityAgent,
    )
except ImportError:  # e.g. SkyRL cwd — load the sibling file directly
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _spec = importlib.util.spec_from_file_location(
        "medagentbench_parity_agent",
        _Path(__file__).parent / "medagentbench_parity_agent.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["medagentbench_parity_agent"] = _mod
    _spec.loader.exec_module(_mod)
    MedAgentBenchParityAgent = _mod.MedAgentBenchParityAgent


class MedAgentBenchTrainableAgent(MedAgentBenchParityAgent):
    INVALID_ACTION_FEEDBACK = True  # up to two corrective messages; see the parity agent
    def __init__(
        self,
        *args,
        session_id: str | None = None,
        collect_rollout_details: bool = True,
        model_info: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._session_id = session_id
        self._collect_rollout_details = collect_rollout_details
        self._model_info = model_info
        self._extra_llm_kwargs = llm_kwargs or {}
        self._chat: Chat | None = None

    @staticmethod
    def name() -> str:
        return "medagentbench-trainable-agent"

    def version(self) -> str | None:
        return "1.0.0"

    def _ensure_chat(self) -> Chat:
        if self._chat is None:
            llm = LiteLLM(
                model_name=self.model_name,
                temperature=self._temperature,
                api_base=self._openai_base,
                session_id=self._session_id,
                collect_rollout_details=self._collect_rollout_details,
                model_info=self._model_info,
                max_tokens=self._max_tokens,
                **self._extra_llm_kwargs,
            )
            self._chat = Chat(llm, record_truncated_turns=True)
        return self._chat

    async def _call_model(
        self,
        *,
        model_name: str,
        api_key: str,
        history: list[dict[str, str]],
    ) -> tuple[str, dict[str, int]]:
        # The base loop appends to `history` and calls this once per round, so the
        # newest entry is always the next user prompt; Chat keeps the same
        # conversation state internally and accumulates rollout details per call.
        if not history or history[-1]["role"] != "user":
            raise RuntimeError(
                "trainable agent expects history to end with a user message, "
                f"got: {history[-1]['role'] if history else 'empty'}"
            )
        chat = self._ensure_chat()
        try:
            llm_response = await chat.chat(history[-1]["content"])
        except OutputLengthExceededError as exc:
            # Reply cut off at max_tokens: continue with the truncated text instead
            # of failing the trial. Chat has already recorded the turn (tokens +
            # history), so training keeps the completion; downstream parsing
            # treats an incomplete action as a parse failure and feeds back —
            # burning the budget mid-thought becomes learnable behaviour, not a
            # crashed rollout.
            if exc.llm_response is None:
                raise
            llm_response = exc.llm_response
        usage = llm_response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cached_tokens": usage.cache_tokens if usage else 0,
        }
        # Thinking capture (RSFT): base loop stores this on the transcript turn.
        self._last_reasoning = llm_response.reasoning_content
        return llm_response.content or "", usage_dict

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await super().run(instruction, environment, context)
        if self._chat is not None:
            context.rollout_details = self._chat.rollout_details
            context.metadata = context.metadata or {}
            context.metadata["n_episodes"] = 1
            details = self._chat.rollout_details
            context.metadata["rollout_turns"] = (
                len(details[0].get("completion_token_ids", [])) if details else 0
            )
