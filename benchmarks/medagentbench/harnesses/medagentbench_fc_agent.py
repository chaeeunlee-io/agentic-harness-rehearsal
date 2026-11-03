"""MedAgentBench harness using native OpenAI tool calls.

Uses native `get`, `post`, and `finish` tool calls while keeping the same task
semantics and verifier behavior as the parity harness.

Runs entirely on the harness side. Registry key: `"fc"`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import requests

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

try:
    from benchmarks.medagentbench.harnesses.medagentbench_parity_agent import (
        DEFAULT_POST_ACK,
        MEDAGENTBENCH_PROMPT,
        MedAgentBenchParityAgent,
    )
except ImportError:  # SkyRL cwd fallback (same pattern as dispatch/trainable)
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _spec = importlib.util.spec_from_file_location(
        "medagentbench_parity_agent",
        _Path(__file__).parent / "medagentbench_parity_agent.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("medagentbench_parity_agent", _mod)
    _spec.loader.exec_module(_mod)
    DEFAULT_POST_ACK = _mod.DEFAULT_POST_ACK
    MEDAGENTBENCH_PROMPT = _mod.MEDAGENTBENCH_PROMPT
    MedAgentBenchParityAgent = _mod.MedAgentBenchParityAgent

# Tool-native prompt (2026-09-07). The parity template was reused here originally,
# which instructs text-format GET/POST/FINISH — contradicting the tool-call modality.
# With tool_choice=required the contradiction was masked (guided decoding forced tools);
# with tool_choice=auto (needed so <think> reasoning can flow), near-identical prompts
# with conflicting target formats caused cross-harness format interference in SFT
# (v3 pool: parity 66->46). fc now has a DISTINCT, self-consistent interface signature.
FC_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. \
You are given a question and access to tools for querying and updating a FHIR server. \
Use the provided tools to accomplish the task:

- Call the `get` tool to retrieve FHIR resources (pass the full request URL).
- Call the `post` tool to create or update FHIR resources (pass the URL and JSON payload).
- When you have completed all requested tasks, call the `finish` tool with your final \
answer(s) as a JSON-loadable list.

Think through what the question requires before each tool call. Make exactly one tool \
call per turn. Note that you should use {api_base} as the api_base.

Reference catalog of the underlying FHIR operations available through the tools:
{functions}

Context: {context}
Question: {question}"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get",
            "description": "Send a GET request to the FHIR API. Returns the response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full FHIR URL including query parameters."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post",
            "description": "Send a POST request to the FHIR API with a JSON payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full FHIR URL to POST to."},
                    "payload": {"type": "object", "description": "JSON body of the POST."},
                },
                "required": ["url", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Final answer, JSON-encoded when the task asks for structured output.",
                    }
                },
                "required": ["result"],
            },
        },
    },
]


class MedAgentBenchFCAgent(MedAgentBenchParityAgent):
    """Native function-calling MedAgentBench harness (dialect axis)."""

    @staticmethod
    def name() -> str:
        return "medagentbench-fc"

    def version(self) -> str | None:
        return "fc-v1"

    # ------------------------------------------------------------------
    def _infer_fc(self, *, model_name: str, api_key: str,
                  history: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self._openai_base}/chat/completions"
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}
        body = {
            "model": model_name,
            "messages": history,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=body,
                                     timeout=self._request_timeout_sec)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 2:
                    time.sleep(attempt + 2)
        raise RuntimeError(f"FC inference failed after retries: {last_error}")

    # ------------------------------------------------------------------
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for MedAgentBenchFCAgent")
        if not self.model_name:
            raise RuntimeError("model_name is required for MedAgentBenchFCAgent")
        model_name = (self.model_name.split("/", 1)[1]
                      if "/" in self.model_name else self.model_name)

        metadata = await self._load_task_metadata(environment)
        verifier_fhir_base = self._normalize_fhir_base(metadata.get("fhir_api_base"))
        original_instruction, task_context = self._parse_harbor_instruction(instruction)
        prompt = FC_PROMPT.format(
            api_base="http://localhost:8080/fhir/",
            functions=json.dumps(self._funcs),
            context=task_context,
            question=original_instruction,
        )

        history: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        # parity-shaped transcript so _build_answer_payload works unchanged
        transcript: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        finish_result: str | None = None

        for _ in range(self._max_rounds):
            msg = await asyncio.to_thread(
                self._infer_fc, model_name=model_name, api_key=api_key,
                history=history)
            tool_calls = msg.get("tool_calls") or []
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
            history.append(assistant_entry)

            if not tool_calls:
                history.append({"role": "user",
                                "content": "You must call one of the tools: get, post, finish."})
                continue

            done = False
            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name")
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                except Exception:
                    args = {}
                tc_id = tc.get("id") or "call_0"

                if fn == "get":
                    url_arg = str(args.get("url", ""))
                    resp = await self._send_get_request(environment, url_arg)
                    content = json.dumps(resp)
                    transcript.append({"role": "agent", "content": f"GET {url_arg}"})
                    transcript.append({"role": "user", "content": content})
                elif fn == "post":
                    url_arg = str(args.get("url", ""))
                    payload = args.get("payload")
                    # mirror parity semantics: accumulate + ACK (verifier replays)
                    transcript.append({
                        "role": "agent",
                        "content": "POST " + url_arg + "\n" + json.dumps(payload),
                    })
                    transcript.append({"role": "user", "content": DEFAULT_POST_ACK})
                    content = DEFAULT_POST_ACK
                elif fn == "finish":
                    finish_result = str(args.get("result", ""))
                    content = "Task finished."
                    done = True
                else:
                    content = f"Unknown tool: {fn}"
                history.append({"role": "tool", "tool_call_id": tc_id,
                                "content": content})
                if done:
                    break
            if done:
                break

        payload = self._build_answer_payload(
            finish_result=finish_result,
            transcript=transcript,
            verifier_fhir_base=verifier_fhir_base,
        )
        await self._write_answer_json(environment, payload)

        context.metadata = context.metadata or {}
        context.metadata["harness"] = "fc"
        # raw FC history for trace conversion / privilege bank
        try:
            if self.logs_dir is not None:
                from pathlib import Path
                p = Path(self.logs_dir) / "fc_history.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(history))
        except Exception:  # noqa: BLE001 — logging must never kill the trial
            pass
