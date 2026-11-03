"""Held-out XML harness for MedAgentBench.

Uses the same task semantics, actions, budgets, FHIR handling, and verifier
contract as parity, but represents GET, POST, and FINISH actions with XML tags.

Evaluation only; do not use this harness for training or RL rollouts.

`XML_PROMPT` remains XML-specific. Successful POSTs are stored in parity-shaped
transcript form because `_build_answer_payload()` reconstructs POST history from it.

Registry key: `"xml"`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.harness import fhir as harness_fhir
from harbor.models.agent.context import AgentContext

try:  # dotted import from the harbor repo root
    from benchmarks.medagentbench.harnesses.medagentbench_trainable_agent import (
        MedAgentBenchTrainableAgent,
    )
except ImportError:  # SkyRL cwd — sibling-file fallback (same pattern as fc/dispatch)
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


XML_PROMPT = """You are an expert in using FHIR functions to assist medical \
professionals. You are given a question and a set of possible functions. Based on \
the question, you will need to make one or more function calls to achieve the purpose.

You interact by emitting XML action tags. Exactly one action tag per response.

1. To issue a GET request:
<get><url>url?param_name1=param_value1&param_name2=param_value2</url></get>

2. To issue a POST request:
<post><url>url</url><payload>{{"key": "value"}}</payload></post>
The payload must be valid JSON.

3. When you have answered all the questions and finished all the requested tasks:
<finish><answer>[answer1, answer2, ...]</answer></finish>
The answer must be a JSON-loadable list.

Your response must contain exactly one of these three action tags and no other \
text. Note that you should use {api_base} as the api_base.

Here is a list of functions in JSON format that describe the underlying FHIR \
operations available to you.
{functions}

Context: {context}
Question: {question}"""

# Must retain the literal "POST request accepted" prefix: _build_answer_payload
# gates on that substring when reconstructing the verifier's POST history.
XML_POST_ACK = (
    "POST request accepted and executed successfully. Emit "
    "<finish><answer>[...]</answer></finish> if you have got answers for all the "
    "questions and finished all the requested tasks"
)

_GET_RE = re.compile(r"<get>\s*(?:<url>)?\s*(.*?)\s*(?:</url>)?\s*</get>", re.DOTALL | re.IGNORECASE)
_POST_RE = re.compile(
    r"<post>\s*<url>\s*(.*?)\s*</url>\s*<payload>\s*(.*?)\s*</payload>\s*</post>",
    re.DOTALL | re.IGNORECASE,
)
_FINISH_RE = re.compile(
    r"<finish>\s*(?:<answer>)?\s*(.*?)\s*(?:</answer>)?\s*</finish>", re.DOTALL | re.IGNORECASE
)


class MedAgentBenchXMLAgent(MedAgentBenchTrainableAgent):
    """XML-tagged-action MedAgentBench harness (held-out format rung)."""

    @staticmethod
    def name() -> str:
        return "medagentbench-xml"

    def version(self) -> str | None:
        return "xml-v1"

    # -- pure parsing helpers --
    @staticmethod
    def parse_action(text: str) -> tuple[str, Any]:
        """Return (kind, payload) for the first well-formed action tag.

        kind is one of "get" | "post" | "finish" | "none".
        payload is str (get: url; finish: answer text) or tuple[str, str] (post).
        """
        m = _POST_RE.search(text)
        if m is not None:
            return "post", (m.group(1).strip(), m.group(2).strip())
        m = _FINISH_RE.search(text)
        if m is not None:
            return "finish", m.group(1).strip()
        m = _GET_RE.search(text)
        if m is not None:
            return "get", m.group(1).strip()
        return "none", ""

    @staticmethod
    def ensure_json_format(url: str) -> str:
        """Mirror parity: force _format=json so responses are parseable."""
        if re.search(r"(?:\?|&)_format=", url) is None:
            url += ("&" if "?" in url else "?") + "_format=json"
        return url

    # ------------------------------------------------------------------
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for MedAgentBenchXMLAgent")
        if not self.model_name:
            raise RuntimeError("model_name is required for MedAgentBenchXMLAgent")
        model_name = (
            self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name
        )

        metadata = await self._load_task_metadata(environment)
        verifier_fhir_base = self._normalize_fhir_base(metadata.get("fhir_api_base"))
        harness_arm = await self._load_harness_arm(environment)

        original_instruction, task_context = self._parse_harbor_instruction(instruction)
        if not original_instruction.strip():
            # The 2026-09 drill bug shipped 413 trials with an empty question
            # because a body lacked the '## Original Instruction' header. Fail
            # loudly instead of scoring a whole column of empty prompts.
            raise RuntimeError(
                "MedAgentBenchXMLAgent: empty question after parsing harbor "
                "instruction; expected '## Original Instruction' / '## Context' "
                "headers."
            )
        prompt = XML_PROMPT.format(
            api_base="http://localhost:8080/fhir/",
            functions=json.dumps(self._funcs),
            context=task_context,
            question=original_instruction,
        )

        llm_history: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        transcript: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        finish_result: str | None = None
        final_status = "task_limit_reached"
        invalid_streak = 0
        harness_log: list[dict[str, Any]] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_tokens = 0

        for _ in range(self._max_rounds):
            agent_text, usage = await self._call_model(
                model_name=model_name, api_key=api_key, history=llm_history
            )
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
            total_cache_tokens += usage.get("cached_tokens", 0)

            cleaned = agent_text.strip().replace("```xml", "").replace("```", "").strip()

            agent_entry: dict[str, str] = {"role": "agent", "content": cleaned}
            reasoning = getattr(self, "_last_reasoning", None)
            if reasoning:
                agent_entry["reasoning"] = reasoning
            llm_history.append({"role": "assistant", "content": cleaned})

            kind, parsed = self.parse_action(cleaned)

            if kind == "get":
                transcript.append(agent_entry)
                get_url = self.ensure_json_format(str(parsed))
                plan = harness_fhir.apply_request(get_url, harness_arm)
                if plan.synthetic_response is not None:
                    get_res: dict[str, str] = {
                        "status_code": "200",
                        "data": plan.synthetic_response,
                    }
                else:
                    get_res = await self._send_get_request(environment, plan.url)
                harness_log.append(
                    {
                        "turn": len(harness_log) + 1,
                        "model_url": get_url,
                        "forwarded_url": plan.url,
                        "synthetic": plan.synthetic_response is not None,
                        "chars_in": len(get_res["data"]) if "data" in get_res else None,
                    }
                )
                if "data" in get_res:
                    response_data = harness_fhir.apply_response(get_res["data"], harness_arm)
                    harness_log[-1]["chars_out"] = len(response_data)
                    user_message = (
                        "Here is the response from the GET request:\n"
                        f"{response_data}. "
                        "Emit <finish><answer>[...]</answer></finish> if you have got "
                        "answers for all the questions and finished all the requested tasks"
                    )
                    transcript_user_message = self._truncate_for_logs(user_message)
                else:
                    user_message = f"Error in sending the GET request: {get_res['error']}"
                    transcript_user_message = user_message

                invalid_streak = 0
                transcript.append({"role": "user", "content": transcript_user_message})
                llm_history.append({"role": "user", "content": user_message})
                continue

            if kind == "post":
                post_url, raw_payload = parsed
                try:
                    parsed_payload = json.loads(raw_payload)
                except Exception:
                    user_message = "Invalid POST request"
                    transcript.append(agent_entry)
                else:
                    user_message = XML_POST_ACK
                    # PARITY-SHAPED transcript entry — see module docstring §2.
                    transcript.append(
                        {
                            "role": "agent",
                            "content": "POST " + post_url + "\n" + json.dumps(parsed_payload),
                        }
                    )

                invalid_streak = 0
                transcript.append({"role": "user", "content": user_message})
                llm_history.append({"role": "user", "content": user_message})
                continue

            if kind == "finish":
                transcript.append(agent_entry)
                finish_result = str(parsed)
                final_status = "completed"
                break

            transcript.append(agent_entry)
            if self.INVALID_ACTION_FEEDBACK and invalid_streak < 2:
                invalid_streak += 1
                feedback = (
                    "Your reply contained no action tag. Reply with exactly one of: "
                    "<get><url>...</url></get>, "
                    "<post><url>...</url><payload>{...}</payload></post>, or "
                    "<finish><answer>[...]</answer></finish>. "
                    "Do not reply with anything else."
                )
                transcript.append({"role": "user", "content": feedback})
                llm_history.append({"role": "user", "content": feedback})
                continue
            final_status = "invalid_action"
            break

        answer_payload = self._build_answer_payload(
            finish_result=finish_result,
            transcript=transcript,
            verifier_fhir_base=verifier_fhir_base,
        )
        await self._write_answer_json(environment, answer_payload)
        self._write_debug_log(
            transcript=transcript,
            finish_result=finish_result,
            answer_payload=answer_payload,
            status=final_status,
            harness_arm=harness_arm,
            harness_log=harness_log,
        )

        context.n_input_tokens = total_input_tokens
        context.n_output_tokens = total_output_tokens
        context.n_cache_tokens = total_cache_tokens
        context.metadata = {
            "status": final_status,
            "max_rounds": self._max_rounds,
            "openai_base_url": self._openai_base,
            "harness": "xml",
            "harness_arm": harness_arm.name,
            "harness_knobs": harness_arm.non_stock_knobs(),
        }
        if self._chat is not None:
            context.rollout_details = self._chat.rollout_details
            context.metadata["n_episodes"] = 1
            details = self._chat.rollout_details
            context.metadata["rollout_turns"] = (
                len(details[0].get("completion_token_ids", [])) if details else 0
            )
