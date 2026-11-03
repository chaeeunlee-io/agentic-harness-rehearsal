"""Single-turn runner for the BFCL fc, json, and xml harnesses.

Runs inside the task sandbox: make one model call, parse the response, and write
`/app/result.json`.

For distillation, the saved transcript keeps the model's literal `raw_response`,
including native tool calls for `fc`, separately from the normalized `answer`
used by the verifier. Training should use `raw_response`, not `answer`, so the
harness-specific output format is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from litellm import completion

import bfcl_protocols as proto

RESULT_PATH = Path("/app/result.json")


def _parse_message(response: Any) -> dict[str, Any]:
    choices = response["choices"]
    if not choices:
        raise ValueError("LLM response has no choices")
    msg = choices[0]["message"]
    return msg if isinstance(msg, dict) else dict(msg)


def run(
    *,
    model: str,
    harness: str,
    instruction: str,
    temperature: float,
    seed: int | None,
    llm_args_json: str | None,
) -> dict[str, Any]:
    llm_args: dict[str, Any] = {}
    if llm_args_json:
        llm_args.update(json.loads(llm_args_json))
    llm_args.setdefault("temperature", temperature)
    if seed is not None:
        llm_args.setdefault("seed", seed)

    query, funcs = proto.parse_instruction(instruction)

    if harness == "fc":
        tools, name_map = proto.to_openai_tools(funcs)
        messages = proto.build_fc_messages(query)
        kwargs: dict[str, Any] = dict(llm_args)
        if tools:
            kwargs["tools"] = tools
            # NEVER "required": it forces a call even on irrelevance tasks
            # (31.3% of this corpus is irrelevance/relevance, where the correct
            # answer is []), and guided decoding under `required` also
            # suppresses <think> for a reasoning student -- incident #2 in
            # docs/eidf/pipeline-incidents-and-rerun-audit.md.
            kwargs["tool_choice"] = "auto"
        response = completion(model=model, messages=messages, **kwargs)
        message = _parse_message(response)
        answer = proto.parse_fc_response(message, name_map)
        raw_response = {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
    else:
        spec = proto.PROTOCOLS[harness]
        messages = spec["build_messages"](query, funcs)
        response = completion(model=model, messages=messages, **llm_args)
        message = _parse_message(response)
        text = message.get("content") or ""
        answer = spec["parse_response"](text)
        raw_response = {"content": text, "tool_calls": None}

    usage = response.get("usage") or {}
    return {
        "answer": answer,
        "raw_response": raw_response,
        "messages": messages,
        "harness": harness,
        "model": model,
        "n_input_tokens": int(usage.get("prompt_tokens") or 0),
        "n_output_tokens": int(usage.get("completion_tokens") or 0),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", required=True, choices=sorted(proto.PROTOCOLS))
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed", type=lambda v: None if v == "none" else int(v), default=None
    )
    parser.add_argument("--llm-args-json", default=None)
    parser.add_argument(
        "--summary-path", type=Path, default=Path("/logs/agent/bfcl-agent-summary.json")
    )
    parser.add_argument(
        "--debug-path", type=Path, default=Path("/logs/agent/bfcl-agent-transcript.json")
    )
    args = parser.parse_args()

    instruction = args.instruction_file.read_text(encoding="utf-8")

    try:
        result = run(
            model=args.model,
            harness=args.harness,
            instruction=instruction,
            temperature=args.temperature,
            seed=args.seed,
            llm_args_json=args.llm_args_json,
        )
    except Exception as exc:
        # Still write an (empty) result.json so the verifier scores a real 0
        # rather than erroring on a missing file -- keeps "model got it wrong"
        # and "infrastructure broke" distinguishable in the trial record.
        _write_json(RESULT_PATH, [])
        _write_json(
            args.summary_path,
            {
                "n_input_tokens": 0,
                "n_output_tokens": 0,
                "cost_usd": 0.0,
                "metadata": {
                    "agent": "bfcl-agent",
                    "harness": args.harness,
                    "model": args.model,
                    "stop_reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            },
        )
        raise

    _write_json(RESULT_PATH, result["answer"])
    _write_json(
        args.summary_path,
        {
            "n_input_tokens": result["n_input_tokens"],
            "n_output_tokens": result["n_output_tokens"],
            "cost_usd": 0.0,
            "metadata": {
                "agent": "bfcl-agent",
                "harness": result["harness"],
                "model": result["model"],
                "stop_reason": "completed",
                "n_calls_emitted": len(result["answer"]),
            },
        },
    )
    _write_json(args.debug_path, result)
    print(f"[bfcl-{args.harness}] wrote {len(result['answer'])} call(s) to {RESULT_PATH}")


if __name__ == "__main__":
    main()
