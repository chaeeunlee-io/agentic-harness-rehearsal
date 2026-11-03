"""Multi-turn AppWorld runner for the code, fc, and xml harnesses.

The runner repeatedly calls the model, extracts Python, executes it against the
persistent AppWorld world, and returns the result until completion or a budget
limit. Harness-specific prompts, formats, and trace recording live in
`appworld_protocols.py`.

The model endpoint is called with `requests` rather than LiteLLM because
AppWorld uses pydantic v1. Final state is written with `world.save_logs()` for
the verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

import appworld_protocols as protocols

DEFAULT_MAX_STEPS = 40
DEFAULT_MAX_ERRORS = 8
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 4  # transient 5xx / connection / read-timeout only
DEFAULT_HARNESS = "code"

_LITELLM_PREFIXES = ("hosted_vllm/", "openai/")


def _resolve_model_name(model: str) -> str:
    """Strip litellm provider prefixes -- vLLM's OpenAI endpoint wants the
    bare served-model name (hosted_vllm/Qwen3-4B-base -> Qwen3-4B-base)."""
    for prefix in _LITELLM_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _resolve_api_base(cli_api_base: str | None) -> str:
    base = (
        cli_api_base
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if not base:
        raise ValueError(
            "No api_base: pass --api-base or set OPENAI_API_BASE / OPENAI_BASE_URL"
        )
    return base.rstrip("/")


@dataclass
class AppWorldRunResult:
    harness: str
    system_prompt: str
    history_messages: list[dict[str, Any]]
    turns_completed: int
    stop_reason: str
    num_errors: int
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    cost_usd: float = 0.0


def _chat_completion(
    *,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    **llm_args: Any,
) -> dict[str, Any]:
    """POST to the vLLM OpenAI-compatible endpoint, retrying ONLY transient
    faults (connection errors, read timeouts, 5xx). litellm used to retry
    implicitly; raw requests does not, and a transient 503 twenty turns into an
    episode would otherwise throw away the most expensive part of a trial. A
    4xx (e.g. malformed request) is a real bug -- surface it immediately, don't
    retry. Backoff is 1s, 2s, 4s (capped)."""
    url = f"{api_base}/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages}
    payload.update(llm_args)
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    headers = {"Authorization": f"Bearer {api_key}"}

    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT
            )
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                resp.raise_for_status()  # 4xx -> raise now, no retry
                return resp.json()
        except requests.exceptions.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(
        f"chat completion failed after {max_retries} attempts: {last_err}"
    )


def _usage(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


_NO_ACTION_FEEDBACK = {
    "code": "No ```python code block found. Reply with exactly one fenced ```python block.",
    "fc": "You made no execute_python tool call. Call execute_python with your code this turn.",
    "xml": "No <code>...</code> block found. Wrap your Python in <code></code> tags.",
}


def run_appworld_task(
    *,
    model: str,
    api_base: str,
    harness: str,
    instruction: str,
    task_id: str,
    experiment_name: str,
    max_steps: int,
    max_errors: int,
    temperature: float,
    seed: int | None,
    llm_args_json: str | None,
) -> AppWorldRunResult:
    from appworld import AppWorld

    proto = protocols.get_protocol(harness)
    served_model = _resolve_model_name(model)

    llm_args: dict[str, Any] = {}
    if llm_args_json:
        llm_args.update(json.loads(llm_args_json))
    llm_args.setdefault("temperature", temperature)
    llm_args.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    if seed is not None:
        llm_args.setdefault("seed", seed)
    request_extra = proto.request_extra()

    system_prompt = proto.system_prompt
    history: list[dict[str, Any]] = [
        {"role": "user", "content": f"Task: {instruction}"}
    ]

    total_in = total_out = 0
    num_errors = 0
    stop_reason = "max_steps"
    turns_completed = 0

    world = AppWorld(task_id=task_id, experiment_name=experiment_name)
    try:
        for turns_completed in range(1, max_steps + 1):
            response = _chat_completion(
                api_base=api_base,
                model=served_model,
                messages=[{"role": "system", "content": system_prompt}, *history],
                **request_extra,
                **llm_args,
            )
            message = response["choices"][0]["message"]
            in_tok, out_tok = _usage(response)
            total_in += in_tok
            total_out += out_tok

            code = proto.extract_code(message)
            proto.record_assistant(history, message)

            if code is None:
                num_errors += 1
                history.append({"role": "user", "content": _NO_ACTION_FEEDBACK[harness]})
                if num_errors >= max_errors:
                    stop_reason = "too_many_errors"
                    break
                continue

            output = world.execute(code)
            proto.record_observation(history, message, output)
            if "Execution failed. Traceback:" in output:
                num_errors += 1
                if num_errors >= max_errors:
                    stop_reason = "too_many_errors"
                    break

            if world.task_completed():
                stop_reason = "task_completed"
                break
        else:
            stop_reason = "max_steps"
    finally:
        world.save_logs()
        world.close()

    return AppWorldRunResult(
        harness=harness,
        system_prompt=system_prompt,
        history_messages=history,
        turns_completed=turns_completed,
        stop_reason=stop_reason,
        num_errors=num_errors,
        n_input_tokens=total_in,
        n_output_tokens=total_out,
        cost_usd=0.0,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--harness", default=DEFAULT_HARNESS, choices=sorted(protocols.PROTOCOLS))
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed", type=lambda v: None if v == "none" else int(v), default=None
    )
    parser.add_argument("--llm-args-json", default=None)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("/logs/agent/appworld-code-agent-summary.json"),
    )
    parser.add_argument(
        "--debug-path",
        type=Path,
        default=Path("/logs/agent/appworld-code-agent-transcript.json"),
    )
    args = parser.parse_args()

    task_id = os.environ["APPWORLD_TASK_ID"]
    experiment_name = os.environ.get("APPWORLD_EXPERIMENT_NAME", "harbor_trial")
    instruction = args.instruction_file.read_text(encoding="utf-8")
    api_base = _resolve_api_base(args.api_base)

    try:
        result = run_appworld_task(
            model=args.model,
            api_base=api_base,
            harness=args.harness,
            instruction=instruction,
            task_id=task_id,
            experiment_name=experiment_name,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            temperature=args.temperature,
            seed=args.seed,
            llm_args_json=args.llm_args_json,
        )
    except Exception as exc:
        _write_json(
            args.summary_path,
            {
                "n_input_tokens": 0,
                "n_output_tokens": 0,
                "n_cache_tokens": 0,
                "cost_usd": 0.0,
                "metadata": {
                    "agent": "appworld-code-agent",
                    "model": args.model,
                    "harness": args.harness,
                    "task_id": task_id,
                    "stop_reason": "error",
                    "error": str(exc),
                },
            },
        )
        raise

    _write_json(
        args.summary_path,
        {
            "n_input_tokens": result.n_input_tokens,
            "n_output_tokens": result.n_output_tokens,
            "n_cache_tokens": result.n_cache_tokens,
            "cost_usd": result.cost_usd,
            "metadata": {
                "agent": "appworld-code-agent",
                "model": args.model,
                "harness": result.harness,
                "task_id": task_id,
                "turns_completed": result.turns_completed,
                "stop_reason": result.stop_reason,
                "num_errors": result.num_errors,
            },
        },
    )
    _write_json(args.debug_path, {"model": args.model, "harness": result.harness, "result": asdict(result)})


if __name__ == "__main__":
    main()
