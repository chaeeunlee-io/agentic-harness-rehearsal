"""Build BFCL multi-turn distillation datasets: fc-only, json-only, pycall-only, and 3h.

Input: Qwen3-32B teacher traces from the training split, with one or more jobs
per harness passed through `--job`. Each solved trial becomes one row;
`agent/mt_transcript.json` already contains the OpenAI-style chat and is only
cleaned here. `3h` combines all three harnesses.

`--max-tokens N` drops rows that exceed N tokens under the student's chat
template, avoiding mid-episode truncation during training.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HARNESSES = ["fc", "json", "pycall"]
JOBS = {h: f"teacher-{h}" for h in HARNESSES}
AS_RUN = {"pycall": "code"}  # the pycall harness is called "code" in as-run names and in meta


def clean(messages):
    """Keep only the fields the chat template reads; tool-call arguments as strings."""
    out = []
    for m in messages:
        row = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_call_id"):
            row["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            row["tool_calls"] = []
            for i, call in enumerate(m["tool_calls"]):
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args) if args else "{}"
                row["tool_calls"].append({"id": call.get("id") or f"call_{i}", "type": "function",
                                          "function": {"name": fn.get("name") or "", "arguments": args}})
        out.append(row)
    return out


def solved(trial):
    try:
        return open(os.path.join(trial, "verifier", "reward.txt")).read().strip() in ("1", "1.0")
    except OSError:
        return False


def too_long(rows, max_tokens, tokenizer):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    template = C.TEMPLATE.read_text()
    return {i for i, r in enumerate(rows)
            if len(tok(tok.apply_chat_template(r["messages"], chat_template=template, tokenize=False)).input_ids) > max_tokens}


def main():
    p = C.parser(__doc__, JOBS)
    p.add_argument("--max-tokens", type=int, help="drop rows longer than this (off by default, as shipped)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="for --max-tokens")
    args = p.parse_args()
    jobs = C.job_dirs(args, JOBS)
    mix = []
    for h in HARNESSES:
        rows = []
        for trial in C.trials(jobs[h], "bfclmt-*/"):
            if not solved(trial):
                continue
            path = os.path.join(trial, "agent", "mt_transcript.json")
            if not os.path.exists(path):
                print(f"warning: solved trial has no transcript: {trial}", file=sys.stderr)
                continue
            msgs = clean(json.load(open(path))["messages"])
            if any(m["role"] == "assistant" for m in msgs):
                task = os.path.basename(trial.rstrip("/")).split("__")[0]
                rows.append({"messages": msgs, "meta": {"harness": f"mt-{AS_RUN.get(h, h)}", "task": task}})
        if args.max_tokens:
            drop = too_long(rows, args.max_tokens, args.tokenizer)
            rows = [r for i, r in enumerate(rows) if i not in drop]
            print(f"{h}: dropped {len(drop)} rows over {args.max_tokens} tokens")
        C.write_pool(args.out_dir, f"{h}-only", rows)
        mix += rows
    C.write_pool(args.out_dir, "3h", mix)


if __name__ == "__main__":
    main()
