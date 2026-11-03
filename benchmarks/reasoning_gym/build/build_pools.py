"""Build Reasoning Gym distillation datasets: 3h, t2-only, miniswe-only, and goose-only.

Input: Qwen3-32B teacher traces from both training tiers, one job per
`(tier, harness)`. Each solved trajectory (reward 1) becomes a row, with at most
two rows per task and tier and 250 rows per harness. `3h` combines all three
harnesses.

Each harness has its own trace reader. Goose logs omit the system prompt, so its
rows add the prompt from `goose_system_prompt.json` together with the task
instruction.

Before writing, rows are checked with the student's chat template for non-empty
renders, reasoning, tool calls in mini-swe-agent rows, and the Goose system prompt.
"""
import json
import os
import random
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HARNESSES = ["t2", "miniswe", "goose"]
TIERS = ["easy", "hard"]
JOBS = {f"{tier}-{h}": f"teacher-{tier}-{h}" for h in HARNESSES for tier in TIERS}
CAP = 250  # per harness, both tiers together
GOOSE_PROMPT = Path(__file__).resolve().parent / "goose_system_prompt.json"
TEMPLATE = C.REPO / "training" / "templates" / "qwen3_acc_thinking.jinja2"


def reward(trial):
    try:
        result = json.load(open(f"{trial}/result.json"))
        return ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0
    except Exception:
        return 0


def chat_miniswe(trial):
    """mini-swe-agent's own trajectory file (content plus tool calls)."""
    path = f"{trial}/agent/mini-swe-agent.trajectory.json"
    if not os.path.exists(path):
        return None
    try:
        log = json.load(open(path))
    except Exception:
        return None
    out = []
    for m in log.get("messages", []):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        row = {"role": role, "content": (str(m.get("content")) if m.get("content") is not None else "")[:6000]}
        if role == "assistant" and m.get("tool_calls"):
            row["tool_calls"] = m["tool_calls"]
        if role == "tool":
            row["tool_call_id"] = m.get("tool_call_id", "")
        out.append(row)
    return out if len(out) >= 3 else None


def chat_goose(trial, system_prompt):
    """Goose's event stream (agent/goose.txt), merged per message id into a chat."""
    path = f"{trial}/agent/goose.txt"
    if not os.path.exists(path):
        return None
    order, merged = [], {}
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line.startswith('{"type"'):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("type") != "message":
            continue
        m = event.get("message") or {}
        mid = m.get("id") or f"anon{len(order)}"
        if mid not in merged:
            merged[mid] = {"role": m.get("role", "assistant"), "think": "", "text": "", "calls": [], "results": []}
            order.append(mid)
        a = merged[mid]
        for part in m.get("content") or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "thinking":
                a["think"] += part.get("thinking", "")
            elif kind == "text":
                a["text"] += part.get("text", "")
            elif kind == "toolRequest":
                call = (part.get("toolCall") or {}).get("value", {})
                a["calls"].append({"id": part.get("id", ""), "type": "function",
                                   "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}))}})
            elif kind == "toolResponse":
                res = (part.get("toolResult") or {}).get("value", {})
                a["results"].append({"id": part.get("id", ""), "out": json.dumps(res)[:4000]})
    try:
        task_dir = json.load(open(f"{trial}/config.json"))["task"]["path"]
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": open(os.path.join(task_dir, "instruction.md")).read()}]
    except Exception:
        return None
    for mid in order:
        a = merged[mid]
        if a["role"] == "assistant":
            row = {"role": "assistant", "content": (f"<think>{a['think']}</think>" if a["think"].strip() else "") + a["text"]}
            if a["calls"]:
                row["tool_calls"] = a["calls"]
            msgs.append(row)
        elif a["results"]:
            msgs += [{"role": "tool", "tool_call_id": r["id"], "content": r["out"]} for r in a["results"]]
        elif a["text"]:
            msgs.append({"role": "user", "content": a["text"]})
    return msgs if len(msgs) >= 2 else None


def check(pool, tokenizer_path):
    """Render every row with the student's template and fail on the problems seen while building."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.chat_template = TEMPLATE.read_text()
    think = {h: 0 for h in HARNESSES}
    for r in pool:
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        assert len(text) > 200, "empty render"
        think[r["meta"]["harness"]] += "<think>" in text
    print("rows with reasoning:", think)
    assert all(think.values()), f"a harness has no reasoning at all: {think}"
    ms = [r for r in pool if r["meta"]["harness"] == "miniswe"]
    acting = sum(any(m.get("tool_calls") for m in r["messages"] if m["role"] == "assistant") for r in ms)
    assert acting >= 0.8 * len(ms), f"only {acting}/{len(ms)} mini-swe-agent rows have tool calls"
    gs = [r for r in pool if r["meta"]["harness"] == "goose"]
    prompted = sum(r["messages"][0]["role"] == "system" and "goose" in r["messages"][0]["content"][:200] for r in gs)
    assert prompted >= 0.95 * len(gs), f"only {prompted}/{len(gs)} goose rows have the goose system prompt"


def main():
    p = C.parser(__doc__, JOBS)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="the student's tokenizer, for the render check")
    p.add_argument("--min-trajectories", type=int, default=30,
                   help="stop if a harness has fewer solved trajectories than this (lower it only for small test runs)")
    args = p.parse_args()
    jobs = C.job_dirs(args, JOBS)
    system_prompt = json.load(open(GOOSE_PROMPT))["system"]
    extract = {"t2": C.chat_t2, "miniswe": chat_miniswe, "goose": lambda t: chat_goose(t, system_prompt)}

    rng = random.Random(0)
    pool, counts = [], {}
    for h in HARNESSES:
        rows = []
        for tier in TIERS:
            per_task = {}
            for trial in sorted(glob(f"{jobs[f'{tier}-{h}']}/*__*")):
                if reward(trial) < 1:
                    continue
                task = C.task_of(trial)
                if per_task.get(task, 0) >= C.PER_TASK:
                    continue
                msgs = extract[h](trial)
                if not msgs:
                    continue
                rows.append({"messages": msgs, "meta": {"harness": h, "tier": tier, "task": task}})
                per_task[task] = per_task.get(task, 0) + 1
        counts[h] = len(rows)
        pool += C.capped(rng, rows, CAP)
    print("solved trajectories per harness:", counts)
    if min(counts.values()) < args.min_trajectories:
        raise SystemExit(f"a harness has fewer than {args.min_trajectories} trajectories: {counts}")

    check(pool, args.tokenizer)
    rng.shuffle(pool)
    C.write_pool(args.out_dir, "3h", pool)
    for h in HARNESSES:
        C.write_pool(args.out_dir, f"{h}-only", [r for r in pool if r["meta"]["harness"] == h])


if __name__ == "__main__":
    main()
