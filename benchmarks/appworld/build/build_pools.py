"""Build AppWorld distillation datasets: 3h, code-only, fc-only, and mini-only.

Input: Qwen3-32B teacher traces from training tasks, with one job per harness
(`exp collect -b appworld`). Mini traces may come from multiple jobs; pass all
with `--job`.

Each solved trial (goal completion 1.0) whose trajectory contains the harness's
action form becomes a row, with at most two rows per task and 250 per harness.
3h combines all three harnesses; the other datasets contain one harness each.

Before writing, render every row with the student's chat template and verify
that renders are non-empty and each harness includes reasoning and its action.
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HARNESSES = ["code", "fc", "mini"]
JOBS = {h: f"teacher-{h}" for h in HARNESSES}
CAP = 250
PER_TASK = 2


def reward(trial):
    try:
        return float(open(f"{trial}/verifier/reward.txt").read().strip())
    except Exception:
        return 0.0


def chat_code_or_fc(trial):
    """Our runner's transcript: the system prompt plus the literal turn sequence the model saw and produced."""
    path = f"{trial}/agent/appworld-code-agent-transcript.json"
    if not os.path.exists(path):
        return None
    try:
        result = json.load(open(path)).get("result", {})
    except Exception:
        return None
    system, history = result.get("system_prompt") or "", result.get("history_messages") or []
    if not system or len(history) < 2:
        return None
    return [{"role": "system", "content": system}, *history]


def chat_mini(trial):
    """mini-swe-agent's own trajectory: system, task, then assistant actions and observations."""
    path = f"{trial}/agent/mini-swe-agent.trajectory.json"
    if not os.path.exists(path):
        return None
    try:
        msgs = json.load(open(path)).get("messages") or []
    except Exception:
        return None
    msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant")]
    return msgs if len(msgs) >= 3 else None


EXTRACT = {"code": chat_code_or_fc, "fc": chat_code_or_fc, "mini": chat_mini}


def has_action(harness, messages):
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if harness == "fc" and m.get("tool_calls"):
            return True
        if harness == "code" and "```python" in content:
            return True
        if harness == "mini" and ("mswea_bash_command" in content or "apis." in content):
            return True
    return False


def check(pool, counts, tokenizer_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    template = C.TEMPLATE.read_text()
    think, action, empty = dict.fromkeys(HARNESSES, 0), dict.fromkeys(HARNESSES, 0), 0
    for r in pool:
        text = tok.apply_chat_template(r["messages"], chat_template=template, tokenize=False)
        empty += len(text) <= 200
        think[r["meta"]["harness"]] += "<think>" in text
        action[r["meta"]["harness"]] += has_action(r["meta"]["harness"], r["messages"])
    print("rows with reasoning:", think, "| with an action:", action, "| empty renders:", empty)
    assert empty == 0, f"{empty} rows render empty"
    for h in HARNESSES:
        if counts[h] == 0:
            print(f"warning: no solved trajectories for {h}; it is left out of the pools")
            continue
        assert think[h] > 0, f"no {h} row has reasoning"
        assert action[h] > 0, f"no {h} row has an action"


def main():
    p = C.parser(__doc__, JOBS)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="the student's tokenizer, for the render check")
    args = p.parse_args()
    jobs = C.job_dirs(args, JOBS)
    rng = random.Random(0)
    pool, counts = [], {}
    for h in HARNESSES:
        rows, per_task = [], {}
        for trial in C.trials(jobs[h], "*__*"):
            if reward(trial) < 1.0:
                continue
            task = os.path.basename(trial).split("__")[0]
            if per_task.get(task, 0) >= PER_TASK:
                continue
            msgs = EXTRACT[h](trial)
            if not msgs or not has_action(h, msgs):
                continue
            rows.append({"messages": msgs, "meta": {"harness": h, "task": task}})
            per_task[task] = per_task.get(task, 0) + 1
        counts[h] = len(rows)
        pool += rows if len(rows) <= CAP else rng.sample(rows, CAP)
    print("solved trajectories per harness:", counts)
    if not pool:
        raise SystemExit("empty pool: no solved trajectories")
    check(pool, counts, args.tokenizer)
    rng.shuffle(pool)
    C.write_pool(args.out_dir, "3h", pool)
    for h in HARNESSES:
        C.write_pool(args.out_dir, f"{h}-only", [r for r in pool if r["meta"]["harness"] == h])


if __name__ == "__main__":
    main()
