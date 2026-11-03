"""Build BFCL multi-turn rehearsal datasets.

Input: rehearsal traces (a demonstrator run on the rehearsal tasks that
`gen_rehearsal.py` generates and `qc_rehearsal.py` checks), plus teacher
datasets from `build_pools.py`. Each solved rehearsal trial becomes a row, with
at most two rows per task.

For `--target fc`, builds:
- `json+pycall`
- `reh-fc`
- `json+pycall+reh-fc`

For `--target xml`, builds `3h+reh-xml`; `3h` is the matched control because it
contains no xml rows.

All messages use the same four fields (`role`, `content`, `tool_calls`,
`tool_call_id`) so mixed datasets share a consistent schema.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PER_TASK = 2


def solved_rehearsal(dirs, harness_label):
    rows, per_task = [], {}
    for trial in C.trials(dirs, "*/"):
        reward, transcript = os.path.join(trial, "verifier", "reward.txt"), os.path.join(trial, "agent", "mt_transcript.json")
        if not (os.path.exists(reward) and os.path.exists(transcript)):
            continue
        try:
            if open(reward).read().strip() not in ("1", "1.0"):
                continue
            messages = json.load(open(transcript))["messages"]
        except Exception:
            continue
        task = os.path.basename(trial.rstrip("/")).split("__")[0]
        if per_task.get(task, 0) >= PER_TASK:
            continue
        per_task[task] = per_task.get(task, 0) + 1
        msgs = []
        for m in messages:
            row = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_call_id"):
                row["tool_call_id"] = m["tool_call_id"]
            if m.get("tool_calls"):
                row["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                                      "function": {"name": (tc.get("function") or {}).get("name") or "",
                                                   "arguments": (tc.get("function") or {}).get("arguments") or "{}"}}
                                     for i, tc in enumerate(m["tool_calls"])]
            msgs.append(row)
        rows.append({"messages": msgs, "meta": {"harness": harness_label, "task": task}})
    return rows


def uniform(rows):
    return [{"messages": [{"role": m["role"], "content": m.get("content") or "", "tool_calls": m.get("tool_calls") or [],
                           "tool_call_id": m.get("tool_call_id") or ""} for m in r["messages"]],
             "meta": r.get("meta", {})} for r in rows]


def main():
    p = C.parser(__doc__, {"rehearsal-fc": "rehearsal-fc", "rehearsal-xml": "rehearsal-xml"})
    p.add_argument("--target", choices=["fc", "xml"], default="fc")
    args = p.parse_args()
    role = f"rehearsal-{args.target}"
    jobs = C.job_dirs(args, {role: role})
    out = args.out_dir

    if args.target == "fc":
        rehearsal = solved_rehearsal(jobs[role], "mt-fc-drill")
        cross = C.read_pool(out, "json-only") + C.read_pool(out, "pycall-only")
        C.write_pool(out, "json+pycall", uniform(cross))
        C.write_pool(out, "reh-fc", uniform(rehearsal))
        C.write_pool(out, "json+pycall+reh-fc", uniform(rehearsal + cross))
        return

    rehearsal = solved_rehearsal(jobs[role], "mt-xml-drill")
    # Drop over-length rehearsal rows (estimated at 3 characters per token; none were, as run).
    # 3h's own over-length rows stay, since its control arm was trained with them.
    rehearsal = [r for r in rehearsal if len(json.dumps(r["messages"])) // 3 <= 32768]
    # rehearsal rows go first (see the module docstring: populated tool_calls must come first)
    pool = sorted(uniform(rehearsal + C.read_pool(out, "3h")), key=lambda r: r["meta"]["harness"] != "mt-xml-drill")
    C.write_pool(out, "3h+reh-xml", pool)


if __name__ == "__main__":
    main()
