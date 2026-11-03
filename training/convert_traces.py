"""Convert Harbor trials into a TRL SFT dataset using rejection sampling.

Reads trial directories under `<jobs_dir>/<job>/<task>__<id>/`, keeps trials
whose verifier reward passes the threshold (default: > 0), and writes one JSONL
row per trial in TRL chat format:

    {"messages": [...],
     "meta": {"task": ..., "arm": ..., "reward": ..., "trial": ...}}

Harbor's `agent` role is mapped to `assistant`. Transcripts contain parsed
actions without `<think>` blocks.

`--per-task-cap` limits rows per `(task, arm)`, preferring higher-reward and then
shorter trials. The converter works with adapters that write a transcript under
`agent/<name>.json`.

Usage:
    python3 convert_traces.py --jobs DIR [DIR ...] --out train.jsonl \
        [--min-reward 0.5] [--per-task-cap 2] [--require-finish]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROLE_MAP = {"agent": "assistant", "assistant": "assistant", "user": "user", "system": "system"}


def messages_from_atif(steps: list, include_thinking: bool = True,
                       trial_dir: Path | None = None) -> list | None:
    """ATIF steps (terminus-2 style) -> chat messages.

    Assistant surface form: terminus-2 speaks PLAIN JSON ({analysis, plan,
    commands}) — the tool_calls in ATIF are a recording artifact, NOT the
    live wire format. When trial_dir is given, each agent step's exact raw
    body is read from agent/episode-<j>/response.txt (byte-true surface);
    thinking is prepended from reasoning_content. Without trial_dir we fall
    back to reconstructing the JSON from the recorded fields.

    Observations become the following user turn (approximates the live t2
    prompt builder — acceptable for SFT).
    """
    messages = []
    agent_idx = 0
    for s in steps:
        if s.get("source") == "user":
            if s.get("message"):
                messages.append({"role": "user", "content": s["message"]})
            continue
        if s.get("source") != "agent":
            return None
        body = None
        if trial_dir is not None:
            rf = Path(trial_dir) / "agent" / f"episode-{agent_idx}" / "response.txt"
            if rf.is_file():
                body = rf.read_text().strip()
        agent_idx += 1
        if body is None:
            # fallback: rebuild the JSON wire format from recorded fields
            cmds = []
            for tc in s.get("tool_calls") or []:
                args = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
                if args.get("keystrokes") is not None:
                    cmds.append({"keystrokes": args["keystrokes"],
                                 "duration": args.get("duration", 0.1)})
            body = json.dumps({"analysis": "", "plan": s.get("message") or "",
                               "commands": cmds}, indent=2)
        parts = []
        think = s.get("reasoning_content")
        if think and include_thinking:
            parts.append(f"<think>\n{think.strip()}\n</think>")
        parts.append(body)
        messages.append({"role": "assistant", "content": "\n\n".join(parts)})
        results = (s.get("observation") or {}).get("results") or []
        obs = "\n".join(r.get("content", "") if isinstance(r, dict) else str(r) for r in results).strip()
        if obs:
            messages.append({"role": "user", "content": obs})
    # trailing user turn carries no training signal for the assistant
    while messages and messages[-1]["role"] == "user":
        messages.pop()
    return messages or None


def load_trial(trial_dir: Path, include_thinking: bool = True) -> dict | None:
    rj = trial_dir / "result.json"
    if not rj.is_file():
        return None
    try:
        result = json.loads(rj.read_text())
    except Exception:
        return None
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if reward is None:
        return None
    agent_files = sorted((trial_dir / "agent").glob("*.json")) if (trial_dir / "agent").is_dir() else []
    transcript, atif_steps = None, None
    for af in agent_files:
        try:
            blob = json.loads(af.read_text())
        except Exception:
            continue
        if isinstance(blob, dict) and isinstance(blob.get("transcript"), list):
            transcript = blob["transcript"]
            break
        if isinstance(blob, dict) and isinstance(blob.get("steps"), list):
            atif_steps = blob["steps"]
            break
    if atif_steps is not None:
        messages = messages_from_atif(atif_steps, include_thinking=include_thinking,
                                      trial_dir=trial_dir)
        if not messages:
            return None
    elif transcript:
        messages = []
        for m in transcript:
            role = ROLE_MAP.get(m.get("role", ""), None)
            content = m.get("content")
            if role is None or not isinstance(content, str) or not content.strip():
                return None  # unknown role / empty turn -> drop trial, not just turn
            reasoning = m.get("reasoning")
            if role == "assistant" and reasoning and include_thinking:
                content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
            messages.append({"role": role, "content": content})
    else:
        return None
    if not any(m["role"] == "assistant" for m in messages):
        return None
    task_path = str((result.get("task_id") or {}).get("path", ""))
    return {
        "messages": messages,
        "meta": {
            "task": result.get("task_name", trial_dir.name.split("__")[0]),
            "arm": (result.get("source") or "native"),
            "reward": reward,
            "trial": trial_dir.name,
            "task_path": task_path,
            "n_turns": len(messages),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-reward", type=float, default=0.0, help="keep trials with reward STRICTLY above this")
    ap.add_argument("--per-task-cap", type=int, default=0, help="max kept trials per (task,arm); 0 = no cap")
    ap.add_argument("--require-finish", action="store_true",
                    help="drop trials whose last assistant turn is not a FINISH(...) call")
    ap.add_argument("--no-thinking", action="store_true",
                    help="omit recorded <think> blocks from assistant turns")
    args = ap.parse_args()

    seen = kept = 0
    by_cell: dict[tuple, list] = defaultdict(list)
    for jobs_root in args.jobs:
        for rj in sorted(jobs_root.glob("**/result.json")):
            trial_dir = rj.parent
            if trial_dir.name == jobs_root.name or not (trial_dir / "agent").exists():
                continue
            seen += 1
            row = load_trial(trial_dir, include_thinking=not args.no_thinking)
            if row is None or row["meta"]["reward"] <= args.min_reward:
                continue
            if args.require_finish:
                last_a = [m for m in row["messages"] if m["role"] == "assistant"][-1]
                if "FINISH(" not in last_a["content"]:
                    continue
            by_cell[(row["meta"]["task"], row["meta"]["arm"])].append(row)

    rows = []
    for cell, cell_rows in sorted(by_cell.items()):
        cell_rows.sort(key=lambda r: (-r["meta"]["reward"], r["meta"]["n_turns"]))
        rows += cell_rows[: args.per_task_cap] if args.per_task_cap else cell_rows
    kept = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tasks = {r["meta"]["task"] for r in rows}
    n_think = sum(1 for r in rows if any("<think>" in m["content"] for m in r["messages"]))
    print(f"trials seen {seen} -> kept {kept} ({len(tasks)} tasks, {len(by_cell)} cells; "
          f"{n_think} with thinking) -> {args.out}")
    if rows:
        turns = [r["meta"]["n_turns"] for r in rows]
        print(f"turns/trial: min {min(turns)} med {sorted(turns)[len(turns)//2]} max {max(turns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
