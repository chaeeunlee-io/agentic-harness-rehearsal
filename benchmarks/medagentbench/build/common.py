"""Shared helpers for building MedAgentBench distillation datasets.

Each row contains one verified trajectory as
`{"messages": [...], "meta": {...}}`. The trainer reads only `messages`.

Trials are read in directory-listing order. Because the per-task cap keeps the
first matching trials, changing that order can change which trajectories are
selected.
"""
import argparse
import json
import os
from glob import glob
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO = BENCH_DIR.parent.parent
DEFAULT_JOBS = REPO / "runs" / "jobs" / "medagentbench"
DEFAULT_OUT = REPO / "runs" / "pools" / "medagentbench"
PER_TASK = 2  # at most two trajectories per task, so easy tasks do not dominate


def parser(doc: str, jobs: dict) -> argparse.ArgumentParser:
    """CLI shared by the builders. `jobs` maps a role ("parity", "rehearsal", ...) to its default job name."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS, help=f"harbor job directories (default {DEFAULT_JOBS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"where pools are written and read (default {DEFAULT_OUT})")
    p.add_argument("--job", action="append", default=[], metavar="ROLE=PATH",
                   help="use this job directory for ROLE instead of <jobs-dir>/<default name>; "
                        f"roles and default names: {', '.join(f'{k}={v}' for k, v in jobs.items())}")
    return p


def job_dirs(args, jobs: dict) -> dict:
    """Role -> job directory, from --jobs-dir and the default names, overridden by --job ROLE=PATH."""
    out = {role: args.jobs_dir / name for role, name in jobs.items()}
    for spec in args.job:
        role, _, path = spec.partition("=")
        if role not in jobs:
            raise SystemExit(f"unknown job role {role!r}; roles: {', '.join(jobs)}")
        out[role] = Path(path)
    for role, d in out.items():
        if not d.is_dir():
            raise SystemExit(f"job directory for {role} not found: {d}")
    return out


def trials(job_dir: Path, reward: str = "1") -> list:
    """Trial directories whose verifier wrote exactly `reward` ("1" = solved, "0" = failed)."""
    out = []
    for t in glob(f"{job_dir}/task*/"):
        try:
            if open(f"{t}/verifier/reward.txt").read().strip() == reward:
                out.append(t.rstrip("/"))
        except OSError:
            continue
    return out


def task_of(trial: str) -> str:
    return os.path.basename(trial).split("__")[0]


def with_think(message: dict) -> str:
    """Assistant text with its reasoning inline as <think>...</think>.

    Served with a reasoning parser, the model's reasoning arrives in a separate field;
    the student is trained on it inline."""
    content = message.get("content") or ""
    if "<think>" in content:
        return content
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content


def chat_parity(trial: str):
    """The parity agent's transcript as a chat."""
    path = f"{trial}/agent/medagentbench-parity-agent.json"
    if not os.path.exists(path):
        return None
    transcript = json.load(open(path)).get("transcript", [])
    if len(transcript) < 2:
        return None
    msgs = []
    for m in transcript:
        if m.get("role") == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif m.get("role") in ("agent", "assistant"):
            msgs.append({"role": "assistant", "content": with_think(m)})
    return msgs if len(msgs) >= 2 and msgs[0]["role"] == "user" else None


def chat_t2(trial: str):
    """Terminus-2's episodes (prompt.txt / response.txt per episode) as a chat."""
    episodes = sorted(glob(f"{trial}/agent/episode-*"), key=lambda p: int(p.split("-")[-1]))
    if not episodes:
        return None
    try:
        msgs = [{"role": "user", "content": open(f"{episodes[0]}/prompt.txt").read()}]
        for i, ep in enumerate(episodes):
            if not os.path.exists(f"{ep}/response.txt"):
                return None
            msgs.append({"role": "assistant", "content": open(f"{ep}/response.txt").read()})
            if i + 1 < len(episodes):
                msgs.append({"role": "user", "content": open(f"{episodes[i + 1]}/prompt.txt").read()})
        return msgs
    except OSError:
        return None


def chat_fc(trial: str, inline_think: bool = True):
    """The fc agent's tool-calling history as a chat."""
    found = glob(f"{trial}/agent/fc_history.json")
    if not found:
        return None
    history = json.load(open(found[0]))
    if len(history) < 2:
        return None
    if not inline_think:
        return history
    out = []
    for m in history:
        if m.get("role") == "assistant":
            m = dict(m)
            m["content"] = with_think(m)
        out.append(m)
    return out


def collect(job_dir: Path, extract, meta: dict, reward: str = "1", keep_trial: bool = True, limit=None) -> list:
    """One row per trial with the given reward, at most PER_TASK per task, in listing order."""
    rows, per_task = [], {}
    for t in trials(job_dir, reward):
        task = task_of(t)
        if per_task.get(task, 0) >= PER_TASK:
            continue
        msgs = extract(t)
        if not msgs:
            continue
        m = {"task": task, **meta}
        if keep_trial:
            m["trial"] = os.path.basename(t)
        rows.append({"messages": msgs, "meta": m})
        per_task[task] = per_task.get(task, 0) + 1
        if limit and len(rows) >= limit:
            break
    return rows


def think_rate(rows: list) -> float:
    assistant = [m for r in rows for m in r["messages"] if m["role"] == "assistant"]
    return sum("<think>" in (m.get("content") or "") for m in assistant) / len(assistant) if assistant else 0.0


def capped(rng, rows: list, n: int) -> list:
    """All rows, or a random `n` of them."""
    return rows if len(rows) <= n else rng.sample(rows, n)


def read_pool(out_dir: Path, setting: str) -> list:
    path = out_dir / f"{setting}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: build {setting} first")
    return [json.loads(line) for line in open(path)]


def write_pool(out_dir: Path, setting: str, rows: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{setting}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{setting:24s} {len(rows):4d} rows -> {path}")
