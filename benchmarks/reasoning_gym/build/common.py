"""Shared helpers for building Reasoning Gym distillation datasets.

Each row contains one verified trajectory as
`{"messages": [...], "meta": {...}}`. The trainer reads only `messages`.
"""
import argparse
import json
import os
from glob import glob
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO = BENCH_DIR.parent.parent
DEFAULT_JOBS = REPO / "runs" / "jobs" / "reasoning_gym"
DEFAULT_OUT = REPO / "runs" / "pools" / "reasoning_gym"
PER_TASK = 2  # at most two trajectories per task, so easy tasks do not dominate


def parser(doc: str, jobs: dict) -> argparse.ArgumentParser:
    """CLI shared by the builders. `jobs` maps a role to its default job name."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS, help=f"harbor job directories (default {DEFAULT_JOBS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"where pools are written and read (default {DEFAULT_OUT})")
    p.add_argument("--job", action="append", default=[], metavar="ROLE=PATH",
                   help="use this job directory for ROLE instead of <jobs-dir>/<default name>; "
                        f"roles and default names: {', '.join(f'{k}={v}' for k, v in jobs.items())}")
    return p


def job_dirs(args, jobs: dict) -> dict:
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


def task_of(trial: str) -> str:
    return os.path.basename(trial.rstrip("/")).split("__")[0]


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


def think_rate(rows: list) -> float:
    assistant = [m for r in rows for m in r["messages"] if m["role"] == "assistant"]
    return sum("<think>" in (m.get("content") or "") for m in assistant) / len(assistant) if assistant else 0.0


def capped(rng, rows: list, n: int) -> list:
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
