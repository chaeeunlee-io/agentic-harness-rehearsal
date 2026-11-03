"""Shared helpers for building AppWorld distillation datasets.

Each row is {"messages": [...], "meta": {...}} and contains one verified
trajectory rendered as the student's training chat. The trainer reads only
"messages".

A harness's traces may span multiple job directories. Pass each with
`--job ROLE=PATH`; trials are read from each directory in sorted order.
"""
import argparse
import json
from glob import glob
from pathlib import Path

BENCH = "appworld"
REPO = Path(__file__).resolve().parents[3]
DEFAULT_JOBS = REPO / "runs" / "jobs" / BENCH
DEFAULT_OUT = REPO / "runs" / "pools" / BENCH
TEMPLATE = REPO / "training" / "templates" / "qwen3_acc_thinking.jinja2"


def parser(doc: str, jobs: dict) -> argparse.ArgumentParser:
    """CLI shared by the builders. `jobs` maps a role to its default job name."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS, help=f"harbor job directories (default {DEFAULT_JOBS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"where pools are written and read (default {DEFAULT_OUT})")
    p.add_argument("--job", action="append", default=[], metavar="ROLE=PATH",
                   help="read ROLE's trials from PATH instead of <jobs-dir>/<default name>; repeat for several "
                        f"directories; roles and default names: {', '.join(f'{k}={v}' for k, v in jobs.items())}")
    return p


def job_dirs(args, jobs: dict) -> dict:
    """Role -> list of job directories."""
    given = {}
    for spec in args.job:
        role, _, path = spec.partition("=")
        if role not in jobs:
            raise SystemExit(f"unknown job role {role!r}; roles: {', '.join(jobs)}")
        given.setdefault(role, []).append(Path(path))
    out = {role: given.get(role, [args.jobs_dir / name]) for role, name in jobs.items()}
    for role, dirs in out.items():
        for d in dirs:
            if not d.is_dir():
                raise SystemExit(f"job directory for {role} not found: {d}")
    return out


def trials(dirs: list, pattern: str = "*__*/") -> list:
    """Trial directories, sorted within each job directory, directories in the given order."""
    return [t for d in dirs for t in sorted(glob(f"{d}/{pattern}"))]


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
