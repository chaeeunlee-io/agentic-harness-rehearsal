"""`exp build-pool`: build SFT pools from collected traces with the benchmark's build/ scripts.

configs/benchmarks/<b>.yaml `build:` lists, in order, which script writes which settings.
`--pool <setting>` runs the one step that writes it; without --pool every step runs in
order (later steps read pools written by earlier ones). Traces are read from
runs/jobs/<benchmark>/ (where `exp collect` writes them) and pools are written to
runs/pools/<benchmark>/<setting>.jsonl; train on one with `exp sft --data <path>`.
"""
from __future__ import annotations

import subprocess
import sys

from exp import config as C
from exp.render import RUNS

ROOT = C.ROOT


def steps_for(b, pool: str | None) -> list[dict]:
    steps = b.cfg.get("build") or []
    if not steps:
        raise SystemExit(f"{b.name}: no build steps configured")
    if pool is None:
        return steps
    hit = [s for s in steps if pool in s["writes"]]
    if not hit:
        known = [w for s in steps for w in s["writes"]]
        raise SystemExit(f"{b.name}: no build step writes {pool!r}; buildable: {', '.join(known)}")
    return hit[:1]


def main(a) -> int:
    b = C.benchmark(a.benchmark)
    jobs = a.jobs_dir or str(RUNS / "jobs" / b.name)
    out = a.out_dir or str(RUNS / "pools" / b.name)
    rc = 0
    for step in steps_for(b, a.pool):
        script = ROOT / "benchmarks" / b.name / step["script"]
        argv = [str(script), *(step.get("args") or []), "--out-dir", out]
        if step.get("reads", "jobs") == "jobs":  # `reads: pools` = a step that only reads earlier pools
            argv += ["--jobs-dir", jobs]
        if step.get("env") == "sft" and a.tokenizer:
            argv += ["--tokenizer", a.tokenizer]
        # --min-trajectories goes only to the builders whose step lists it under `options`
        if a.min_trajectories is not None and "min-trajectories" in (step.get("options") or []):
            argv += ["--min-trajectories", str(a.min_trajectories)]
        # builders that render with the student's tokenizer run in the SFT environment (transformers)
        py = ["uv", "run", "--project", str(ROOT / "setup" / "sft"), "python"] if step.get("env") == "sft" else [sys.executable]
        print(f"# {b.name}: {', '.join(step['writes'])}")
        print("+ " + " ".join(py + argv))
        if not a.dry_run:
            rc |= subprocess.run(py + argv, cwd=ROOT).returncode
            if rc:
                break
    if not a.dry_run and not rc:
        print(f"train on a pool with: uv run exp sft -b {b.name} --pool <setting> --data {out}/<setting>.jsonl")
    return rc
