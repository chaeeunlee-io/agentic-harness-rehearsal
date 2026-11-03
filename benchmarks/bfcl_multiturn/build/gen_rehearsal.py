#!/usr/bin/env python3
"""Generate BFCL multi-turn rehearsal tasks.

Each rehearsal task specifies literal API calls and a mechanical derivation rather
than a domain goal.

Two constraints are enforced:
1. Turn 2 depends on a value returned by turn 1, so copying the instruction without
   executing it cannot pass.
2. Parent tasks come only from the training split, and instructions contain no
   domain rationale or target value.

Ground truth is computed by executing the reference calls against each parent task's
initial state. A unique execution ID is used for every generation to avoid cached
state carrying across tasks.

Parent tasks are read from the generated training tasks, so run
`exp tasks -b bfcl_multiturn --split train` first. Outputs standard Harbor task
directories under `runs/tasks/bfcl_multiturn/rehearsal/`, and
`rehearsal.manifest.json` next to that directory (read by `qc_rehearsal.py`).

Usage: `python gen_rehearsal.py [--limit N] [--train-tasks DIR] [--out DIR]`
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Callable

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # benchmarks/bfcl_multiturn
BFCL_ENV = os.path.join(BENCH, "tasks", "bfcl_env")                     # BFCL's API simulators + checker
sys.path.insert(0, os.path.join(BENCH, "tasks"))
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call  # noqa: E402

SPLIT = os.path.join(BENCH, "splits", "split.json")
RUNS_TASKS = os.path.join(os.path.dirname(os.path.dirname(BENCH)), "runs", "tasks", "bfcl_multiturn")
TASKS = os.path.join(RUNS_TASKS, "train")      # parents: the generated training tasks (--train-tasks)
EVAL_TEMPLATE = os.path.join(BENCH, "tasks", "evaluate_template.py")


class Family:
    """One rehearsal pattern for one API class.

    probe:   read-only calls issued in turn 1.
    derive:  pure function from turn-1 output -> the value V the agent must carry over.
    act:     builds turn 2's state-mutating call from V.
    """

    def __init__(self, cls: str, probe: list[str], derive: Callable[[str], str | None],
                 act: Callable[[str], str], describe: Callable[[str], str],
                 guard: Callable[[str], bool] | None = None):
        self.cls, self.probe, self.derive, self.act, self.describe = cls, probe, derive, act, describe
        self.guard = guard or (lambda _v: True)


def _first_entry(out: str) -> str | None:
    """First name listed by ls(a=True)."""
    try:
        items = json.loads(out).get("current_directory_content") or []
    except Exception:
        return None
    items = [i for i in items if not i.startswith(".")]
    return items[0] if items else None



def _first_watchlist_symbol(out: str) -> str | None:
    """First ticker on the trading bot's watchlist."""
    try:
        wl = json.loads(out).get("watchlist") or []
    except Exception:
        return None
    return wl[0] if wl else None


def _fuel_level(out: str) -> str | None:
    """Current fuel level, as a bare number string."""
    try:
        lvl = json.loads(out).get("fuelLevel")
    except Exception:
        return None
    return None if lvl is None else str(lvl)


FAMILIES = [
    Family(
        cls="GorillaFileSystem",
        probe=["ls(a=True)"],
        derive=_first_entry,
        act=lambda v: f"touch(file_name='rehearsal_{v}.txt')",
        describe=lambda _: (
            "Turn 1. Call `ls(a=True)` and read its result.\n"
            "Turn 2. Let V be the FIRST entry listed by that result which does not start "
            "with a dot. Call `touch(file_name='rehearsal_<V>.txt')`, substituting V."
        ),
    ),
    Family(
        cls="TradingBot",
        probe=["get_watchlist()"],
        derive=_first_watchlist_symbol,
        act=lambda v: f"remove_stock_from_watchlist(symbol='{v}')",
        describe=lambda _: (
            "Turn 1. Call `get_watchlist()` and read its result.\n"
            "Turn 2. Let V be the FIRST ticker symbol in that watchlist. Call "
            "`remove_stock_from_watchlist(symbol='<V>')`, substituting V."
        ),
    ),
    Family(
        cls="VehicleControlAPI",
        probe=["displayCarStatus(option='fuel')"],
        derive=_fuel_level,
        act=lambda v: f"fillFuelTank(fuelAmount={round(float(v), 1)})",
        describe=lambda _: (
            "Turn 1. Call `displayCarStatus(option='fuel')` and read its result.\n"
            "Turn 2. Let F be the reported fuelLevel. Call `fillFuelTank(fuelAmount=F)`, "
            "substituting F rounded to one decimal place."
        ),
        guard=lambda v: float(v) <= 25.0,   # F + F must stay within the 50-unit tank
    ),
]


def build_drill(parent: str, fam: Family, idx: int) -> dict | None:
    """Execute the probe in the parent's world, derive V, return a drill spec."""
    entry = json.load(open(os.path.join(TASKS, parent, "tests", "entry.json")))
    if fam.cls not in (entry.get("involved_classes") or []):
        return None
    out, _ = execute_multi_turn_func_call(
        fam.probe, entry.get("initial_config") or {}, entry["involved_classes"],
        "mt_drillgen", f"drillgen-{idx}-{parent}", long_context=False,
    )
    v = fam.derive(out[-1]) if out else None
    if not v or not fam.guard(v):
        return None
    turns = fam.describe(v).split("\n")
    if any(v in turn for turn in turns):
        # the derived value appears verbatim in the instruction -> the drill could be
        # passed by transcription. Never emit these; QC gate G3 also re-checks.
        return None
    return {
        "id": f"drill-{fam.cls.lower()}-{idx}",
        "parent": parent,
        "involved_classes": entry["involved_classes"],
        "initial_config": entry.get("initial_config") or {},
        "turns": turns,
        "ground_truth": [list(fam.probe), [fam.act(v)]],
    }


def write_task(spec: dict, funcs: list[dict], out_dir: str) -> str:
    d = os.path.join(out_dir, spec["id"])
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    payload = {
        "id": spec["id"], "category": "multi_turn_drill",
        "involved_classes": spec["involved_classes"],
        "initial_config": spec["initial_config"],
        "turns": spec["turns"], "functions": funcs,
    }
    open(os.path.join(d, "instruction.md"), "w").write(
        "BFCL multi-turn rehearsal task. The runner supplies turns sequentially.\n\n"
        "```json\n" + json.dumps(payload) + "\n```\n")
    entry = {"id": spec["id"], "category": "multi_turn_drill",
             "involved_classes": spec["involved_classes"],
             "initial_config": spec["initial_config"],
             "question": [[{"role": "user", "content": t}] for t in spec["turns"]]}
    json.dump(entry, open(os.path.join(d, "tests", "entry.json"), "w"))
    json.dump(spec["ground_truth"], open(os.path.join(d, "tests", "ground_truth.json"), "w"))
    shutil.copyfile(EVAL_TEMPLATE, os.path.join(d, "tests", "evaluate.py"))
    open(os.path.join(d, "tests", "test.sh"), "w").write(
        "#!/bin/bash\n# evaluate.py writes /logs/verifier/reward.txt itself.\n"
        "cd /tests\npython3 /tests/evaluate.py\nexit 0\n")
    os.chmod(os.path.join(d, "tests", "test.sh"), 0o755)
    env_dst = os.path.join(d, "tests", "bfcl_env")
    if not os.path.exists(env_dst):
        shutil.copytree(BFCL_ENV, env_dst)
    os.makedirs(os.path.join(d, "environment"), exist_ok=True)
    open(os.path.join(d, "environment", "Dockerfile"), "w").write("FROM python:3.10-slim\nWORKDIR /app\n")
    open(os.path.join(d, "task.toml"), "w").write(
        'schema_version = "1.0"\n\n[task]\nname = "bfclmt-drill/' + spec["id"] + '"\n'
        'keywords = ["bfcl-multi-turn", "rehearsal"]\n\n[metadata]\nsource = "bfcl-mt-drill"\n'
        'source_id = "' + spec["parent"] + '"\n\n[agent]\ntimeout_sec = 900.0\n\n'
        '[verifier]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 3600.0\n'
        'cpus = 1\nmemory_mb = 4096\nstorage_mb = 10240\n')
    return d


def main() -> None:
    global TASKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max drills per family (0 = all)")
    ap.add_argument("--train-tasks", default=TASKS, help="generated training tasks, the drills' parents")
    ap.add_argument("--out", default=os.path.join(RUNS_TASKS, "rehearsal"))
    args = ap.parse_args()
    TASKS = args.train_tasks   # build_drill reads each parent's world from here

    train = json.load(open(SPLIT))["train"]
    made, manifest = 0, []
    seen: set[tuple[str, str]] = set()
    for fam in FAMILIES:
        n = 0
        for parent in train:
            if args.limit and n >= args.limit:
                break
            try:
                spec = build_drill(parent, fam, made)
            except Exception as exc:  # a parent whose world rejects the probe
                print(f"skip {parent}: {type(exc).__name__}", file=sys.stderr)
                continue
            if not spec:
                continue
            key = (json.dumps(spec["initial_config"], sort_keys=True),
                   json.dumps(spec["ground_truth"]))
            if key in seen:      # same world + same reference calls: redundant
                continue
            seen.add(key)
            raw = open(os.path.join(TASKS, parent, "instruction.md")).read()
            funcs = json.loads(raw.split("```json", 1)[1].split("```", 1)[0])["functions"]
            write_task(spec, funcs, args.out)
            manifest.append({"drill": spec["id"], "parent": parent,
                             "ground_truth": spec["ground_truth"]})
            made += 1
            n += 1
        print(f"{fam.cls}: {n} drills")
    json.dump(manifest, open(args.out.rstrip("/") + ".manifest.json", "w"), indent=1)
    print(f"total {made} drills -> {args.out}")


if __name__ == "__main__":
    main()
