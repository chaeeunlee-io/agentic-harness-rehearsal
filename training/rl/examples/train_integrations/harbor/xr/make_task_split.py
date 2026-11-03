"""Stratified train/held-out task split for MedAgentBench (240/60).

MedAgentBench ships only its 300-task test set (10 categories x 30 patient
instantiations; adapter consumes upstream test_data_v2.json). No official train
split exists, so we make one — and never train on the held-out 60.

Stratification:
  * primary stratum: category (task1..task10) — instances within a category are
    near-clones (same template, different patient), so category explains most
    difficulty variance; 24/6 per category keeps both splits comparable.
  * finer stratum where ground truth demands it: within task4 and task6, the
    absent-answer subset (true answer is the -1 sentinel; measured from the
    graders' own printed references) is split 80/20 SEPARATELY, so the
    sentinel arms' reporting subsets keep the same absent-rate on both sides.

Decision 2026-08-20: instance-level split only; no held-out category.

Usage:
    python make_task_split.py --dataset <path to medagentbench> \
        --ground-truth <medagentbench_refsol_ground_truth.json> \
        --out <medagentbench_task_split.json> [--seed 20260820]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HELDOUT_FRACTION = 0.2


def split_stratum(tasks: list[str], rng: random.Random) -> tuple[list[str], list[str]]:
    """80/20 split of one stratum; held-out count rounds to nearest, min 1 if the
    stratum has >= 2 members (a stratum must never vanish from either side)."""
    tasks = sorted(tasks)
    n_held = max(1, round(len(tasks) * HELDOUT_FRACTION)) if len(tasks) >= 2 else 0
    held = set(rng.sample(tasks, n_held))
    return [t for t in tasks if t not in held], sorted(held)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    tasks = sorted(
        p.name for p in args.dataset.iterdir() if p.is_dir() and (p / "task.toml").is_file()
    )
    if len(tasks) != 300:
        raise SystemExit(f"expected 300 tasks, found {len(tasks)} — wrong dataset dir?")

    absent = set(json.loads(args.ground_truth.read_text())["absent_tasks"])

    # Build strata: category, refined by absent-status where absence occurs.
    strata: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        cat = t.split("_")[0]
        if cat in ("task4", "task6"):
            strata[f"{cat}/{'absent' if t in absent else 'present'}"].append(t)
        else:
            strata[cat].append(t)

    rng = random.Random(args.seed)
    train: list[str] = []
    heldout: list[str] = []
    report = {}
    for name in sorted(strata):
        tr, he = split_stratum(strata[name], rng)
        train += tr
        heldout += he
        report[name] = {"train": len(tr), "heldout": len(he)}

    train.sort()
    heldout.sort()
    if set(train) & set(heldout):
        raise SystemExit("overlap between splits — bug")
    if len(train) + len(heldout) != 300:
        raise SystemExit("split does not cover all tasks — bug")

    args.out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "strata": report,
                "n_train": len(train),
                "n_heldout": len(heldout),
                "train": train,
                "heldout": heldout,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"strata: {json.dumps(report, indent=1)}")
    print(f"train {len(train)} / heldout {len(heldout)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
