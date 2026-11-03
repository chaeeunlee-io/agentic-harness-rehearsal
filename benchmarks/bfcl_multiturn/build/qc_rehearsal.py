#!/usr/bin/env python3
"""Quality checks for BFCL multi-turn rehearsal tasks.

Run before collecting traces or training on the rehearsal dataset.

Checks:
- Solvability: the reference execution must pass grading.
- Echo guard: executing turn 1 without carrying its returned value into turn 2
  must fail.
- No leak: the derived value must not appear in the instruction, and all parent
  tasks must come from the training split.
- Diversity: report duplicate `(world, ground-truth)` pairs.

The script reports how many tasks pass each check, marks failures, and writes a
JSON report (default `<tasks>.qc_report.json`); it does not stop on a failure.

Usage:
    python qc_rehearsal.py [--tasks DIR] [--json OUT]

`--tasks` defaults to `runs/tasks/bfcl_multiturn/rehearsal`.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, collections

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # benchmarks/bfcl_multiturn
sys.path.insert(0, os.path.join(BENCH, "tasks"))
from bfcl_env.multi_turn_checker import multi_turn_checker  # noqa: E402

SPLIT = os.path.join(BENCH, "splits", "split.json")
REHEARSAL = os.path.join(os.path.dirname(os.path.dirname(BENCH)), "runs", "tasks", "bfcl_multiturn", "rehearsal")


def gate(drill_dir: str, train: set[str]) -> dict:
    name = os.path.basename(drill_dir.rstrip("/"))
    entry = json.load(open(os.path.join(drill_dir, "tests", "entry.json")))
    gt = json.load(open(os.path.join(drill_dir, "tests", "ground_truth.json")))
    instr = open(os.path.join(drill_dir, "instruction.md")).read()
    res = {"drill": name}

    full = [[list(t)] if t else [] for t in gt]
    res["g1_solvable"] = bool(multi_turn_checker(full, gt, entry, "multi_turn_drill", f"g1-{name}").get("valid"))

    probe_only = [[list(gt[0])]] + [[] for _ in gt[1:]]
    res["g2_echo_blocked"] = not bool(
        multi_turn_checker(probe_only, gt, entry, "multi_turn_drill", f"g2-{name}").get("valid"))

    # the derived value is whatever turn 2's call carries that turn 1 had to produce
    derived = []
    for call in (gt[1] if len(gt) > 1 else []):
        for piece in call.replace("'", "=").replace("(", "=").replace(")", "=").split("="):
            piece = piece.strip()
            if piece and not piece.replace("_", "").replace(".", "").isalpha():
                derived.append(piece)
    payload = json.loads(instr.split("```json", 1)[1].split("```", 1)[0])
    shown = " ".join(payload["turns"])
    res["g3_value_hidden"] = not any(d and d in shown for d in derived)
    res["g3_parent_in_train"] = True  # checked by caller against manifest
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=REHEARSAL)
    ap.add_argument("--json", help="report path (default: <tasks>.qc_report.json)")
    args = ap.parse_args()

    args.json = args.json or args.tasks.rstrip("/") + ".qc_report.json"
    train = set(json.load(open(SPLIT))["train"])
    test = set(json.load(open(SPLIT))["test"])
    manifest = {m["drill"]: m for m in json.load(open(args.tasks.rstrip("/") + ".manifest.json"))}

    rows, dupes = [], collections.Counter()
    for d in sorted(glob.glob(os.path.join(args.tasks, "*/"))):
        r = gate(d, train)
        m = manifest.get(r["drill"], {})
        r["parent"] = m.get("parent")
        r["g3_parent_in_train"] = r["parent"] in train
        r["parent_in_TEST"] = r["parent"] in test          # must always be False
        entry = json.load(open(os.path.join(d, "tests", "entry.json")))
        dupes[(json.dumps(entry["initial_config"], sort_keys=True),
               json.dumps(m.get("ground_truth")))] += 1
        rows.append(r)

    n = len(rows)
    def pct(k): return sum(1 for r in rows if r[k]), n
    print(f"drills checked: {n}")
    for k, label in [("g1_solvable", "G1 ground truth solvable"),
                     ("g2_echo_blocked", "G2 echo path blocked"),
                     ("g3_value_hidden", "G3 derived value absent from instruction"),
                     ("g3_parent_in_train", "G3 parent in TRAIN split")]:
        good, tot = pct(k)
        print(f"  {label:44s} {good}/{tot}" + ("" if good == tot else "   <-- FAILURES"))
    leaked = sum(1 for r in rows if r["parent_in_TEST"])
    print(f"  {'G3 parent in TEST split (must be 0)':44s} {leaked}")
    redundant = sum(c - 1 for c in dupes.values() if c > 1)
    print(f"  {'G4 redundant (duplicate world+GT) drills':44s} {redundant} of {n}")
    json.dump(rows, open(args.json, "w"), indent=1)
    print(f"report -> {args.json}")


if __name__ == "__main__":
    main()
