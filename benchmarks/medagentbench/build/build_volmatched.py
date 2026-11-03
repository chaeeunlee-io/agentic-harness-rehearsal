"""Build volume-matched MedAgentBench datasets for `matched-volume-mab`.

These datasets reduce `3h` to the size of a specialist dataset while keeping an
equal number of rows from each harness:

- `3h-n108`, `3h-n150`: random rows per harness, seed 20260922.
- `3h-n108-s2`, `3h-n108-s3`: same sampling with seeds 20260923 and 20260924.
- `3h-n108-cov`, `3h-n150-cov`: favor coverage of distinct tasks.

They match row counts, not token counts. Input is `3h.jsonl` in `--out-dir`,
produced by `build_pools.py`.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "runs" / "pools" / "medagentbench"
SEED = 20260922


def quotas(harnesses, total):
    """Equal rows per harness; any remainder goes to the first harnesses (sorted)."""
    per, extra = divmod(total, len(harnesses))
    return {h: per + (1 if i < extra else 0) for i, h in enumerate(harnesses)}


def random_cut(by_harness, total, seed):
    rng = random.Random(seed)
    rows = []
    for h, n in quotas(sorted(by_harness), total).items():
        rows.extend(rng.sample([line for _, line in by_harness[h]], n))
    rng.shuffle(rows)
    return rows


def coverage_cut(by_harness, total, seed):
    """Round-robin over harnesses; each pick prefers a row whose task is not covered yet."""
    harnesses = sorted(by_harness)
    quota = quotas(harnesses, total)
    rng = random.Random(seed)
    remaining = {h: list(by_harness[h]) for h in harnesses}
    for h in harnesses:
        rng.shuffle(remaining[h])
    covered, picked, taken = set(), [], {h: 0 for h in harnesses}
    while any(taken[h] < quota[h] for h in harnesses):
        for h in harnesses:
            if taken[h] >= quota[h] or not remaining[h]:
                continue
            i = next((i for i, (task, _) in enumerate(remaining[h]) if task not in covered), 0)
            task, line = remaining[h].pop(i)
            covered.add(task)
            picked.append(line)
            taken[h] += 1
    rng.shuffle(picked)
    return picked


POOLS = {
    "3h-n108": (random_cut, 108, SEED),
    "3h-n150": (random_cut, 150, SEED),
    "3h-n108-s2": (random_cut, 108, 20260923),
    "3h-n108-s3": (random_cut, 108, 20260924),
    "3h-n108-cov": (coverage_cut, 108, SEED),
    "3h-n150-cov": (coverage_cut, 150, SEED),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"reads 3h.jsonl, writes the pools (default {DEFAULT_OUT})")
    args = p.parse_args()
    source = args.out_dir / "3h.jsonl"
    if not source.exists():
        raise SystemExit(f"{source} not found: run build_pools.py first")

    by_harness = defaultdict(list)  # harness -> [(task, raw line)]
    for line in open(source):
        line = line.strip()
        if line:
            meta = json.loads(line).get("meta") or {}
            by_harness[meta.get("harness", "?")].append((meta.get("task"), line))

    for name, (cut, total, seed) in POOLS.items():
        rows = cut(by_harness, total, seed)
        path = args.out_dir / f"{name}.jsonl"
        path.write_text("\n".join(rows) + "\n")
        tasks = len({json.loads(r)["meta"]["task"] for r in rows})
        print(f"{name:12s} {len(rows)} rows, {tasks} distinct tasks -> {path}")


if __name__ == "__main__":
    main()
