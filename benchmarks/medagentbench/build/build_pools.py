"""Build MedAgentBench distillation datasets: parity-only, t2-only, fc-only, and 3h.

Input: Qwen3-32B teacher traces from the training split, one job per harness.
Each verified trajectory becomes a row, with at most two rows per task and 150
rows per harness. `3h` combines the three harness datasets.

With `--with-parity-rehearsal`, also writes `parity+reh-t2` for the Gemma arm
using the t2 rehearsal dataset from `build_rehearsal.py` (build that first).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

JOBS = {"parity": "teacher-parity", "t2": "teacher-t2", "fc": "teacher-fc"}
EXTRACT = {"parity": C.chat_parity, "t2": C.chat_t2, "fc": C.chat_fc}
CAP = 150


def main():
    p = C.parser(__doc__, JOBS)
    p.add_argument("--with-parity-rehearsal", action="store_true", help="also write parity+reh-t2 (needs reh-t2.jsonl in --out-dir)")
    args = p.parse_args()
    jobs = C.job_dirs(args, JOBS)
    rng = random.Random(0)

    rows = {}
    for h in JOBS:
        rows[h] = C.collect(jobs[h], EXTRACT[h], {"harness": h, "reward": 1.0})
        rate = C.think_rate(rows[h])
        print(f"{h}: {len(rows[h])} rows from {jobs[h]} (think rate {rate:.2f})")
        if rows[h] and rate < 0.9:
            print(f"  warning: {h} think rate below 0.9; check that reasoning was captured before training")
        if not rows[h]:
            raise SystemExit(f"no usable trajectories for {h}")

    for h in JOBS:
        C.write_pool(args.out_dir, f"{h}-only", C.capped(rng, rows[h], CAP))
    mix = [r for h in JOBS for r in C.capped(rng, rows[h], CAP)]
    rng.shuffle(mix)
    C.write_pool(args.out_dir, "3h", mix)

    if args.with_parity_rehearsal:
        combo = C.capped(rng, rows["parity"], CAP) + C.read_pool(args.out_dir, "reh-t2")
        rng.shuffle(combo)
        C.write_pool(args.out_dir, "parity+reh-t2", combo)


if __name__ == "__main__":
    main()
