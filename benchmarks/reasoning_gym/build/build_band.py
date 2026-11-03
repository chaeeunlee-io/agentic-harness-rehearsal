"""Build the Reasoning Gym RL learning band (`splits/band.json`).

GRPO needs tasks with mixed rollout outcomes. Tasks where every rollout receives
the same reward give zero group-relative advantage.

The band is built from probe runs of the 3H SFT checkpoint across both training
tiers and all harnesses. A task is kept if at least one harness has mixed probe
outcomes. Because Reasoning Gym rewards lie in [0, 1], a mean reward strictly
between 0 and 1 also counts as mixed.

Usage:
    python build_band.py --jobs-dir <dir containing rgbandv3-easy-t2/, ..., rgbandv3-hard-goose/>

Writes `benchmarks/reasoning_gym/splits/band.json` by default (114 tasks as run).
The GRPO training tasks come from:
    exp tasks -b reasoning_gym --split train
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

TIERS = ["easy", "hard"]
HARNESSES = ["t2", "miniswe", "goose"]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "splits" / "band.json"


def read_rewards(jobs_dir: Path, job_prefix: str) -> dict:
    """(task, harness) -> list of rewards from the probe runs, both tiers."""
    rewards = defaultdict(list)
    for tier in TIERS:
        for harness in HARNESSES:
            job_dir = jobs_dir / f"{job_prefix}-{tier}-{harness}"
            result_files = list(job_dir.glob("*__*/result.json"))
            if not result_files:
                raise SystemExit(f"no probe results in {job_dir}; check --jobs-dir / --job-prefix")
            for result_file in result_files:
                result = json.loads(result_file.read_text())
                reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
                if reward is None:  # trial crashed before the verifier ran
                    continue
                rewards[(result.get("task_name", "?"), harness)].append(reward)
    return rewards


def is_mixed(rewards: list) -> bool:
    """True if the task was neither always solved nor always failed."""
    mean = sum(rewards) / len(rewards)
    solved = sum(1 for r in rewards if r >= 1)
    return 0 < mean < 1 or 0 < solved < len(rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--job-prefix", default="rgbandv3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rewards = read_rewards(args.jobs_dir, args.job_prefix)
    band = sorted({task for (task, _), rs in rewards.items() if is_mixed(rs)})
    if len(band) < 30:
        raise SystemExit(f"band has only {len(band)} tasks (expected >= 30); not writing {args.out}")

    with open(args.out, "w") as f:
        json.dump({"train": band, "test": []}, f, indent=1)

    in_band = {h: sum(1 for (t, hh) in rewards if hh == h and t in band) for h in HARNESSES}
    print(f"band: {len(band)} tasks (of these, probed on each harness: {in_band}) -> {args.out}")


if __name__ == "__main__":
    main()
