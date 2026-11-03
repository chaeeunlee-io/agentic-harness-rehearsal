"""Build the MedAgentBench RL learning band (`splits/band.json`).

GRPO needs tasks with mixed rollout outcomes. Tasks where every rollout receives
the same reward give zero group-relative advantage.

The band is built from probe runs of the 3H SFT checkpoint on training tasks.
A task is kept if at least one harness has both successful and failed probe
rollouts.

Usage:
    python build_band.py --jobs-dir <dir containing probev4-parity/, probev4-t2/, probev4-fc/>

Writes `benchmarks/medagentbench/splits/band.json` by default (96 tasks as run).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HARNESSES = ["parity", "t2", "fc"]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "splits" / "band.json"


def read_rewards(jobs_dir: Path, job_prefix: str) -> dict:
    """(task, harness) -> list of rewards from the probe runs."""
    rewards = defaultdict(list)
    for harness in HARNESSES:
        job_dir = jobs_dir / f"{job_prefix}-{harness}"
        result_files = list(job_dir.glob("task*/result.json"))
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
    parser.add_argument("--job-prefix", default="probev4")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rewards = read_rewards(args.jobs_dir, args.job_prefix)
    band = sorted({task for (task, _), rs in rewards.items() if is_mixed(rs)})

    with open(args.out, "w") as f:
        json.dump({"train": band, "test": []}, f, indent=1)

    in_band = {h: sum(1 for (t, hh) in rewards if hh == h and t in band) for h in HARNESSES}
    print(f"band: {len(band)} tasks (of these, probed on each harness: {in_band}) -> {args.out}")


if __name__ == "__main__":
    main()
