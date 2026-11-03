"""AppWorld verifier. Runs INSIDE the sandbox, independent of the agent
process -- reloads the on-disk experiment output the agent's process left
under $APPWORLD_ROOT/experiments/outputs/$APPWORLD_EXPERIMENT_NAME/ and
re-evaluates fresh via appworld's own evaluate_task(). Rule-based, no LLM
anywhere in this path (confirmed against the real package: evaluate_task ->
TestTracker runs Python assertions comparing final DB/model state to
ground_truth/evaluation.py's checks -- see adapters/appworld/README.md for
the exact live round-trip that established this).

Writes /logs/verifier/reward.txt (1.0 / 0.0, matching AppWorld's own binary
TGC "did every assertion in ground_truth/evaluation.py pass" definition --
NOT pass_percentage, which would give partial credit for a task where, e.g.,
the answer is right but some other assertion incidentally failed; TGC/
`tracker.success` is the benchmark's own reported metric and is what pass^k
in the original paper is computed over) and /logs/verifier/result.json (full
detail: pass/fail counts, task_completed flag, error if the agent's process
never actually produced experiment output at all).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
RESULT_PATH = Path("/logs/verifier/result.json")


def main() -> None:
    task_id = os.environ["APPWORLD_TASK_ID"]
    experiment_name = os.environ.get("APPWORLD_EXPERIMENT_NAME", "harbor_trial")

    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    from appworld import evaluate_task

    try:
        tracker = evaluate_task(
            task_id, experiment_name=experiment_name, save_report=True
        )
    except Exception as exc:  # agent process never ran / produced no output
        REWARD_PATH.write_text("0.0")
        RESULT_PATH.write_text(
            json.dumps(
                {
                    "reward": 0.0,
                    "task_id": task_id,
                    "experiment_name": experiment_name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        raise SystemExit(0)  # exit 0 -- a real 0.0 reward, not an infra failure

    reward = 1.0 if tracker.success else 0.0
    REWARD_PATH.write_text(str(reward))
    RESULT_PATH.write_text(
        json.dumps(
            {
                "reward": reward,
                "task_id": task_id,
                "experiment_name": experiment_name,
                "success": tracker.success,
                "pass_count": tracker.pass_count,
                "fail_count": tracker.fail_count,
                "pass_percentage": tracker.pass_percentage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
