"""Summarize Harbor jobs under a directory.

Reports attempts, errored attempts, error types, and mean reward for each job,
and marks jobs where every attempt errored with `ALL ERRORED`. This is useful
because `harbor run` may exit successfully even when all attempts failed.

Usage:
    python setup/summarize_jobs.py [jobs folder]

Defaults to `runs/smoke/jobs` and prints a tab-separated table, sorted by the
share of errored attempts, highest first.
"""

import json
import sys
from pathlib import Path

jobs_root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/smoke/jobs")

rows = []
for result_file in sorted(jobs_root.rglob("result.json")):
    result = json.loads(result_file.read_text())
    if "n_total_trials" not in result:   # one attempt's result, inside a job folder
        continue
    stats = result.get("stats") or {}
    attempts = result.get("n_total_trials", 0)
    errors = stats.get("n_errored_trials", 0)
    rewards, error_types = [], set()
    for evaluation in (stats.get("evals") or {}).values():
        error_types.update(evaluation.get("exception_stats") or {})
        for metric in evaluation.get("metrics") or []:
            if "mean" in metric:
                rewards.append(metric["mean"])
    job = result_file.parent.relative_to(jobs_root)
    mean_reward = f"{sum(rewards) / len(rewards):.2f}" if rewards else "-"
    flag = "ALL ERRORED" if attempts and errors == attempts else ""
    rows.append((errors / attempts if attempts else 0, str(job), attempts, errors,
                 ",".join(sorted(error_types)) or "-", mean_reward, flag))

print("job\tattempts\terrored\terror types\tmean reward\t")
for _, job, attempts, errors, error_types, mean_reward, flag in sorted(rows, key=lambda r: (-r[0], r[1])):
    print(f"{job}\t{attempts}\t{errors}\t{error_types}\t{mean_reward}\t{flag}")
