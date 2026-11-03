#!/bin/bash
# Oracle solution: replays AppWorld's own ground_truth/compiled_solution.py
# through the SAME AppWorld(...)/execute()/close() path a real agent uses, so
# `harbor run -a oracle` end-to-end-verifies the whole chain (env vars ->
# agent-side world open -> code execution -> on-disk experiment output ->
# independent verifier process -> reward.txt) and should score 1.0.
#
# Only train/dev tasks ship ground_truth solutions -- test_normal /
# test_challenge deliberately do not, so oracle verification is only
# meaningful on train/dev tasks (this is upstream's train-test-leakage
# guard, not a gap in this adapter).
set -eu
cd /opt/appworld
python3 - <<'PYEOF'
import os
from pathlib import Path

from appworld import AppWorld
from appworld.common.path_store import path_store

task_id = os.environ["APPWORLD_TASK_ID"]
experiment_name = os.environ.get("APPWORLD_EXPERIMENT_NAME", "harbor_trial")

solution_path = (
    Path(path_store.data) / "tasks" / task_id / "ground_truth" / "compiled_solution.py"
)
if not solution_path.exists():
    raise SystemExit(
        f"No ground_truth solution for {task_id} (expected for test splits): {solution_path}"
    )

# compiled_solution.py defines `def solution(apis, requester)`. Run it inside
# the world's own execution context so its DB writes land in the same session
# state the verifier will later re-read.
source = solution_path.read_text()
world = AppWorld(task_id=task_id, experiment_name=experiment_name)
try:
    world.execute(source + "\n\nsolution(apis, requester)\n")
finally:
    world.save_logs()
    world.close()
PYEOF
