# AppWorld → Harbor adapter

Adapter for [AppWorld](https://github.com/StonyBrookNLP/appworld) (ACL'24 Best
Resource Paper): 9 simulated everyday apps, 457 APIs, tasks solved by writing
**stateful multi-step Python** against them. Generate tasks with
`uv run exp tasks -b appworld --split test|train`, which runs this package's
`appworld-bench` CLI. The harnesses that run on these tasks (`code`, `fc`, `mini`
trained; `xml` held out) and why the action space is the same across them are described in
[../harnesses/DESIGN.md](../harnesses/DESIGN.md).

## Status

| Piece | State |
|---|---|
| Task generation (`src/appworld_adapter/`, `appworld-bench` CLI) | **Working**: train 90, dev 57, test_normal 168 (`exp tasks` uses train and test_normal) |
| Container image | build your own: `build_appworld_image.sh <image>`, then `APPWORLD_IMAGE=<image>` in `.env` (the as-run image was in a private registry and is not published) |
| Verifier (`task-template/tests/`) | **Working**, rule-based, writes `/logs/verifier/reward.txt` |
| Oracle solution (`task-template/solution/solve.sh`) | **Verified**: reward 1.0, 2/2 assertions; check on your setup with `uv run exp oracle -b appworld --split train` |
| Harnesses | `code`, `fc`, `xml` (`../harnesses/appworld_code_agent.py`) and `mini` (`../harnesses/appworld_mini_swe_agent.py`), selected by `../harnesses/harness_dispatch.py` |

## One container: the agent/verifier split goes through disk

Harbor's `EIDFEnvironment` runs exactly one container (no docker-compose, no
sidecars), and AppWorld fits that. Its agent/verifier split is done **entirely
through disk**, not a live connection:

1. The agent process opens `AppWorld(task_id, experiment_name=X)`, calls
   `.execute(code)` (stateful REPL), then `.save_logs()` / `.close()` — which
   persists world state + logs under
   `$APPWORLD_ROOT/experiments/outputs/X/`.
2. The verifier process **independently** calls
   `evaluate_task(task_id, experiment_name=X)`, which re-reads that on-disk
   state fresh and re-runs the task's `ground_truth/evaluation.py` assertions.

No HTTP, no MCP, no server, no `entrypoint.sh` background process, no port.
(The `mini` harness alone starts AppWorld's own environment server inside the same
container, because mini-swe-agent runs each action as a separate process; it saves the
world to the same place on disk, so the verifier side is unchanged. See
`../harnesses/DESIGN.md`.)
Single container, and the only per-task input anything needs is
`APPWORLD_TASK_ID`, delivered via `task.toml`'s `[environment].env` (harbor's
`BaseEnvironment._maybe_resolve_task_env` merges that into **every** `exec()`
on this backend, agent and verifier alike).

## Reward is rule-based, with no LLM anywhere

Verified by reading the real generated `ground_truth/evaluation.py` files and
running the chain live: evaluation is Python assertions comparing final
DB/model state (and the submitted answer) against ground truth — e.g.
`test.answer(predicted, ground_truth)` plus
`test.case(models.changed_model_names(), "is_falsy")` for
no-collateral-damage. `tests/evaluate.py` maps this to a binary reward using
`tracker.success` (AppWorld's own TGC definition — *not* `pass_percentage`,
which would hand out partial credit for a task whose answer was wrong but
whose "no side effects" assertion happened to pass).

No LLM judge and no user-simulator LLM is involved: the "world" is
deterministic simulated data, not a live LLM persona.

## Checking the adapter by hand

```bash
# 1. Environment (Python 3.11+ REQUIRED -- appworld uses typing.Self, which
#    3.10 lacks; hit this live).
uv venv --python 3.11 /tmp/appworld-venv
source /tmp/appworld-venv/bin/activate
uv pip install "appworld==0.1.3.post1"     # see version note below

export APPWORLD_ROOT=/tmp/appworld-root && mkdir -p $APPWORLD_ROOT && cd $APPWORLD_ROOT
appworld install        # unpacks bundled engine + tests, no network
appworld download data  # ~15s, no account/API key/gating

# 2. Build the image (no GPU needed), then set APPWORLD_IMAGE=<image> in .env
benchmarks/appworld/tasks/build_appworld_image.sh <image>   # docker, or kaniko on EIDF (setup/eidf/kaniko_build.sh)

# 3. Generate Harbor tasks (every task names $APPWORLD_IMAGE)
uv run exp tasks -b appworld --split train

# 4. No-GPU check of the full chain: the oracle runs each task's reference solution
uv run exp oracle -b appworld --split train --backend eidf   # or --backend docker

# 5. Verify the oracle chain locally (no cluster needed)
APPWORLD_TASK_ID=82e2fac_1 APPWORLD_EXPERIMENT_NAME=oracle_local python3 - <<'EOF'
import os; from pathlib import Path
from appworld import AppWorld, evaluate_task
from appworld.common.path_store import path_store
tid, exp = os.environ["APPWORLD_TASK_ID"], os.environ["APPWORLD_EXPERIMENT_NAME"]
src = (Path(path_store.data)/"tasks"/tid/"ground_truth"/"compiled_solution.py").read_text()
w = AppWorld(task_id=tid, experiment_name=exp)
try: print(w.execute(src + "\n\nsolution(apis, requester)\n"))
finally: w.save_logs(); w.close()
t = evaluate_task(tid, experiment_name=exp, save_report=False)
print("success:", t.success, t.pass_count, t.fail_count)   # -> True 2 0
EOF
```

## Gotchas paid for while building this (do not re-discover)

- **Pin `appworld==0.1.3.post1`, not `0.1.3`.** Bare `0.1.3`'s metadata
  declares `typer<0.13.0,>=0.12.3`; `.post1` loosens it to `typer>=0.16.0`
  unbounded. Two *different* build failures came from this: an unpinned
  install resolved a typer that crashed the CLI outright (`TypeError:
  Secondary flag is not valid for non-boolean flag`), and then pinning
  `typer==0.16.1` alongside plain `0.1.3` produced a hard `ResolutionImpossible`.
  Pin the `.post1` package version and let typer/click resolve inside its range.
- **Python 3.11+ is mandatory** — `appworld/environment.py` imports
  `typing.Self`. On 3.10 the package installs fine and then fails at import.
- **Set `APPWORLD_CACHE` explicitly in the image.** `appworld install`
  unpacks a *second* bundle (the tests source) to `$APPWORLD_CACHE`, which
  defaults to a `$HOME`-relative path. kaniko builds under one HOME;
  `EIDFEnvironment.exec()` may run agent/verifier commands as a different
  user with a different HOME. The Dockerfile pins it to `/opt/appworld/.cache`
  so build-time and run-time always agree.
- **`task.toml` must be rendered with `.replace()`, not `.format()`** — the
  `authors = [{ name = ... }]` inline TOML table's braces make `.format()`
  raise `KeyError`.
- **`DatasetConfig`'s task filter is `task_names`, not `task_ids`** — the
  wrong key is silently dropped and you run the FULL dataset.
- **A nop-agent smoke test cannot validate this adapter's real chain.** With
  no agent, no experiment output exists, so the verifier correctly reports
  `from_db_home_path ... does not exist` and scores 0.0. That path IS worth
  testing (it proves env vars + image + verifier invocation work), but only
  the **oracle** run exercises agent-side world creation → disk handoff →
  independent verification. Don't read a green nop run as "the adapter works."

## Split counts (verified live via `load_task_ids`)

| split | tasks |
|---|---|
| train | 90 |
| dev | 57 |
| test_normal | 168 |
| test_challenge | 417 |

Note these are each one higher than the corresponding `data/datasets/*.txt`
line counts, because those files lack trailing newlines — use
`load_task_ids()`, not `wc -l`.

`test_challenge` is excluded from the adapter's default splits (opt in with
`splits="...,test_challenge"`). Only train/dev ship `ground_truth` solutions,
so oracle verification is only meaningful there — that's upstream's
train-test-leakage guard, not an adapter gap.

## Licensing note

AppWorld's protected portion (API implementations, task solutions, evaluation
tests) is Apache-2.0 with an added requirement that public redistribution be
in encrypted form. This adapter **never copies that content into a generated
task directory** — it stays inside the pip package / baked image and is read
in-place by `evaluate_task()` at verify time. Generated task dirs contain only
the instruction text, a task.toml, and a task-id metadata file. `exp tasks`
writes them under `runs/`, which `.gitignore` excludes, so generated tasks are
never committed. Do not "simplify" the adapter by inlining ground-truth
content into task dirs.
