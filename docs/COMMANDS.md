# Run experiments

Run these commands from the repository root. `--backend eidf` uses the Kubernetes sandbox backend used for the original runs; use `--backend docker` on a single machine. Cluster settings come from `.env` (see `env.example`). See [PATHS.md](PATHS.md) for directory locations.

[What each experiment runs](#what-each-experiment-runs) summarises each experiment. Sections 0–8 show the individual commands. To print the full command list for an experiment:

```bash
uv run exp plan <experiment>
```

## What each experiment runs

`uv run exp run <experiment>` writes a self-contained script to `runs/plans/<experiment>.sh` and runs it. For each benchmark, the script runs the required task generation, SFT, GRPO, serving, evaluation, teardown, and finally `exp table <experiment>`. `base` and `teacher` require no training.

### `specialists-vs-3h`

Single-harness specialists vs. 3H on the training harnesses.

```bash
uv run exp run specialists-vs-3h --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, t2-only, fc-only, 3h | - | base, parity-only, t2-only, fc-only, 3h on parity, t2, fc, `test` |
| appworld | test | code-only, fc-only, mini-only, 3h | - | base, code-only, fc-only, mini-only, 3h on code, fc, mini, `test` |
| bfcl_multiturn | test | fc-only, json-only, pycall-only, 3h | - | base, fc-only, json-only, pycall-only, 3h on fc, json, pycall, `test` |
| reasoning_gym | test-hard | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, `test-hard` |

### `heldout-transfer`

The same policies evaluated on the held-out harness.

```bash
uv run exp run heldout-transfer --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, t2-only, fc-only, 3h | - | base, parity-only, t2-only, fc-only, 3h on xml, `test` |
| appworld | test | code-only, fc-only, mini-only, 3h | - | base, code-only, fc-only, mini-only, 3h on xml, `test` |
| bfcl_multiturn | test | fc-only, json-only, pycall-only, 3h | - | base, fc-only, json-only, pycall-only, 3h on xml, `test` |
| bfcl_singleturn | test | fc-only, json-only, t2-only, 3h | - | base, fc-only, json-only, t2-only, 3h on xml, `test` |
| reasoning_gym | test, test-hard | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on aider, `test`; same on `test-hard` |

### `cross-harness-vs-rehearsal`

Cross-harness task supervision vs. rehearsal on the target harness.

```bash
uv run exp run cross-harness-vs-rehearsal --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity+fc, reh-t2, parity+fc+reh-t2, t2-only, 3h | - | base, parity+fc, reh-t2, parity+fc+reh-t2, t2-only, 3h on parity, t2, fc, `test` |
| reasoning_gym | test | miniswe+goose, reh-t2, miniswe+goose+reh-t2, t2-only, 3h | - | base, miniswe+goose, reh-t2, miniswe+goose+reh-t2, t2-only, 3h on t2, miniswe, goose, `test` |
| bfcl_multiturn | test | json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h | - | base, json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h on fc, json, pycall, `test` |

### `grpo-3h-vs-single`

3H GRPO vs. single-harness GRPO, both starting from the 3H SFT checkpoint.

```bash
uv run exp run grpo-3h-vs-single --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | 3h | 3h, parity | grpo-3h, grpo-parity on parity, t2, fc, xml, `test` |
| reasoning_gym | test, train, test-hard | 3h | 3h, miniswe | grpo-3h, grpo-miniswe on t2, miniswe, goose, aider, `test`; same on `test-hard` |

### `base-and-teacher`

Base model and teacher, evaluated per harness.

```bash
uv run exp run base-and-teacher --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | - | - | base, teacher on parity, t2, fc, `test` |
| appworld | test | - | - | base, teacher on code, fc, mini, `test` |
| bfcl_multiturn | test | - | - | base, teacher on fc, json, pycall, `test` |
| bfcl_singleturn | test | - | - | base, teacher on fc, json, t2, `test` |
| reasoning_gym | test, test-hard | - | - | base, teacher on t2, miniswe, goose, `test`; teacher on the same harnesses for `test-hard` |

### `matched-volume-mab`

MedAgentBench 3H at matched data volume.

```bash
uv run exp run matched-volume-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, fc-only, 3h-n108, 3h-n108-s2, 3h-n108-s3, 3h-n108-cov, 3h-n150, 3h-n150-cov, 3h | - | all listed settings on parity, t2, fc, `test` |

### `specialists-vs-3h-bfclst-rgeasy`

Single-harness specialists vs. 3H on BFCL single-turn and Reasoning Gym easy.

```bash
uv run exp run specialists-vs-3h-bfclst-rgeasy --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_singleturn | test | fc-only, json-only, t2-only, 3h | - | base, fc-only, json-only, t2-only, 3h on fc, json, t2, `test` |
| reasoning_gym | test | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, `test` |

### `all-conditions-mab`

All MedAgentBench SFT and GRPO conditions.

```bash
uv run exp run all-conditions-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | parity-only, t2-only, fc-only, 3h, parity+fc, reh-t2, parity+fc+reh-t2, parity+fc+reh-t2-32b, reh-t2-fail, parity+t2, reh-fc, parity+t2+reh-fc | 3h, parity | base, all listed SFT settings, grpo-3h and grpo-parity on parity, t2, fc, xml, `test` |

### `gemma-mab`

Gemma-3-4B-it replication with SFT and GRPO.

```bash
uv run exp run gemma-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | parity-only, t2-only, fc-only, reh-t2, reh-t2-fail, parity+reh-t2, parity+t2, reh-fc-gemma, parity+t2+reh-fc-gemma, 3h | 3h | base, all listed SFT settings, and grpo-3h on parity, t2, xml, and separately fc, `test` |

All settings in this experiment use `gemma-3-4b-it`.

### `gemma-rg-easy`

Gemma-3-4B-it SFT replication on Reasoning Gym easy.

```bash
uv run exp run gemma-rg-easy --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| reasoning_gym | test | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, aider, `test` |

All settings in this experiment use `gemma-3-4b-it`.

### `rehearsal-all-harnesses-bfclmt`

Rehearsal on each BFCL multi-turn training harness.

```bash
uv run exp run rehearsal-all-harnesses-bfclmt --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_multiturn | test | json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h | - | base, json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h on fc, json, pycall, `test` |

### `heldout-rehearsal-bfclmt`

Rehearsal on the held-out harness, which is absent from the 3H dataset.

```bash
uv run exp run heldout-rehearsal-bfclmt --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_multiturn | test | 3h, 3h+reh-xml | - | 3h, 3h+reh-xml on fc, json, pycall, `test` |

### Checkpoints shared between experiments

Each experiment writes SFT checkpoints to `runs/checkpoints/<benchmark>/<model>/<pool>/` and GRPO runs to `runs/rl/<benchmark>/<model>/grpo-<rollout>/`.

If another experiment needs the same completed checkpoint, `exp sft` or `exp grpo` reuses it. Pass `--force` to retrain.

| benchmark | model | checkpoint | trained by |
|---|---|---|---|
| appworld | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft code-only | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft mini-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt, heldout-rehearsal-bfclmt |
| bfcl_multiturn | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json+pycall | cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json+pycall+reh-fc | cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft pycall-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft reh-fc | cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt |
| bfcl_singleturn | qwen3-4b | sft 3h | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft fc-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft json-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft t2-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| medagentbench | qwen3-4b | grpo 3h | grpo-3h-vs-single, all-conditions-mab |
| medagentbench | qwen3-4b | grpo parity | grpo-3h-vs-single, all-conditions-mab |
| medagentbench | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, grpo-3h-vs-single, matched-volume-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer, matched-volume-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+fc | cross-harness-vs-rehearsal, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+fc+reh-t2 | cross-harness-vs-rehearsal, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity-only | specialists-vs-3h, heldout-transfer, matched-volume-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft reh-t2 | cross-harness-vs-rehearsal, all-conditions-mab |
| medagentbench | qwen3-4b | sft t2-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, all-conditions-mab |
| reasoning_gym | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, grpo-3h-vs-single, specialists-vs-3h-bfclst-rgeasy |
| reasoning_gym | qwen3-4b | sft goose-only | specialists-vs-3h, heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| reasoning_gym | qwen3-4b | sft miniswe-only | specialists-vs-3h, heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| reasoning_gym | qwen3-4b | sft t2-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, specialists-vs-3h-bfclst-rgeasy |

## 0. Session setup

For each new shell or tmux window, start tmux first and then source the session setup inside it:

```bash
cd ${REPO}
. setup/eidf/session.sh
```

On EIDF this loads `.env` and sets `PATH`, the uv cache, `KUBECONFIG`, and `POD_IP`. See `setup/eidf/README.md`.

The serve/stop pattern used below is:

```bash
serve () {
  setsid uv run exp serve -b "$1" --model qwen3-4b "${@:2}" --port 8000 & SERVER=$!
  uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
}

down () {
  uv run exp stop --pid $SERVER
}

EV="--backend eidf --api-base http://127.0.0.1:8000/v1 --sandbox-api-base http://$POD_IP:8000/v1 --concurrency 16"
```

`setsid` gives the server its own process group so `exp stop` can clean it up. `exp wait --pid` exits if the server crashes, and `exp stop` waits for the GPUs to become free. The GPU smoke test uses the same pattern.

These examples use `--concurrency 16`. Without it, each benchmark uses its configured value: 10 for MedAgentBench, Reasoning Gym, and BFCL single-turn; 18 for BFCL multi-turn; and 8 for AppWorld.

Run long jobs inside tmux. Avoid SIGKILL, which can leave sandbox jobs behind.

## 1. Task directories

Generate each required benchmark split once:

```bash
uv run exp tasks -b medagentbench   --split test
uv run exp tasks -b reasoning_gym   --split test
uv run exp tasks -b reasoning_gym   --split test-hard
uv run exp tasks -b bfcl_singleturn --split test
uv run exp tasks -b bfcl_multiturn  --split test
uv run exp tasks -b appworld        --split test
```

## 2. No training

Base policy on all harnesses, including held out:

```bash
serve medagentbench --pool base
uv run exp eval -b medagentbench --model qwen3-4b --pool base --harness all --split test $EV
down
```

Teacher evaluation uses Qwen3-32B with tensor parallelism 2, as configured in `configs/models.yaml`, so it needs two GPUs:

```bash
setsid uv run exp serve -b medagentbench --model qwen3-4b --teacher --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp eval -b medagentbench --model qwen3-4b --teacher --harness all --split test $EV
down
```

## 3. Supervised training

Available datasets by benchmark (`uv run exp list pools` is authoritative):

| benchmark | single-harness | multi | rehearsal: cross-task, rehearsal-only, both |
|---|---|---|---|
| medagentbench | `parity-only` `t2-only` `fc-only` | `3h` | `parity+fc` `reh-t2` `parity+fc+reh-t2` |
| reasoning_gym | `t2-only` `miniswe-only` `goose-only` | `3h` | `miniswe+goose` `reh-t2` `miniswe+goose+reh-t2` |
| bfcl_singleturn | `fc-only` `json-only` `t2-only` | `3h` | — |
| bfcl_multiturn | `fc-only` `json-only` `pycall-only` | `3h` | `json+pycall` `reh-fc` `json+pycall+reh-fc` |
| appworld | `code-only` `fc-only` `mini-only` | `3h` | — |

Additional settings, including volume-matched datasets, MedAgentBench fc-target rehearsal, rehearsal variants, BFCL multi-turn `3h+reh-xml`, and AppWorld `3h-matched`, are documented in each benchmark README.

Example:

```bash
B=medagentbench
for P in parity-only t2-only fc-only 3h parity+fc reh-t2 parity+fc+reh-t2; do
  uv run exp sft -b $B --model qwen3-4b --pool $P --wandb-run ${B}-$P
done
```

Checkpoints are written to:

```text
runs/checkpoints/<benchmark>/qwen3-4b/<pool>/
```

## 4. Supervised evaluation

Evaluate one trained setting at a time:

```bash
B=medagentbench
for P in base parity-only t2-only fc-only 3h parity+fc reh-t2 parity+fc+reh-t2; do
  serve $B --pool $P
  uv run exp eval -b $B --model qwen3-4b --pool $P --harness all --split test $EV
  down
done
```

`--harness all` evaluates the three training harnesses and the held-out harness. Use `heldout`, `trained`, or a specific harness name to select a subset. Add `--checkpoint <path>` to `serve` to serve a checkpoint from another location.

Reasoning Gym is evaluated on both tiers:

```bash
serve reasoning_gym --pool 3h
for S in test test-hard; do
  uv run exp eval -b reasoning_gym --model qwen3-4b --pool 3h --harness all --split $S $EV
done
down
```

Volume-matched settings are ordinary SFT datasets:

```bash
for P in 3h-n108 3h-n150 3h-n108-cov 3h-n150-cov 3h-n108-s2 3h-n108-s3; do
  uv run exp sft -b medagentbench --model qwen3-4b --pool $P
  serve medagentbench --pool $P
  uv run exp eval -b medagentbench --model qwen3-4b --pool $P --harness all --split test $EV
  down
done
```

## 5. Reinforcement learning

By default, `exp grpo` starts from the model's 3H SFT checkpoint at:

```text
runs/checkpoints/<benchmark>/<model>/3h/final
```

Pass `--init` to use another checkpoint.

```bash
uv run exp grpo -b medagentbench --model qwen3-4b --rollout 3h      --backend eidf --run-name mab-grpo-3h
uv run exp grpo -b medagentbench --model qwen3-4b --rollout parity  --backend eidf --run-name mab-grpo-parity
uv run exp grpo -b reasoning_gym --model qwen3-4b --rollout 3h      --backend eidf --run-name rg-grpo-3h
uv run exp grpo -b reasoning_gym --model qwen3-4b --rollout miniswe --backend eidf --run-name rg-grpo-miniswe
```

GRPO uses two GPUs with TP=2 and colocated vLLM, while also running the sandbox backend. `exp grpo` adds the repository root to `PYTHONPATH`.

Evaluate RL checkpoints with:

```bash
for RL in 3h parity; do
  serve medagentbench --rl $RL
  uv run exp eval -b medagentbench --model qwen3-4b --rl $RL --harness all --split test $EV
  down
done
```

Exports are written to:

```text
runs/rl/<benchmark>/<model>/grpo-<rollout>/exports/global_step_<n>/policy/
```

Published `grpo-<rollout>` results use the final checkpoint. `grpo-<rollout>-step5` uses `global_step_5`, served with `--rl-step 5`.

## 6. Building datasets from scratch

Only needed when rebuilding the distillation datasets.

Collect teacher traces first, then rehearsal traces, then build the datasets. Trace jobs are stored under `runs/jobs/<benchmark>/<job>` and datasets under `runs/pools/<benchmark>/<setting>.jsonl`. Each benchmark README gives the full sequence.

Example for MedAgentBench:

```bash
uv run exp tasks -b medagentbench --split train

setsid uv run exp serve -b medagentbench --model qwen3-4b --teacher --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER

uv run exp collect -b medagentbench --backend eidf \
  --api-base http://127.0.0.1:8000/v1 \
  --sandbox-api-base http://$POD_IP:8000/v1

down

uv run exp tasks -b medagentbench --split rehearsal-t2

serve medagentbench --pool 3h
uv run exp collect -b medagentbench --rehearsal --job rehearsal-t2 $EV
down

uv run exp build-pool -b medagentbench

uv run exp sft -b medagentbench --model qwen3-4b --pool 3h \
  --data runs/pools/medagentbench/3h.jsonl
```

## 7. Scoring and checks

```bash
uv run exp table <experiment>
uv run exp table <experiment> --shipped
uv run exp table <experiment> --format cells --out mine.csv
```

The first reads local runs, the second reads the shipped results, and the third writes local results using the same schema as `results/cells.csv`.

## 8. Gemma replication

Use the same commands with:

```bash
--model gemma-3-4b-it
```

Gemma experiments cover MedAgentBench and Reasoning Gym. Training and serving use the Gemma-specific chat templates defined in the benchmark configs. `exp grpo --model gemma-3-4b-it` starts from Gemma's 3H SFT checkpoint.

For MedAgentBench, Gemma uses different serving settings for `fc` than for `parity`, `t2`, and `xml`. Start separate servers with `exp serve --harness fc` and `exp serve --harness parity`. `exp run gemma-mab` handles this automatically.

The `serve` helper above fixes `--model qwen3-4b`, so call `exp serve` directly for Gemma.

## Notes

- `--concurrency 16` uses roughly 32 CPUs of sandbox quota; leave at least 20 CPUs free in the namespace.
- One MedAgentBench harness at `n=4` is about 244 trials and takes roughly 40 minutes at concurrency 10.
- Sandboxes reach the model server through `$POD_IP`, not `127.0.0.1`.
- Run `down` between settings. A leftover vLLM process can hold 72 GB of GPU memory and prevent the next server from starting.
- `--dry-run` prints the underlying command for `tasks`, `serve`, `sft`, `eval`, `collect`, `build-pool`, `oracle`, and `grpo`. `exp run --dry-run` only writes the generated script.
