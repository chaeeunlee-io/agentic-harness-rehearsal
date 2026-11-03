# BFCL single-turn

Training harnesses: `fc`, `json`, `t2`; held out: `xml`. Evaluation uses one greedy attempt per task (`pass1-greedy`), reports the task-level mean, and bootstraps over tasks. Serving uses `max-model-len=32768`, the Hermes tool-call parser, and no reasoning parser.

There is no evaluation with several sampled attempts per task for this benchmark. The runs use `splits/split.json`, whose test set does not overlap with tasks in the SFT dataset.

Configuration: `configs/benchmarks/bfcl_singleturn.yaml`. Published results are in `results/cells.csv` with `benchmark=bfcl_singleturn`.

## Layout

| directory | contents |
|---|---|
| `harnesses/` | the agents for every harness and the `DispatchAgent` that picks one per trial (the `import_path` in the config): `DESIGN.md`, `bfcl_agent.py`, `bfcl_protocols.py`, `harness_dispatch.py`, `run_bfcl_agent.py` |
| `tasks/` | the task generator (harbor adapter, template, vendored upstream data); run through `exp tasks`: `README.md`, `adapter_metadata.json`, `bfcl.yaml`, `bfcl_parity.yaml`, `generate_parity_sample.py`, `parity_experiment.json`, `parity_sample_source_ids.txt`, `pyproject.toml`, `src/`, `uv.lock` |
| `splits/` | test / train task lists: `split.json` |
| `pools/` | one SFT pool per setting, `<setting>.jsonl`: 4 files, listed below |
| `build/` | the script that builds the pools from exp's runs: `build_pools.py`, `common.py` |

Splits: `test` (355 tasks, `splits/split.json` [test]); `train` (1051 tasks, `splits/split.json` [train]).

## Settings

Each setting is one pool file, one `--pool` value, and one `arm` in `results/cells.csv`. Names follow the grammar in `results/README.md` (`<h>-only`, `3h`, `<h1>+<h2>`, `reh-<h>`, ...).

| setting | paper | rows | harness mix | built by | reported for |
|---|---|---|---|---|---|
| `base` | Base | — | — | — | qwen3-4b |
| `fc-only` | fc-only | 929 | fc 929 | `build_pools.py` | qwen3-4b |
| `json-only` | json-only | 901 | json 901 | `build_pools.py` | qwen3-4b |
| `t2-only` | Terminus-2-only | 849 | t2 849 | `build_pools.py` | qwen3-4b |
| `3h` | 3H | 2679 | fc 929, json 901, t2 849 | `build_pools.py` | qwen3-4b |

Evaluated without a pool of their own: `teacher` (qwen3-32b).

## Run

```bash
uv run exp tasks -b bfcl_singleturn --split test
uv run exp sft   -b bfcl_singleturn --pool 3h
setsid uv run exp serve -b bfcl_singleturn --pool 3h --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp eval  -b bfcl_singleturn --pool 3h --harness all
uv run exp stop --pid $SERVER
```

## Building the pools from scratch

The shipped pools are what the paper trained on; these commands rebuild them from new runs. Traces land in `runs/jobs/bfcl_singleturn/<job>`, pools in `runs/pools/bfcl_singleturn/<setting>.jsonl` (train on one with `uv run exp sft -b bfcl_singleturn --pool <setting> --data <file>`).

```bash
uv run exp tasks -b bfcl_singleturn --split train
setsid uv run exp serve -b bfcl_singleturn --teacher --port 8000 & SERVER=$!   # the qwen3-32b teacher (2 GPUs)
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp collect -b bfcl_singleturn                         # teacher-fc, teacher-json, teacher-t2
uv run exp stop --pid $SERVER
uv run exp build-pool -b bfcl_singleturn                      # every setting below, in order
uv run exp build-pool -b bfcl_singleturn --pool <setting>      # or one
```

Which script writes which setting is in the table above (config `build:`); each script's docstring describes its selection rule.
