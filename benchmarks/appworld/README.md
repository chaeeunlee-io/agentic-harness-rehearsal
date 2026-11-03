# AppWorld

Training harnesses `code`, `fc`, `mini`; held out `xml`. Evaluation: 4 attempt(s) per task (layer n4-T0.7), task-level mean, bootstrap over scenarios. Serving: max-model-len 32768, tool-call parser hermes, no reasoning parser. Ran on cluster 1.

- The bootstrap resamples AppWorld's 56 scenarios, not the 168 tasks.
- AppWorld's data package is licensed: install it with the appworld CLI (`appworld install && appworld download data`); tasks run in the image built by `tasks/build_appworld_image.sh <image>`; set `APPWORLD_IMAGE=<image>` in `.env` before `exp tasks -b appworld`.

Configured in `configs/benchmarks/appworld.yaml`. Reported numbers: `results/cells.csv` rows with `benchmark=appworld`.

## Layout

| directory | contents |
|---|---|
| `harnesses/` | the agents for every harness and the `DispatchAgent` that picks one per trial (the `import_path` in the config): `DESIGN.md`, `appworld_code_agent.py`, `appworld_mini_swe_agent.py`, `appworld_mswe.yaml`, `appworld_mswe_env.py`, `appworld_protocols.py`, `aw_env_init.py`, `aw_env_save.py`, `harness_dispatch.py`, `run_appworld_code_agent.py` |
| `tasks/` | the task generator (harbor adapter, template, vendored upstream data); run through `exp tasks`: `README.md`, `build_appworld_image.sh`, `pyproject.toml`, `src/` |
| `splits/` | the test task list: `test.json` |
| `pools/` | one SFT pool per setting, `<setting>.jsonl`: 5 files, listed below |
| `build/` | the script that builds the pools from exp's runs: `build_pools.py`, `common.py` |

Splits: `test` (168 tasks, `splits/test.json`); `train` (whole task directory).

## Settings

Each setting is one pool file, one `--pool` value, and one `arm` in `results/cells.csv`. Names follow the grammar in `results/README.md` (`<h>-only`, `3h`, `<h1>+<h2>`, `reh-<h>`, ...).

| setting | paper | rows | harness mix | built by | reported for |
|---|---|---|---|---|---|
| `base` | Base | — | — | — | qwen3-4b |
| `code-only` | code-only | 74 | code 74 | `build_pools.py` | qwen3-4b |
| `fc-only` | fc-only | 45 | fc 45 | `build_pools.py` | qwen3-4b |
| `mini-only` | mini-swe-only | 89 | mini 89 | `build_pools.py` | qwen3-4b |
| `3h` | 3H | 208 | mini 89, code 74, fc 45 | `build_pools.py` | qwen3-4b |
| `3h-matched` | 3H (volume-matched) | 74 | mini 32, code 26, fc 16 | no builder shipped | qwen3-4b |

Evaluated without a pool of their own: `teacher` (qwen3-32b).

## Run

```bash
uv run exp tasks -b appworld --split test
uv run exp sft   -b appworld --pool 3h
setsid uv run exp serve -b appworld --pool 3h --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp eval  -b appworld --pool 3h --harness all
uv run exp stop --pid $SERVER
```

## Building the pools from scratch

The shipped pools are what the paper trained on; these commands rebuild them from new runs. Traces land in `runs/jobs/appworld/<job>`, pools in `runs/pools/appworld/<setting>.jsonl` (train on one with `uv run exp sft -b appworld --pool <setting> --data <file>`).

```bash
uv run exp tasks -b appworld --split train
setsid uv run exp serve -b appworld --teacher --port 8000 & SERVER=$!   # the qwen3-32b teacher (2 GPUs)
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp collect -b appworld                         # teacher-code, teacher-fc, teacher-mini
uv run exp stop --pid $SERVER
uv run exp build-pool -b appworld                      # every setting below, in order
uv run exp build-pool -b appworld --pool <setting>      # or one
```

Which script writes which setting is in the table above (config `build:`); each script's docstring describes its selection rule.
