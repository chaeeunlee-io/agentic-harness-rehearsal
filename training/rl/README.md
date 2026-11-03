# training/rl — the GRPO trainer

The files the paper's GRPO runs changed relative to upstream
[SkyRL](https://github.com/NovaSky-AI/SkyRL) at commit
`4bdf9766629dd54850c8885617aec5d483a5ee57`: everything under
`examples/train_integrations/harbor/`, plus `pyproject.toml`, `uv.lock` and `.gitignore`.
`setup/install.sh` clones upstream SkyRL at that commit into `third_party/SkyRL/`, copies these
files over it, and points its `harbor` dependency at the vendored `third_party/harbor`.

## What runs

Every GRPO run enters through one entrypoint,
`examples.train_integrations.harbor.entrypoints.main_harbor_xh`. It builds on `main_harbor.py`
and the `xr/` package:

- `xr/harness_dataset.py`: one row per (task, harness); each row tells the harbor
  `DispatchAgent` which harness to roll out
- `xr/group_trainer.py`: `XRTrainer`
- `xr/advantage.py`: group-relative advantage per (task, harness) group
- `xr/bank.py`, `xr/grouping.py`, `xr/canonical.py`

`harbor_trial_config/default.yaml` is the harbor trial template; `exp grpo` sets the sandbox
backend and output folders on the command line.

## Launching a GRPO run

For MedAgentBench or Reasoning Gym:

    uv run exp tasks -b <benchmark> --split train
    uv run exp sft   -b <benchmark> --pool 3h          # the starting checkpoint (or --init <path>)
    uv run exp grpo  -b <benchmark> --rollout 3h       # or a single harness; --backend <backend>

`exp grpo` renders the entrypoint and every override from `configs/rl.yaml` and the
benchmark config's `rl:` block (band splits shipped at `benchmarks/*/splits/band.json`),
and writes to `runs/rl/<benchmark>/<model>/grpo-<rollout>/` (docs/PATHS.md). GRPO needs 2 GPUs.
The GRPO environment pins `skyrl==0.3.0`, `torch==2.11.0+cu128` and `transformers==5.8.0`
(`uv.lock`), separate from the SFT environment in `setup/sft/`.
