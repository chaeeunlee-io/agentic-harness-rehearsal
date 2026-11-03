# Paths and dependencies

Where files live, what each command reads, and what it writes.

## Paths

`exp` finds the repo root from `exp/config.py` and builds all paths from there. There are no machine-specific paths in the code.

Cluster-specific values go in `.env` (copy `env.example`).

Most outputs go under `runs/`, which is gitignored. The few exceptions are listed below.

## Files in the repo

| Path | Contents |
|---|---|
| `configs/experiments.yaml` | experiment definitions |
| `configs/benchmarks/<benchmark>.yaml` | benchmark splits, harnesses, serving settings, evaluation settings, datasets, build steps, and GRPO settings |
| `configs/models.yaml`, `configs/rl.yaml` | model settings, chat templates, and GRPO config |
| `benchmarks/<benchmark>/pools/<setting>.jsonl` | SFT datasets used for training |
| `benchmarks/<benchmark>/splits/` | train and test task lists |
| `benchmarks/<benchmark>/tasks/` | task generators |
| `benchmarks/<benchmark>/harnesses/` | harness implementations |
| `benchmarks/<benchmark>/build/` | scripts for rebuilding datasets from traces |
| `training/templates/` | chat templates |
| `results/cells.csv` | shipped results used by `exp table --shipped` |
| `third_party/harbor/` | vendored Harbor |

## Files written under `runs/`

| Path | Written by | Contents |
|---|---|---|
| `runs/tasks/<benchmark>/<split>/` | `exp tasks` | generated task folders and `.hc-tasks.json` |
| `runs/staging/` | `exp tasks` | temporary clones and generator output before tasks are copied into `runs/tasks/` |
| `runs/checkpoints/<benchmark>/<model>/<pool>/final/` | `exp sft` | SFT checkpoint |
| `runs/rl/<benchmark>/<model>/grpo-<rollout>/` | `exp grpo` | trainer checkpoints, exported models, and rollout trials |
| `runs/serve/` | `exp serve` | a copy of the base model's chat template, written when a setting is served with it |
| `runs/job_configs/<job>.yaml` | `exp eval`, `exp collect` | Harbor job config |
| `runs/jobs/<benchmark>/<job>/` | `exp eval`, `exp collect` | trial outputs, trajectories, and rewards |
| `runs/pools/<benchmark>/<setting>.jsonl` | `exp build-pool` | datasets rebuilt from local traces |
| `runs/plans/<experiment>.sh` | `exp run` | generated experiment script |
| `runs/wandb/` | `exp sft --wandb-run` | local W&B files |
| `runs/smoke/` | `setup/smoke_gpu.sh` | smoke-test outputs |
| `runs/plans-check/` | `setup/check_plans.sh` | checked experiment scripts |

`runs/staging/` can be deleted after task generation.

## `.env`

| Variable | Used for |
|---|---|
| `SHARED_MODEL_STORE` | shared model weights under `<store>/<Hugging Face id>`; if unset, models are downloaded from Hugging Face |
| `SANDBOX_API_BASE` | model-server URL reachable from inside sandboxes |
| `RG_TASK_IMAGE`, `APPWORLD_IMAGE` | prebuilt task images for EIDF; AppWorld always needs one |
| `K8S_NAMESPACE`, `KUEUE_QUEUE`, `EIDF_USER`, `EIDF_GROUPS`, `EIDF_HOME`, `KUBECONFIG` | EIDF setup |
| `WANDB_API_KEY` | optional W&B logging |
| `KANIKO_*` | image builds through `setup/eidf/kaniko_build.sh` |

Build the AppWorld image with:

```bash
benchmarks/appworld/tasks/build_appworld_image.sh <image>
```

## Outside the repo

| What | Where | Why |
|---|---|---|
| Model weights | `SHARED_MODEL_STORE` or `~/.cache/huggingface` | large and often shared between users |
| Python installs and uv cache | `~/.local/share/uv/`, `/tmp/uv_cache` on EIDF, otherwise `~/.cache/uv` | managed by uv |
| vLLM, PyTorch, Ray scratch | `~/.cache/vllm`, `/tmp/ray`, etc. | library caches and temporary files |
| Sandboxes | Docker containers or Kubernetes jobs | where agents run |
| Container images | image registry | task images |
| AppWorld data | location managed by the `appworld` CLI | licensed and not redistributed |

The virtual environments themselves stay inside the clone: `.venv`, `third_party/*/.venv`, and `setup/sft/.venv`.

## Command dependencies

| Step | Needs | Writes |
|---|---|---|
| `exp tasks -b B --split S` | upstream task source; AppWorld also needs its data and `APPWORLD_IMAGE` | `runs/tasks/B/S/` |
| `exp sft -b B --pool P` | `benchmarks/B/pools/P.jsonl` or `--data`, plus the base model | `runs/checkpoints/B/<model>/P/final/` |
| `exp serve -b B --pool P` | the matching checkpoint | running model server |
| `exp eval -b B --pool P --split S` | model server, task directory, sandbox backend | `runs/jobs/B/<job>/` |
| `exp grpo -b B --rollout R` | `runs/checkpoints/B/<model>/3h/final/` and `runs/tasks/B/train/` | `runs/rl/B/<model>/grpo-R/` |
| `exp serve` / `exp eval --rl R` | GRPO export, optionally selected with `--rl-step N` | evaluation job |
| `exp collect -b B` | teacher server and training tasks | `runs/jobs/B/teacher-*/` |
| `exp collect -b B --rehearsal` | 3H SFT server and rehearsal tasks | `runs/jobs/B/rehearsal-*/` |
| `exp build-pool -b B` | teacher traces, plus rehearsal traces where required | `runs/pools/B/` |
| `exp table E` | evaluation jobs; `--shipped` reads `results/cells.csv` instead | table on stdout |

`collect` and `build-pool` are only needed if you want to rebuild the training datasets. Normal experiment runs use the shipped datasets.

## Experiments

Experiments can run in any order. Each script produced by

```bash
uv run exp plan <experiment>
```

contains everything that experiment needs: task generation, training, serving, evaluation, and result tables.

Outputs are shared between experiments. SFT checkpoints live at

```text
runs/checkpoints/<benchmark>/<model>/<pool>/
```

and GRPO runs at

```text
runs/rl/<benchmark>/<model>/grpo-<rollout>/
```

If a completed run with the same inputs already exists, `exp sft` and `exp grpo` reuse it. Successful runs record this in `final/exp_sft.json` or `exp_grpo.json`. Short test runs using options such as `--max-steps` or `--batch-size` are not reused. `--force` retrains.

Do not run two experiments at the same time if they may write the same checkpoint; the shared checkpoints are listed in [COMMANDS.md](COMMANDS.md#checkpoints-shared-between-experiments).

SFT, serving, and evaluation need one GPU. GRPO and the Qwen3-32B teacher need two.
