# setup/ — install and checks

| script | purpose |
|---|---|
| `install.sh` | Sets up the repository environments. It syncs the vendored Harbor checkout in `third_party/harbor`, clones SkyRL at `4bdf9766` into `third_party/SkyRL/` and overlays `training/rl/`, installs SkyRL dependencies on Linux, syncs the SFT environment in `sft/` on Linux, and installs the root project. Use `SKYRL_SYNC=1` to force a SkyRL sync, `SKYRL=0` to skip SkyRL, or `SFT=0` to skip the SFT environment. |
| `sft/` | SFT training environment with pinned TRL, Transformers, PyTorch, and Liger Kernel versions. Synced by `install.sh`. |
| `smoke.sh` | CPU-only checks for Harbor imports and CLI commands, all harness `import_path` entries in `configs/benchmarks/*.yaml`, and the SkyRL overlay and entrypoint when its environment is available. |
| `smoke_gpu.sh one-gpu\|two-gpu [benchmarks]` | Runs a small GPU smoke test across every pipeline stage and selected benchmark using the normal `exp` commands. Commands are saved to `runs/smoke/commands-<mode>.sh`. |
| `summarize_jobs.py [jobs folder]` | Summarizes Harbor jobs, including attempts, errors, error types, and mean reward. Defaults to `runs/smoke/jobs`. Smoke tests write the summary to `runs/smoke/jobs-<mode>.tsv`. |
| `check_plans.sh [out dir]` | Renders every experiment's `exp plan`, checks shell syntax, verifies that `exp run --dry-run` produces the same script, and dry-runs each generated `exp` command. |
| `eidf/bootstrap_pod.sh`, `eidf/session.sh` | EIDF setup scripts. `bootstrap_pod.sh` installs pinned uv 0.13.6 and Python 3.13.18 and checks the shared environments. `session.sh` loads `.env` and sets the uv cache, `KUBECONFIG`, `PATH` for `kubectl`, and `POD_IP`. See `eidf/README.md`. |
| `eidf/reap_sandboxes.sh [--delete]` | Lists or deletes sandbox jobs left behind after a crashed EIDF run. |
| `eidf/kaniko_build.sh <name> <ctx> <image>` | Builds and pushes a task image on EIDF using Kaniko rather than Docker. Used by the Reasoning Gym and AppWorld image scripts. |

Pinned versions: Harbor `9233373feb3fce2f4e7351cd75f3af064ff45643` and SkyRL `4bdf9766629dd54850c8885617aec5d483a5ee57`, with the local `training/rl/` overlay.

Harness modules are imported from the repository root (`benchmarks.<b>.harnesses...`), so Harbor and the trainer need `PYTHONPATH=<repo root>`. `exp` sets this automatically for commands it runs.
